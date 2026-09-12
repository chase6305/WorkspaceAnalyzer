"""Command-line interface for joint and Cartesian workspace analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyzer import CartesianConfig, WorkspaceAnalyzer, WorkspaceConfig
from .cache import ResultCache
from .kinematics import create_solver
from .reachability import ReachabilityConfig
from .sampling import SamplingConfig, SamplingStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf")
    parser.add_argument("--base-link")
    parser.add_argument("--tip-link")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--mode", choices=("joint", "cartesian", "targets"), default="joint"
    )
    parser.add_argument(
        "--targets", type=Path, help="target-mode input: .npy, .npz, or XYZ .csv"
    )
    parser.add_argument(
        "--report", type=Path, help="target-mode JSON assessment report"
    )
    parser.add_argument(
        "--dexterity-task", choices=("position", "rotation", "pose"), default="position"
    )
    parser.add_argument("--dexterity-weights", type=float, nargs="+")
    parser.add_argument("--min-singular-value", type=float)
    parser.add_argument("--min-isotropy", type=float)
    parser.add_argument("--min-joint-limit-margin", type=float)
    parser.add_argument("--require-full-rank", action="store_true")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=[item.value for item in SamplingStrategy],
        default="random",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="XYZ intervals; equal endpoints fix an axis for planes, lines, or points",
    )
    parser.add_argument("--full-pose", action="store_true")
    parser.add_argument("--ik-restarts", type=int, default=4)
    parser.add_argument("--ik-rescue-restarts", type=int, default=16)
    parser.add_argument("--ik-rescue-rounds", type=int, default=3)
    parser.add_argument("--reference-joints", type=float, nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--cache-uncompressed",
        action="store_true",
        help="write larger, uncompressed NPZ cache entries for faster local I/O",
    )
    parser.add_argument(
        "--color-metric",
        choices=(
            "manipulability",
            "minimum_singular_value",
            "isotropy",
            "joint_limit_margin",
        ),
        default="manipulability",
        help="joint-workspace metric used for Viser coloring",
    )
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cache_uncompressed and not args.cache_dir:
        parser.error("--cache-uncompressed requires --cache-dir")
    if args.mode == "cartesian" and args.bounds is None:
        parser.error("--bounds is required in Cartesian mode")
    if args.mode == "targets" and args.targets is None:
        parser.error("--targets is required in targets mode")
    if args.mode != "targets" and (
        args.targets is not None
        or args.report is not None
        or args.dexterity_task != "position"
        or args.dexterity_weights is not None
        or args.min_singular_value is not None
        or args.min_isotropy is not None
        or args.min_joint_limit_margin is not None
        or args.require_full_rank
    ):
        parser.error(
            "target input, assessment reports, and quality gates require --mode targets"
        )
    try:
        return _run(parser, args)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))


def entrypoint() -> None:
    """Console scripts must return an exit status, not an analysis object."""
    main()


def _run(parser, args):
    sampling = SamplingConfig(
        SamplingStrategy(args.strategy), args.samples, args.batch_size, args.seed
    )
    solver = create_solver(
        args.urdf,
        base_link=args.base_link,
        tip_link=args.tip_link,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
    )
    analyzer = WorkspaceAnalyzer(solver, WorkspaceConfig(sampling))
    cache = (
        ResultCache(args.cache_dir, compressed=not args.cache_uncompressed)
        if args.cache_dir
        else None
    )
    reference_q = _reference_joints(parser, args.reference_joints, solver)
    cartesian_config = None
    if args.mode == "targets":
        targets, options = _load_targets(args.targets)
        result = analyzer.analyze_targets(
            targets,
            ReachabilityConfig(
                position_only=not args.full_pose,
                batch_size=args.batch_size,
                restarts=args.ik_restarts,
                rescue_restarts=args.ik_rescue_restarts,
                rescue_rounds=args.ik_rescue_rounds,
                random_seed=args.seed,
                dexterity_task=args.dexterity_task,
                dexterity_weights=args.dexterity_weights,
                minimum_singular_value=args.min_singular_value,
                minimum_isotropy=args.min_isotropy,
                minimum_joint_limit_margin=args.min_joint_limit_margin,
                require_full_rank=args.require_full_rank,
            ),
            seed=reference_q,
            cache=cache,
            **options,
        )
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(result.metadata, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    elif args.mode == "cartesian":
        cartesian_config = CartesianConfig(
            bounds=np.asarray(args.bounds).reshape(3, 2),
            sampling=sampling,
            position_only=not args.full_pose,
            restarts=args.ik_restarts,
            rescue_restarts=args.ik_rescue_restarts,
            rescue_rounds=args.ik_rescue_rounds,
            reference_pose=_numpy(solver.forward(reference_q)),
            reference_joints=reference_q,
        )
        result = analyzer.analyze_cartesian(cartesian_config, cache=cache)
    else:
        result = analyzer.analyze(cache=cache)
    if args.output:
        result.save(args.output)
    backend_label = f"{solver.backend}:{solver.device}"
    print(f"{len(result.points)} samples | {solver.dof} DoF | {backend_label}")
    print(f"bounds: {result.points.min(0)} .. {result.points.max(0)}")
    if result.reachable is not None:
        print(f"reachable: {np.mean(result.reachable):.1%}")
        failure_stats = result.metadata.get("residual_summary", {}).get("unreachable")
        if failure_stats is not None:
            print(
                "unreachable residual: "
                f"median={failure_stats['median']:.3g}, "
                f"p95={failure_stats['p95']:.3g}"
            )
    if "assessment" in result.metadata:
        assessment = result.metadata["assessment"]
        print(f"quality accepted: {assessment['quality_pass_rate']:.1%}")
        print(
            f"weighted quality accepted: {assessment['weighted_quality_pass_rate']:.1%}"
        )
    if "cache_hit" in result.metadata:
        print("cache: hit" if result.metadata["cache_hit"] else "cache: miss")
    if args.viser:
        from .visualization import ViserWorkspace

        view = ViserWorkspace(solver, port=args.port, initial_q=reference_q)
        view.add_workspace(result, color_metric=args.color_metric)
        if cartesian_config is not None:
            view.configure_cartesian_recompute(analyzer, cartesian_config)
        view.wait()
    return result


def _reference_joints(parser, values, solver):
    joints = solver.joint_limits.mean(axis=1) if values is None else np.asarray(values)
    if joints.shape != (solver.dof,):
        parser.error(f"--reference-joints requires exactly {solver.dof} values")
    if not np.isfinite(joints).all():
        parser.error("--reference-joints must contain only finite values")
    if np.any(joints < solver.joint_limits[:, 0]) or np.any(
        joints > solver.joint_limits[:, 1]
    ):
        parser.error("--reference-joints contains a value outside the URDF limits")
    return joints


def _load_targets(path):
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", ndmin=2), {}
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False), {}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "targets" not in archive.files:
                raise ValueError("target NPZ must contain a 'targets' array")
            options = {
                name: archive[name]
                for name in ("weights", "base_from_targets")
                if name in archive.files
            }
            return archive["targets"], options
    raise ValueError("target input must be .npy, .npz, or headerless XYZ .csv")


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


if __name__ == "__main__":
    entrypoint()
