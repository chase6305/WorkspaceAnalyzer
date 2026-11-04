"""Marvin M6 single-arm reachability analysis with optional Viser display."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from workspace_analyzer import (
    CartesianConfig,
    KinematicsSolver,
    RobotModel,
    SamplingConfig,
    SamplingStrategy,
    SolverConfig,
    WorkspaceAnalyzer,
    WorkspaceConfig,
)
from workspace_analyzer._arrays import as_numpy as _numpy
from workspace_analyzer._output_paths import validate_output_paths
from workspace_analyzer.presets import (
    default_reference_joints,
    default_robot_urdf,
    require_robot_urdf,
    select_arm_links,
)

DEFAULT_URDF = default_robot_urdf()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--base-link")
    parser.add_argument("--tip-link")
    parser.add_argument("--bounds", type=float, nargs=6)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for sampling, automatic bounds estimation, and IK restarts",
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--strategy", choices=[x.value for x in SamplingStrategy], default="random"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/marvin_left_arm.npz")
    )
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--color-metric",
        choices=(
            "manipulability",
            "minimum_singular_value",
            "isotropy",
            "joint_limit_margin",
        ),
        default="manipulability",
    )
    parser.add_argument("--mode", choices=("joint", "cartesian"), default="joint")
    parser.add_argument("--ik-restarts", type=int, default=4)
    parser.add_argument("--ik-rescue-restarts", type=int, default=16)
    parser.add_argument("--ik-rescue-rounds", type=int, default=3)
    parser.add_argument(
        "--full-pose",
        action="store_true",
        help="require the FK reference orientation in Cartesian mode",
    )
    parser.add_argument(
        "--reference-joints",
        type=float,
        nargs="+",
        help="reference joint vector; FK defines the Cartesian orientation",
    )
    args = parser.parse_args(argv)
    if args.bounds is not None and args.mode != "cartesian":
        parser.error("--bounds requires --mode cartesian")
    try:
        return _run(parser, args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


def _run(parser, args):
    args.output, _, _ = validate_output_paths(output=args.output, inputs=(args.urdf,))
    args.urdf = require_robot_urdf(args.urdf, parser)

    model = RobotModel.from_urdf(args.urdf)
    base, tip = select_arm_links(model, args.arm, args.base_link, args.tip_link)
    solver = KinematicsSolver(
        model,
        base,
        tip,
        SolverConfig(
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            random_seed=args.seed,
        ),
    )
    config = WorkspaceConfig(
        sampling=SamplingConfig(
            strategy=SamplingStrategy(args.strategy),
            num_samples=args.samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    )
    analyzer = WorkspaceAnalyzer(solver, config)
    reference_q = (
        default_reference_joints(solver)
        if args.reference_joints is None
        else np.asarray(args.reference_joints, dtype=float)
    )
    if reference_q.shape != (solver.dof,):
        parser.error(f"--reference-joints requires exactly {solver.dof} values")
    if not np.isfinite(reference_q).all():
        parser.error("--reference-joints must contain finite values")
    if np.any(reference_q < solver.joint_limits[:, 0]) or np.any(
        reference_q > solver.joint_limits[:, 1]
    ):
        parser.error("--reference-joints contains a value outside the URDF limits")
    started = time.perf_counter()
    if args.mode == "cartesian":
        bounds_estimation = None
        if args.bounds is not None:
            bounds = np.asarray(args.bounds).reshape(3, 2)
        else:
            # Estimate an XYZ box from FK before testing targets with IK.
            estimate_config = WorkspaceConfig(
                sampling=SamplingConfig(
                    strategy=SamplingStrategy.RANDOM,
                    num_samples=min(16_384, max(4096, args.samples)),
                    batch_size=args.batch_size,
                    seed=args.seed,
                ),
                compute_jacobians=False,
            )
            estimate = WorkspaceAnalyzer(solver, estimate_config).analyze()
            margin = 0.02
            bounds_estimation = {**estimate.metadata, "margin_m": margin}
            bounds = np.stack(
                (estimate.points.min(0) - margin, estimate.points.max(0) + margin),
                axis=1,
            )
        reference_pose = _numpy(solver.forward(reference_q))
        cartesian_config = CartesianConfig(
            bounds=bounds,
            sampling=config.sampling,
            position_only=not args.full_pose,
            restarts=args.ik_restarts,
            rescue_restarts=args.ik_rescue_restarts,
            rescue_rounds=args.ik_rescue_rounds,
            reference_pose=reference_pose,
            reference_joints=reference_q,
        )
        result = analyzer.analyze_cartesian(cartesian_config)
        result.metadata.update(
            bounds_source="explicit" if args.bounds is not None else "joint_samples",
            bounds_estimation=bounds_estimation,
        )
    else:
        result = analyzer.analyze()
    elapsed = time.perf_counter() - started
    result.save(args.output)

    print(f"URDF: {args.urdf.resolve()}")
    print(
        f"chain: {solver.base_link} -> {solver.tip_link} "
        f"({solver.dof} DoF: {', '.join(solver.joint_names)})"
    )
    print(f"backend: {solver.backend}:{solver.device} ({args.dtype})")
    print(
        f"samples: {len(result.points):,} in {elapsed:.3f}s "
        f"({len(result.points) / elapsed:,.0f} samples/s)"
    )
    print(f"XYZ min: {np.array2string(result.points.min(0), precision=4)}")
    print(f"XYZ max: {np.array2string(result.points.max(0), precision=4)}")
    if result.reachable is not None:
        print(
            f"reachable: {np.count_nonzero(result.reachable):,}/"
            f"{len(result.reachable):,} ({np.mean(result.reachable):.1%})"
        )
        print(f"reference joints: {np.array2string(reference_q, precision=4)}")
        print(f"reference pose:\n{np.array2string(reference_pose, precision=5)}")
    print(f"saved: {args.output.resolve()}")

    if args.viser:
        from workspace_analyzer.visualization import ViserWorkspace

        with ViserWorkspace(
            solver,
            port=args.port,
            label=f"marvin_{args.arm}_arm",
            initial_q=reference_q,
            load_full_robot=True,
        ) as viewer:
            viewer.add_workspace(result, color_metric=args.color_metric)
            if args.mode == "cartesian":
                viewer.configure_cartesian_recompute(analyzer, cartesian_config)
            viewer.wait()


if __name__ == "__main__":
    main()
