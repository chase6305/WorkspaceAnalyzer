"""Analyze robot workspaces, supplied targets, and planes directly from URDF."""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ._arrays import as_numpy as _numpy
from ._output_paths import validate_output_paths
from ._progress import CLIProgress
from .acceptance import AcceptanceConfig, add_acceptance_arguments, apply_acceptance
from .analyzer import CartesianConfig, WorkspaceAnalyzer, WorkspaceConfig
from .cache import ResultCache
from .kinematics import create_solver
from .plane import PlaneConfig, analyze_plane, plane_frame
from .reachability import ReachabilityConfig
from .result_export import save_result_artifacts
from .sampling import SamplingConfig, SamplingStrategy


def build_parser(*, demo=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plane_reachability.py" if demo else "workspace-analyzer",
        usage="%(prog)s [options]" if demo else "%(prog)s URDF [options]",
        description="Scan a plane and export its reachability, quality, and sample map."
        if demo
        else __doc__,
        epilog=(
            "Inspect saved results without IK: "
            "workspace-analyzer inspect RESULT.npz --help"
        ),
    )
    robot = parser.add_argument_group("Robot and backend")
    if demo:
        from .presets import default_robot_urdf

        robot.add_argument("--urdf", type=Path, default=default_robot_urdf())
        robot.add_argument("--arm", choices=("left", "right"), default="left")
    else:
        parser.add_argument("urdf", help="path to the robot URDF")
    robot.add_argument("--base-link")
    robot.add_argument("--tip-link")
    robot.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    robot.add_argument(
        "--device", default="auto", help="Torch: auto, cpu, cuda, or cuda:N"
    )
    robot.add_argument("--dtype", choices=("float32", "float64"), default="float64")

    analysis = parser.add_argument_group("Analysis and sampling")
    analysis.add_argument(
        "--mode",
        choices=("joint", "cartesian", "targets", "plane"),
        default=None,
        help=argparse.SUPPRESS if demo else "analysis mode",
    )
    analysis.add_argument(
        "--targets",
        type=Path,
        help=argparse.SUPPRESS
        if demo
        else "target-mode input: .npy, .npz, or XYZ .csv",
    )
    task = analysis.add_mutually_exclusive_group()
    task.add_argument(
        "--full-pose", action="store_true", help="also constrain TCP rotation"
    )
    task.add_argument(
        "--position-only",
        action="store_false",
        dest="full_pose",
        help="ignore TCP rotation (default)",
    )
    analysis.add_argument(
        "--samples",
        type=int,
        help="number of sampled plane targets; omit for a regular grid"
        if demo
        else "joint, Cartesian, or plane sample count",
    )
    analysis.add_argument("--batch-size", type=int, default=4096)
    analysis.add_argument(
        "--seed", type=int, default=42, help="random seed for sampling and IK restarts"
    )
    analysis.add_argument(
        "--strategy", choices=[item.value for item in SamplingStrategy], default=None
    )
    analysis.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help=argparse.SUPPRESS
        if demo
        else "XYZ intervals; equal endpoints fix an axis for planes, lines, or points",
    )
    add_plane_arguments(parser.add_argument_group("Plane sampling"))

    ik = parser.add_argument_group("IK and configuration quality")
    ik.add_argument("--ik-max-iterations", type=int, default=150)
    ik.add_argument("--ik-restarts", type=int, default=4)
    ik.add_argument("--ik-rescue-restarts", type=int, default=16)
    ik.add_argument("--ik-rescue-rounds", type=int, default=3)
    ik.add_argument("--reference-joints", type=float, nargs="+")
    ik.add_argument(
        "--dexterity-task", choices=("position", "rotation", "pose"), default="position"
    )
    ik.add_argument("--dexterity-weights", type=float, nargs="+")
    ik.add_argument("--min-singular-value", type=float)
    ik.add_argument("--min-isotropy", type=float)
    ik.add_argument("--min-joint-limit-margin", type=float)
    ik.add_argument("--require-full-rank", action="store_true")
    add_acceptance_arguments(parser.add_argument_group("Run acceptance (headless)"))

    output = parser.add_argument_group("Outputs and cache")
    output.add_argument(
        "--output",
        help="write NPZ, JSON and SVG using this .npz/.json stem"
        if demo
        else "save NPZ measurements",
    )
    output.add_argument("--report", type=Path, help="JSON report for any analysis mode")
    output.add_argument("--plot", type=Path, help="SVG plot of plane samples")
    output.add_argument("--cache-dir")
    output.add_argument(
        "--cache-uncompressed",
        action="store_true",
        help="write larger, uncompressed NPZ cache entries for faster local I/O",
    )
    output.add_argument(
        "--progress",
        action="store_true",
        help="print headless computation progress to stderr",
    )

    viewer = parser.add_argument_group("Viser visualization")
    viewer.add_argument("--viser", action="store_true", help="start a browser viewer")
    viewer.add_argument("--port", type=int, default=8080)
    viewer.add_argument(
        "--color-metric",
        choices=(
            "manipulability",
            "minimum_singular_value",
            "isotropy",
            "joint_limit_margin",
        ),
        default="manipulability",
        help=argparse.SUPPRESS
        if demo
        else "joint-workspace metric used for Viser coloring",
    )
    parser.set_defaults(full_pose=False)
    if demo:
        parser.set_defaults(mode="plane", output="outputs/plane_reachability.json")
    return parser


def add_plane_arguments(parser):
    parser.add_argument("--plane", choices=("xy", "xz", "yz"))
    parser.add_argument(
        "--frame",
        help="plane frame: base, world, reference, or a link at zero configuration",
    )
    parser.add_argument(
        "--frame-rpy",
        type=float,
        nargs=3,
        help="rotate the sampling frame at its origin by roll/pitch/yaw (radians)",
    )
    parser.add_argument(
        "--constant",
        type=float,
        help="override the plane center's normal coordinate, in metres",
    )
    parser.add_argument("--center", type=float, nargs=3)
    parser.add_argument("--span", type=float, help="square plane side length in metres")
    parser.add_argument(
        "--resolution",
        type=int,
        help="plane points per axis; total samples = resolution squared",
    )
    parser.add_argument(
        "--rpy",
        type=float,
        nargs=3,
        help="fixed TCP roll/pitch/yaw in the sampling frame, radians",
    )


def plane_config_from_args(args):
    return PlaneConfig(
        plane=args.plane or "xy",
        frame=args.frame or "base",
        center=args.center if args.center is not None else (0, 0, 0),
        constant=args.constant,
        span=args.span if args.span is not None else 0.8,
        resolution=args.resolution if args.resolution is not None else 41,
        rpy=args.rpy if args.rpy is not None else (0, 0, 0),
        frame_rpy=args.frame_rpy if args.frame_rpy is not None else (0, 0, 0),
        sampling=None
        if args.samples is None
        else SamplingConfig(
            SamplingStrategy(args.strategy or "random"),
            args.samples,
            args.batch_size,
            args.seed,
        ),
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "inspect":
        from .inspection import main as inspect_main

        return inspect_main(argv[1:])
    parser = build_parser()
    return run_arguments(parser, parser.parse_args(argv))


def run_arguments(parser, args):
    args.mode = args.mode or ("plane" if args.plane is not None else "joint")
    if args.plot is not None and (
        args.mode != "plane" or args.plot.suffix.lower() != ".svg"
    ):
        parser.error("--plot requires plane mode and an .svg path")
    if args.progress and args.viser:
        parser.error(
            "--progress is for headless runs; --viser displays progress in the browser"
        )
    if args.cache_uncompressed and not args.cache_dir:
        parser.error("--cache-uncompressed requires --cache-dir")
    if args.mode == "cartesian" and args.bounds is None:
        parser.error("--bounds is required in Cartesian mode")
    if args.mode == "targets":
        if args.targets is None:
            parser.error("--targets is required in targets mode")
        if args.samples is not None or args.strategy is not None:
            parser.error(
                "targets mode uses all input rows; "
                "--samples/--strategy are not applicable"
            )
    plane_options = (
        args.plane,
        args.frame,
        args.frame_rpy,
        args.center,
        args.constant,
        args.span,
        args.resolution,
        args.rpy,
    )
    if args.mode != "plane" and any(value is not None for value in plane_options):
        parser.error("plane options require --mode plane")
    if args.mode == "plane":
        if args.resolution is not None and (
            args.samples is not None or args.strategy is not None
        ):
            parser.error(
                "choose --resolution for a grid or --samples with optional --strategy"
            )
        if args.strategy is not None and args.samples is None:
            parser.error("--strategy requires --samples in plane mode")
    if args.bounds is not None and args.mode != "cartesian":
        parser.error("--bounds requires --mode cartesian")
    if args.full_pose and args.mode == "joint":
        parser.error("--full-pose requires a reachability mode")
    if args.mode != "targets" and args.targets is not None:
        parser.error("--targets requires --mode targets")
    if args.mode not in {"targets", "plane"} and (
        args.dexterity_task != "position"
        or args.dexterity_weights is not None
        or args.min_singular_value is not None
        or args.min_isotropy is not None
        or args.min_joint_limit_margin is not None
        or args.require_full_rank
    ):
        parser.error("quality gates require --mode targets or --mode plane")
    try:
        acceptance = AcceptanceConfig(args.min_success_rate, args.min_quality_pass_rate)
        if acceptance.enabled and (args.mode == "joint" or args.viser):
            parser.error("acceptance requires a headless reachability mode")
        if args.min_quality_pass_rate is not None and args.mode not in {
            "targets",
            "plane",
        }:
            parser.error("quality acceptance requires --mode targets or --mode plane")
        args.output, args.report, args.plot = validate_output_paths(
            output=args.output,
            report=args.report,
            plot=args.plot,
            inputs=(args.urdf, args.targets),
        )
        return _run(parser, args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


def entrypoint(run=None) -> None:
    """Console scripts must return an exit status, not an analysis object."""
    try:
        (main if run is None else run)()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None


def _run(parser, args):
    sampling = SamplingConfig(
        SamplingStrategy(args.strategy or "random"),
        args.samples if args.samples is not None else 10_000,
        args.batch_size,
        args.seed,
    )
    solver = create_solver(
        args.urdf,
        base_link=args.base_link,
        tip_link=args.tip_link,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_iterations=args.ik_max_iterations,
        random_seed=args.seed,
    )
    analyzer = WorkspaceAnalyzer(solver, WorkspaceConfig(sampling))
    cache = (
        ResultCache(args.cache_dir, compressed=not args.cache_uncompressed)
        if args.cache_dir
        else None
    )
    reference_q = _reference_joints(parser, args.reference_joints, solver)
    cartesian_config = None
    progress = CLIProgress(args.mode) if args.progress else None
    if progress is not None:
        progress(0.0)
    if args.mode == "plane":
        plane = plane_config_from_args(args)
        plane_frame(
            solver, plane.frame, plane.frame_rpy
        )  # Validate before opening a browser server.
        assessment = _assessment_config(args)
        if args.viser:
            from .visualization import ViserWorkspace

            with ViserWorkspace(
                solver, port=args.port, initial_q=reference_q, load_full_robot=True
            ) as view:
                view.configure_plane_recompute(
                    plane,
                    assessment,
                    cache=cache,
                    output=args.output,
                    report=args.report,
                    plot=args.plot,
                )
                print(
                    f"Plane controls ready: http://localhost:{view.server.get_port()}",
                    flush=True,
                )
                print(
                    "Select the plane and click Compute plane reachability.", flush=True
                )
                view.wait()
                return view.plane_panel.result
        result = analyze_plane(
            solver,
            plane,
            assessment,
            seed=reference_q,
            cache=cache,
            progress_callback=progress,
        )
    elif args.mode == "targets":
        targets, options = _load_targets(args.targets)
        result = analyzer.analyze_targets(
            targets,
            _assessment_config(args),
            seed=reference_q,
            cache=cache,
            progress_callback=progress,
            **options,
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
        result = analyzer.analyze_cartesian(
            cartesian_config, cache=cache, progress_callback=progress
        )
    else:
        result = analyzer.analyze(cache=cache, progress_callback=progress)
    result = apply_acceptance(
        result, AcceptanceConfig(args.min_success_rate, args.min_quality_pass_rate)
    )
    paths = {
        kind: path
        for kind, path in (
            ("npz", args.output),
            ("json", args.report),
            ("svg", args.plot),
        )
        if path is not None
    }
    saved, errors = save_result_artifacts(
        result,
        paths,
        report_format={"plane": "plane", "targets": "metadata"}.get(
            args.mode, "summary"
        ),
    )
    if errors:
        details = "; ".join(f"{paths[kind]}: {error}" for kind, error in errors.items())
        completed = ", ".join(str(path) for path in saved.values()) or "none"
        raise OSError(
            f"{args.mode} calculation completed; save failed: {details}. "
            f"Saved: {completed}"
        )
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
    acceptance = result.metadata.get("acceptance")
    if acceptance is not None:
        print("acceptance: " + ("PASS" if acceptance["passed"] else "FAIL"))
        if not acceptance["passed"]:
            for failure in acceptance["failures"]:
                print(failure, file=sys.stderr)
            raise SystemExit(1)
    if args.viser:
        from .visualization import ViserWorkspace

        with ViserWorkspace(solver, port=args.port, initial_q=reference_q) as view:
            view.add_workspace(result, color_metric=args.color_metric)
            if cartesian_config is not None:
                view.configure_cartesian_recompute(analyzer, cartesian_config)
            view.wait()
    return result


def _assessment_config(args):
    return ReachabilityConfig(
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
    )


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
    try:
        return _read_targets(path)
    except (BadZipFile, EOFError, zlib.error) as exc:
        raise ValueError(f"invalid target file {path}: {exc}") from exc


def _read_targets(path):
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", ndmin=2), {}
    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
        if not isinstance(data, np.ndarray):
            data.close()
            raise ValueError(
                "target .npy file must contain an NPY array, not an NPZ archive"
            )
        return data, {}
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        if isinstance(archive, np.ndarray):
            raise ValueError(
                "target .npz file must contain an NPZ archive, not an NPY array"
            )
        with archive:
            if "targets" not in archive.files:
                raise ValueError("target NPZ must contain a 'targets' array")
            options = {
                name: archive[name]
                for name in ("weights", "base_from_targets")
                if name in archive.files
            }
            return archive["targets"], options
    raise ValueError("target input must be .npy, .npz, or headerless XYZ .csv")


if __name__ == "__main__":
    entrypoint()
