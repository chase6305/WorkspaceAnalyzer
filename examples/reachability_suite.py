"""Run three reachability experiments, then inspect them in one optional viewer.

All reports are saved before the browser opens. The experiment selector switches
between joint sampling, position IK and fixed-orientation IK without restarting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
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
from workspace_analyzer._atomic import atomic_write_text
from workspace_analyzer._output_paths import validate_artifact_paths
from workspace_analyzer.presets import (
    default_reference_joints,
    require_robot_urdf,
    select_arm_links,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--base-link")
    parser.add_argument("--tip-link")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for sampling and IK restarts"
    )
    parser.add_argument(
        "--strategy",
        choices=[item.value for item in SamplingStrategy],
        default="random",
    )
    parser.add_argument("--bounds", type=float, nargs=6)
    parser.add_argument("--ik-max-iterations", type=int, default=150)
    parser.add_argument("--ik-restarts", type=int, default=4)
    parser.add_argument("--ik-rescue-restarts", type=int, default=16)
    parser.add_argument("--ik-rescue-rounds", type=int, default=3)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/reachability_suite")
    )
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


def _run(args, parser):
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise ValueError(f"output directory is not a directory: {args.output_dir}")
    paths = validate_artifact_paths(
        {
            **{
                name: args.output_dir / f"{name}.npz"
                for name in ("joint_workspace", "position_ik", "fixed_pose_ik")
            },
            "report": args.output_dir / "report.json",
        },
        inputs=(args.urdf,),
    )
    model = RobotModel.from_urdf(require_robot_urdf(args.urdf, parser))
    base, tip = select_arm_links(model, args.arm, args.base_link, args.tip_link)
    solver = KinematicsSolver(
        model,
        base,
        tip,
        SolverConfig(
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            max_iterations=args.ik_max_iterations,
            random_seed=args.seed,
        ),
    )
    sampling = SamplingConfig(
        strategy=SamplingStrategy(args.strategy),
        num_samples=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    reference_joints = default_reference_joints(solver)
    reference_pose = _numpy(solver.forward(reference_joints))
    analyzer = WorkspaceAnalyzer(solver, WorkspaceConfig(sampling))
    results = {"joint_workspace": analyzer.analyze()}
    workspace_points = results["joint_workspace"].points
    bounds = (
        np.asarray(args.bounds).reshape(3, 2)
        if args.bounds is not None
        else np.stack(
            (workspace_points.min(0) - 0.02, workspace_points.max(0) + 0.02), axis=1
        )
    )
    cartesian_configs = {}
    for name, position_only in (("position_ik", True), ("fixed_pose_ik", False)):
        cartesian_configs[name] = CartesianConfig(
            bounds=bounds,
            sampling=sampling,
            position_only=position_only,
            restarts=args.ik_restarts,
            rescue_restarts=args.ik_rescue_restarts,
            rescue_rounds=args.ik_rescue_rounds,
            reference_pose=reference_pose,
            reference_joints=reference_joints,
        )
        print(f"Computing {name} ({args.samples:,} targets)...", flush=True)
        results[name] = analyzer.analyze_cartesian(cartesian_configs[name])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reproduce_args = [
        "--urdf",
        str(args.urdf.resolve()),
        "--base-link",
        solver.base_link,
        "--tip-link",
        solver.tip_link,
        "--backend",
        solver.backend,
        "--device",
        solver.device,
        "--dtype",
        solver.config.dtype,
        "--samples",
        str(sampling.num_samples),
        "--batch-size",
        str(sampling.batch_size),
        "--strategy",
        sampling.strategy.value,
        "--seed",
        str(sampling.seed),
        "--ik-max-iterations",
        str(solver.config.max_iterations),
        "--ik-restarts",
        str(args.ik_restarts),
        "--ik-rescue-restarts",
        str(args.ik_rescue_restarts),
        "--ik-rescue-rounds",
        str(args.ik_rescue_rounds),
        "--bounds",
        *(str(value) for value in bounds.ravel()),
    ]
    summary = {
        "format_version": 1,
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "sampling": asdict(sampling),
        "solver": {
            **asdict(solver.config),
            "backend": solver.backend,
            "device": solver.device,
        },
        "ik_restarts": args.ik_restarts,
        "ik_rescue_restarts": args.ik_rescue_restarts,
        "ik_rescue_rounds": args.ik_rescue_rounds,
        "joint_names": list(solver.joint_names),
        "reference_joints": reference_joints.tolist(),
        "reference_pose": reference_pose.tolist(),
        "reproduce_args": reproduce_args,
        "base_link": solver.base_link,
        "tip_link": solver.tip_link,
        "bounds": bounds.tolist(),
        "bounds_source": "explicit" if args.bounds else "joint_samples",
        "cases": {},
    }
    saved, errors = {}, {}
    for name, result in results.items():
        reachable_count = (
            None
            if result.reachable is None
            else int(np.count_nonzero(result.reachable))
        )
        try:
            result.save(paths[name])
        except Exception as exc:
            errors[name] = str(exc)
        else:
            saved[name] = paths[name]
        summary["cases"][name] = {
            "artifact": f"{name}.npz",
            "saved": name in saved,
            "samples": len(result.points),
            "reachable": reachable_count,
            "success_rate": None
            if reachable_count is None
            else reachable_count / len(result.points),
        }
        if name in errors:
            summary["cases"][name]["export_error"] = errors[name]
    summary["exports_complete"] = not errors
    try:
        atomic_write_text(
            paths["report"], json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
    except Exception as exc:
        errors["report"] = str(exc)
    else:
        saved["report"] = paths["report"]
    if errors:
        raise ValueError(
            "export failed: "
            + "; ".join(f"{paths[name]}: {error}" for name, error in errors.items())
            + (
                "; saved: " + ", ".join(str(path) for path in saved.values())
                if saved
                else "; no artifacts saved"
            )
        )
    print(f"All three experiments saved to {args.output_dir.resolve()}", flush=True)
    if args.viser:
        from workspace_analyzer.visualization import ViserWorkspace

        with ViserWorkspace(
            solver, port=args.port, initial_q=reference_joints, load_full_robot=True
        ) as viewer:
            with viewer.server.gui.add_folder("Evaluation results", order=-20):
                selector = viewer.server.gui.add_dropdown(
                    "Experiment", tuple(results), initial_value="position_ik"
                )

            def display(name):
                with viewer._lock:
                    viewer.add_workspace(results[name])
                    if name in cartesian_configs:
                        viewer.configure_cartesian_recompute(
                            analyzer, cartesian_configs[name]
                        )
                    else:
                        viewer.configure_cartesian_recompute(None, None)
                    viewer.frame_view()

            selector.on_update(lambda event: display(event.target.value))
            display("position_ik")
            viewer.wait()
    return results


if __name__ == "__main__":
    main()
