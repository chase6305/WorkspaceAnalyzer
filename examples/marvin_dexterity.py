"""Analyze Marvin single-arm dexterity and ordered-path joint continuity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workspace_analyzer import AnalysisResult, create_solver
from workspace_analyzer.presets import (
    default_reference_joints,
    default_robot_urdf,
    require_robot_urdf,
)

DEFAULT_URDF = default_robot_urdf()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--trajectory-frames", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--color-metric",
        choices=(
            "manipulability",
            "minimum_singular_value",
            "isotropy",
            "joint_limit_margin",
        ),
        default="isotropy",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    args.urdf = require_robot_urdf(args.urdf, parser)
    if args.samples <= 0 or args.trajectory_frames < 2 or args.dt <= 0:
        parser.error("samples and dt must be positive; trajectory-frames must be >= 2")

    solver = create_solver(
        str(args.urdf),
        base_link="base_link",
        tip_link=f"{args.arm}_ee",
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_iterations=200,
    )
    lower, upper = solver.joint_limits.T
    rng = np.random.default_rng(args.seed)
    q = rng.uniform(lower, upper, (args.samples, solver.dof))
    poses = _numpy(solver.forward(q))
    quality = solver.dexterity(q, task="position")
    metrics = {
        "minimum_singular_value": _numpy(quality.minimum_singular_value),
        "isotropy": _numpy(quality.isotropy),
        "joint_limit_margin": _numpy(quality.joint_limit_margin),
    }
    result = AnalysisResult(
        points=poses[:, :3, 3],
        joint_positions=q,
        manipulability=_numpy(quality.manipulability),
        metrics=metrics,
        metadata={
            "robot": solver.model.name,
            "base_link": solver.base_link,
            "tip_link": solver.tip_link,
            "samples": args.samples,
            "mode": "dexterity",
        },
    )

    phase = np.linspace(0.0, 2.0 * np.pi, args.trajectory_frames)
    center = default_reference_joints(solver)
    amplitude = np.minimum(
        (upper - lower) * 0.12, np.minimum(center - lower, upper - center)
    )
    path = center + np.sin(phase[:, None]) * amplitude
    trajectory = solver.solve_trajectory(
        solver.forward(path), seed=path[0], failure_restarts=8, dt=args.dt
    )
    report = {
        "robot": solver.model.name,
        "arm": args.arm,
        "backend": f"{solver.backend}:{solver.device}",
        "samples": args.samples,
        "workspace": {
            "manipulability": _percentiles(result.manipulability),
            **{name: _percentiles(values) for name, values in metrics.items()},
        },
        "trajectory": trajectory.summary(),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        result.save(args.output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if args.viser:
        from workspace_analyzer.visualization import ViserWorkspace

        viewer = ViserWorkspace(
            solver, port=args.port, initial_q=path[0], load_full_robot=True
        )
        viewer.add_workspace(result, color_metric=args.color_metric)
        viewer.add_trajectory(trajectory)
        viewer.wait()


def _percentiles(values) -> dict[str, float]:
    p05, median, p95 = np.percentile(values, (5, 50, 95))
    return {"p05": float(p05), "median": float(median), "p95": float(p95)}


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


if __name__ == "__main__":
    main()
