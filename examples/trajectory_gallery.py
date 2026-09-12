"""Visualize line, circle, figure-eight, or helix Cartesian IK trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from workspace_analyzer import create_solver
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
    parser.add_argument(
        "--shape", choices=("line", "circle", "figure8", "helix"), default="figure8"
    )
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--depth", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--full-pose", action="store_true")
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--viser", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    args.urdf = require_robot_urdf(args.urdf, parser)
    if args.frames < 2 or args.scale <= 0 or args.depth < 0 or args.dt <= 0:
        parser.error("frames >= 2, scale > 0, depth >= 0, and dt > 0 are required")

    solver = create_solver(
        str(args.urdf),
        base_link=f"{args.arm}_arm_base",
        tip_link=f"{args.arm}_ee",
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        max_iterations=200,
    )
    reference_q = default_reference_joints(solver)
    reference_pose = _numpy(solver.forward(reference_q))
    canonical = _canonical_trajectory(args.shape, args.frames, args.scale, args.depth)
    targets = reference_pose[None] @ canonical
    result = solver.solve_trajectory(
        targets,
        seed=reference_q,
        position_only=not args.full_pose,
        failure_restarts=16,
        dt=args.dt,
    )
    report = {
        "robot": solver.model.name,
        "arm": args.arm,
        "shape": args.shape,
        "full_pose": args.full_pose,
        "backend": f"{solver.backend}:{solver.device}",
        **result.summary(),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if args.viser:
        from workspace_analyzer.visualization import ViserWorkspace

        viewer = ViserWorkspace(
            solver,
            port=args.port,
            initial_q=reference_q,
            load_robot_visuals=not args.no_visuals,
            load_full_robot=True,
        )
        viewer.add_trajectory(result)
        viewer.add_dashboard(
            "## Trajectory conclusion\n"
            f"- Shape: **{args.shape}**\n"
            f"- IK success: **{result.success_rate:.1%}**\n"
            f"- Max joint jump: **{result.max_joint_jump:.5f} rad**\n"
            f"- Velocity violations: **{result.summary()['velocity_violations']}**"
        )
        viewer.wait()


def _canonical_trajectory(shape, frames, scale, depth):
    phase = np.linspace(0.0, 2.0 * np.pi, frames)
    xyz = np.zeros((frames, 3))
    if shape == "line":
        xyz[:, 0] = np.linspace(-scale, scale, frames)
    elif shape == "circle":
        xyz[:, 0] = scale * np.cos(phase)
        xyz[:, 1] = scale * np.sin(phase)
    elif shape == "figure8":
        xyz[:, 0] = scale * np.sin(phase)
        xyz[:, 1] = 0.5 * scale * np.sin(2.0 * phase)
    else:
        xyz[:, 0] = scale * np.cos(phase)
        xyz[:, 1] = scale * np.sin(phase)
        xyz[:, 2] = np.linspace(0.5 * scale, -0.5 * scale, frames)
    xyz[:, 2] -= depth
    poses = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
    poses[:, :3, 3] = xyz
    return poses


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


if __name__ == "__main__":
    main()
