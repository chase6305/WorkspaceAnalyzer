"""Batched reachability scan on a plane in an explicit, fixed coordinate frame."""

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--arm", choices=("left", "right"), default="left")
    p.add_argument(
        "--frame",
        choices=("reference", "base", "world"),
        default="reference",
        help="coordinates use the neutral second active link, arm base, or URDF world",
    )
    p.add_argument("--plane", choices=("xy", "xz", "yz"), default="yz")
    p.add_argument(
        "--constant", type=float, default=0.4, help="constant coordinate in metres"
    )
    p.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "plane center in the selected frame; "
            "--constant overrides its normal coordinate"
        ),
    )
    p.add_argument("--span", type=float, default=0.8)
    p.add_argument("--resolution", type=int, default=41)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rpy",
        type=float,
        nargs=3,
        default=(0.0, np.pi / 2, 0.0),
        help="fixed TCP roll, pitch, yaw in the selected frame, in radians",
    )
    p.add_argument("--position-only", action="store_true")
    p.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--output", type=Path, default=Path("outputs/plane_reachability.json")
    )
    p.add_argument("--viser", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    args.urdf = require_robot_urdf(args.urdf, p)
    if args.resolution < 3 or not np.isfinite(args.span) or args.span <= 0:
        p.error("resolution must be >=3 and span finite and positive")
    if args.batch_size < 1 or args.seed < 0:
        p.error("batch-size must be positive and seed non-negative")
    if not np.isfinite(args.constant):
        p.error("constant must be finite")
    if not np.isfinite(args.rpy).all():
        p.error("rpy must contain three finite angles")
    center = np.array(
        args.center if args.center is not None else (0.40, -0.10, -0.10), dtype=float
    )
    if not np.isfinite(center).all():
        p.error("center must contain three finite coordinates")
    solver = create_solver(
        str(args.urdf),
        base_link=f"{args.arm}_arm_base",
        tip_link=f"{args.arm}_ee",
        backend=args.backend,
        device=args.device,
        dtype="float64",
    )
    frame_name, base_from_frame = _plane_frame(solver, args.frame)
    axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[args.plane]
    values = np.linspace(-args.span / 2, args.span / 2, args.resolution)
    xyz = np.zeros((args.resolution * args.resolution, 3))
    xyz[:, axes[0]] = np.repeat(values, args.resolution)
    xyz[:, axes[1]] = np.tile(values, args.resolution)
    # The selected coordinate is always constrained to the requested plane.
    # `center` controls the in-plane location only; its normal component is
    # intentionally overridden so --constant remains effective with --center.
    center[axes[2]] = args.constant
    xyz += center
    result = _scan_plane(
        solver,
        xyz,
        base_from_frame,
        args.rpy,
        position_only=args.position_only,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    result.metadata.update(plane=args.plane, sampling_frame=frame_name)
    success = result.reachable
    report = {
        "plane": args.plane,
        "frame": args.frame,
        "coordinate_frame": frame_name,
        "plane_center": center.tolist(),
        "base_from_plane_frame": base_from_frame.tolist(),
        "solver_base_link": solver.base_link,
        "constant_coordinate_m": args.constant,
        "span_m": args.span,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "backend": solver.backend,
        "device": solver.device,
        "target_rpy_rad": list(args.rpy),
        "task": "position_only" if args.position_only else "fixed_orientation",
        "reachable": int(success.sum()),
        "samples": len(success),
        "success_rate": float(success.mean()),
        "grid": success.reshape(args.resolution, args.resolution).tolist(),
    }
    if args.frame == "reference":
        # Keep the original report keys for consumers of the reference-frame scan.
        report.update(reference_link=frame_name, plane_center_link2=center.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))
    if args.viser:
        from workspace_analyzer.visualization import ViserWorkspace

        viewer = ViserWorkspace(
            solver,
            port=args.port,
            initial_q=default_reference_joints(solver),
            load_full_robot=True,
        )
        viewer.add_workspace(result)
        viewer.wait()


def _plane_frame(solver, frame):
    from workspace_analyzer.visualization import _zero_tree_transforms

    zero = _zero_tree_transforms(solver.model)
    if frame == "reference":
        active = solver.active_joints
        frame_name = active[min(1, len(active) - 1)].child
        world_from_frame = zero[frame_name]
    elif frame == "base":
        frame_name = solver.base_link
        world_from_frame = zero[frame_name]
    else:
        frame_name = "world"
        world_from_frame = np.eye(4)
    return frame_name, np.linalg.inv(zero[solver.base_link]) @ world_from_frame


def _scan_plane(solver, xyz, base_from_frame, rpy, *, position_only, batch_size, seed):
    points = xyz @ base_from_frame[:3, :3].T + base_from_frame[:3, 3]
    rotation = base_from_frame[:3, :3] @ _rpy(rpy)
    joints = np.empty((len(points), solver.dof), dtype=solver.config.dtype)
    success = np.empty(len(points), dtype=bool)
    residual = np.empty(len(points), dtype=solver.config.dtype)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        targets = np.broadcast_to(np.eye(4), (stop - start, 4, 4)).copy()
        targets[:, :3, 3] = points[start:stop]
        targets[:, :3, :3] = rotation
        result = solver.inverse(
            targets,
            position_only=position_only,
            restarts=4,
            rescue_restarts=16,
            rescue_rounds=2,
            random_seed=seed + start,
        )
        joints[start:stop] = _numpy(result.positions)
        success[start:stop] = _numpy(result.success)
        residual[start:stop] = _numpy(result.residual)
    return AnalysisResult(
        points=points,
        joint_positions=joints,
        manipulability=None,
        reachable=success,
        residual=residual,
        metadata={"mode": "fixed_plane", "coordinate_frame": solver.base_link},
    )


def _rpy(rpy):
    r, q, y = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(r),
        np.sin(r),
        np.cos(q),
        np.sin(q),
        np.cos(y),
        np.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


if __name__ == "__main__":
    main()
