"""Duration-based numerical stress test for one Marvin M6 arm."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from workspace_analyzer import create_solver
from workspace_analyzer.presets import default_robot_urdf, require_robot_urdf

DEFAULT_URDF = default_robot_urdf()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ik-targets", type=int, default=32)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--rescue-restarts", type=int, default=16)
    parser.add_argument("--rescue-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.urdf = require_robot_urdf(args.urdf, parser)
    if args.duration <= 0 or args.batch_size <= 0 or args.ik_targets <= 0:
        parser.error("duration, batch-size, and ik-targets must be positive")

    solver = create_solver(
        str(args.urdf),
        base_link="base_link",
        tip_link=f"{args.arm}_ee",
        backend=args.backend,
        device=args.device,
        dtype="float64",
        max_iterations=200,
    )
    rng = np.random.default_rng(args.seed)
    lower, upper = solver.joint_limits[:, 0], solver.joint_limits[:, 1]
    started = time.perf_counter()
    deadline = started + args.duration
    batches = poses_checked = ik_total = ik_success = 0
    max_rotation_error = max_jacobian_error = max_success_residual = 0.0
    min_failure_residual = float("inf")
    max_failure_residual = 0.0

    while time.perf_counter() < deadline:
        q = rng.uniform(lower, upper, (args.batch_size, solver.dof))
        poses, jacobians = solver.forward_with_jacobian(q)
        poses, jacobians = _numpy(poses), _numpy(jacobians)
        rotation = poses[:, :3, :3]
        orthogonality = rotation @ rotation.swapaxes(1, 2) - np.eye(3)
        max_rotation_error = max(
            max_rotation_error, float(np.max(np.abs(orthogonality)))
        )
        poses_checked += len(q)

        if batches % 10 == 0:
            sample_index = batches % len(q)
            joint_index = batches % solver.dof
            epsilon = 1e-7
            plus, minus = q[sample_index].copy(), q[sample_index].copy()
            plus[joint_index] += epsilon
            minus[joint_index] -= epsilon
            finite_difference = (
                _numpy(solver.forward(plus))[:3, 3]
                - _numpy(solver.forward(minus))[:3, 3]
            ) / (2 * epsilon)
            max_jacobian_error = max(
                max_jacobian_error,
                float(
                    np.max(
                        np.abs(
                            finite_difference - jacobians[sample_index, :3, joint_index]
                        )
                    )
                ),
            )

        count = min(args.ik_targets, len(q))
        result = solver.inverse(
            poses[:count],
            restarts=args.restarts,
            rescue_restarts=args.rescue_restarts,
            rescue_rounds=args.rescue_rounds,
        )
        success, residual = _numpy(result.success), _numpy(result.residual)
        ik_total += count
        ik_success += int(np.count_nonzero(success))
        if np.any(success):
            max_success_residual = max(
                max_success_residual, float(np.max(residual[success]))
            )
        if np.any(~success):
            min_failure_residual = min(
                min_failure_residual, float(np.min(residual[~success]))
            )
            max_failure_residual = max(
                max_failure_residual, float(np.max(residual[~success]))
            )
        batches += 1

    elapsed = time.perf_counter() - started
    report = {
        "robot": solver.model.name,
        "arm": args.arm,
        "backend": solver.backend,
        "device": solver.device,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "duration_seconds": elapsed,
        "batches": batches,
        "poses_checked": poses_checked,
        "poses_per_second": poses_checked / elapsed,
        "ik_targets": ik_total,
        "ik_targets_per_second": ik_total / elapsed,
        "ik_failures": ik_total - ik_success,
        "ik_success_rate": ik_success / ik_total,
        "max_success_residual": max_success_residual,
        "min_failure_residual": (
            None if min_failure_residual == float("inf") else min_failure_residual
        ),
        "max_failure_residual": (
            None if min_failure_residual == float("inf") else max_failure_residual
        ),
        "max_rotation_orthogonality_error": max_rotation_error,
        "max_position_jacobian_error": max_jacobian_error,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if report["ik_success_rate"] < 0.99:
        raise SystemExit("IK success rate fell below 99%")
    if max_rotation_error > 1e-10 or max_jacobian_error > 1e-6:
        raise SystemExit("numerical accuracy threshold exceeded")


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


if __name__ == "__main__":
    main()
