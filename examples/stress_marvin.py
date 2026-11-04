"""Duration-based numerical stress test for a selected robot chain."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from workspace_analyzer import KinematicsSolver, RobotModel, SolverConfig
from workspace_analyzer._arrays import as_numpy as _numpy
from workspace_analyzer._atomic import atomic_write_text
from workspace_analyzer._output_paths import validate_output_paths
from workspace_analyzer.presets import (
    default_robot_urdf,
    require_robot_urdf,
    select_arm_links,
)

DEFAULT_URDF = default_robot_urdf()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--base-link")
    parser.add_argument("--tip-link")
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
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


def _run(args, parser):
    args.urdf = require_robot_urdf(args.urdf, parser)
    validate_output_paths(report=args.report, inputs=(args.urdf,))
    if (
        not np.isfinite(args.duration)
        or args.duration <= 0
        or args.batch_size <= 0
        or args.ik_targets <= 0
    ):
        parser.error("duration, batch-size, and ik-targets must be positive")

    if not np.isfinite(args.min_success_rate) or not 0 <= args.min_success_rate <= 1:
        parser.error("--min-success-rate must be finite and between 0 and 1")

    model = RobotModel.from_urdf(args.urdf)
    base, tip = select_arm_links(model, args.arm, args.base_link, args.tip_link)
    solver = KinematicsSolver(
        model,
        base,
        tip,
        SolverConfig(
            backend=args.backend,
            device=args.device,
            dtype="float64",
            random_seed=args.seed,
            max_iterations=200,
        ),
    )
    rng = np.random.default_rng(args.seed)
    lower, upper = solver.joint_limits[:, 0], solver.joint_limits[:, 1]
    started = time.perf_counter()
    deadline = started + args.duration
    batches = poses_checked = ik_total = ik_success = 0
    max_rotation_error = max_jacobian_error = max_success_residual = 0.0
    max_angular_jacobian_error = max_fk_position_error = 0.0
    max_joint_limit_violation = 0.0
    jacobian_columns_checked = set()
    next_progress = started + 30
    min_failure_residual = float("inf")
    max_failure_residual = 0.0

    while batches == 0 or time.perf_counter() < deadline:
        q = rng.uniform(lower, upper, (args.batch_size, solver.dof))
        poses, jacobians = solver.forward_with_jacobian(q)
        poses, jacobians = _numpy(poses), _numpy(jacobians)
        if not np.isfinite(poses).all() or not np.isfinite(jacobians).all():
            raise RuntimeError("non-finite FK or Jacobian output")
        rotation = poses[:, :3, :3]
        orthogonality = rotation @ rotation.swapaxes(1, 2) - np.eye(3)
        max_rotation_error = max(
            max_rotation_error, float(np.max(np.abs(orthogonality)))
        )
        poses_checked += len(q)

        if batches % 10 == 0:
            check_index = batches // 10
            sample_index = check_index % len(q)
            joint_index = check_index % solver.dof
            jacobian_columns_checked.add(joint_index)
            epsilon = 1e-7
            plus, minus = q[sample_index].copy(), q[sample_index].copy()
            plus[joint_index] += epsilon
            minus[joint_index] -= epsilon
            pose_plus, pose_minus = (
                _numpy(solver.forward(plus)),
                _numpy(solver.forward(minus)),
            )
            finite_difference = (pose_plus[:3, 3] - pose_minus[:3, 3]) / (2 * epsilon)
            derivative = (
                (pose_plus[:3, :3] - pose_minus[:3, :3])
                / (2 * epsilon)
                @ rotation[sample_index].T
            )
            angular = np.array([derivative[2, 1], derivative[0, 2], derivative[1, 0]])
            max_angular_jacobian_error = max(
                max_angular_jacobian_error,
                float(
                    np.max(np.abs(angular - jacobians[sample_index, 3:, joint_index]))
                ),
            )
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

        count = args.ik_targets
        ik_q = rng.uniform(lower, upper, (count, solver.dof))
        targets = _numpy(solver.forward(ik_q))
        result = solver.inverse(
            targets,
            restarts=args.restarts,
            rescue_restarts=args.rescue_restarts,
            rescue_rounds=args.rescue_rounds,
            random_seed=args.seed + batches,
        )
        success, residual = _numpy(result.success), _numpy(result.residual)
        positions = _numpy(result.positions)
        if not np.isfinite(positions).all() or not np.isfinite(residual).all():
            raise RuntimeError("non-finite IK output")
        max_joint_limit_violation = max(
            max_joint_limit_violation,
            float(max(np.max(lower - positions), np.max(positions - upper), 0)),
        )
        if np.any(success):
            achieved = _numpy(solver.forward(positions[success]))
            max_fk_position_error = max(
                max_fk_position_error,
                float(
                    np.max(
                        np.linalg.norm(
                            achieved[:, :3, 3] - targets[success, :3, 3], axis=1
                        )
                    )
                ),
            )
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
        if time.perf_counter() >= next_progress:
            print(
                f"{time.perf_counter() - started:.0f}s: {batches} batches, "
                f"IK {ik_success}/{ik_total}",
                file=sys.stderr,
                flush=True,
            )
            next_progress = time.perf_counter() + 30

    elapsed = time.perf_counter() - started
    report = {
        "robot": solver.model.name,
        "arm": args.arm,
        "base_link": solver.base_link,
        "tip_link": solver.tip_link,
        "joint_names": list(solver.joint_names),
        "dof": solver.dof,
        "seed": args.seed,
        "ik_seed_schedule": "seed + batch_index",
        "batch_size": args.batch_size,
        "ik_targets_per_batch": args.ik_targets,
        "requested_duration_seconds": args.duration,
        "ik_restarts": args.restarts,
        "ik_rescue_restarts": args.rescue_restarts,
        "ik_rescue_rounds": args.rescue_rounds,
        "minimum_success_rate": args.min_success_rate,
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
        "max_angular_jacobian_error": max_angular_jacobian_error,
        "jacobian_columns_checked": sorted(jacobian_columns_checked),
        "max_success_fk_position_error": max_fk_position_error,
        "max_joint_limit_violation": max_joint_limit_violation,
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report is not None:
        atomic_write_text(args.report, text + "\n")
    if report["ik_success_rate"] < args.min_success_rate:
        raise SystemExit("IK success rate fell below the requested threshold")
    if (
        max_rotation_error > 1e-10
        or max_jacobian_error > 1e-6
        or max_angular_jacobian_error > 1e-6
        or max_fk_position_error > solver.config.tolerance
        or max_joint_limit_violation > 1e-12
    ):
        raise SystemExit("numerical accuracy threshold exceeded")

    return report


if __name__ == "__main__":
    main()
