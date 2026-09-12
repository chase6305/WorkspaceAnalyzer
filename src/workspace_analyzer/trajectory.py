"""Sequential IK and continuity diagnostics for ordered Cartesian trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._cancellation import check_cancelled
from ._validation import require_integer


@dataclass(frozen=True)
class TrajectoryIKResult:
    target_poses: np.ndarray
    positions: np.ndarray
    success: np.ndarray
    residual: np.ndarray
    joint_delta: np.ndarray
    velocity: np.ndarray | None
    acceleration: np.ndarray | None
    velocity_violation: np.ndarray | None
    minimum_singular_value: np.ndarray
    joint_limit_margin: np.ndarray
    joint_ranges: np.ndarray

    @property
    def max_joint_jump(self) -> float:
        finite = np.abs(self.joint_delta[np.isfinite(self.joint_delta)])
        return float(np.max(finite, initial=0.0))

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.success))

    @property
    def max_normalized_joint_jump(self) -> float:
        normalized = np.abs(self.joint_delta) / self.joint_ranges
        finite = normalized[np.isfinite(normalized)]
        return float(np.max(finite, initial=0.0))

    def summary(self) -> dict[str, float | int | None]:
        """Return JSON-friendly continuity and dexterity diagnostics."""

        def finite_min(values):
            values = values[np.isfinite(values)]
            return None if not len(values) else float(np.min(values))

        return {
            "samples": len(self.positions),
            "success_rate": self.success_rate,
            "max_joint_jump": self.max_joint_jump,
            "max_normalized_joint_jump": self.max_normalized_joint_jump,
            "joint_path_length": float(
                np.nansum(np.linalg.norm(self.joint_delta, axis=1))
            ),
            "normalized_closure_joint_error": _closure_error(self),
            "max_joint_velocity": _finite_abs_max(self.velocity),
            "rms_joint_velocity": _finite_rms(self.velocity),
            "max_joint_acceleration": _finite_abs_max(self.acceleration),
            "minimum_singular_value": finite_min(self.minimum_singular_value),
            "minimum_joint_limit_margin": finite_min(self.joint_limit_margin),
            "velocity_violations": (
                None
                if self.velocity_violation is None
                else int(np.count_nonzero(self.velocity_violation))
            ),
        }


def solve_trajectory(
    solver,
    target_poses,
    seed=None,
    *,
    position_only: bool = False,
    failure_restarts: int = 8,
    dt: float | None = None,
    timestamps=None,
    jump_repair_threshold: float | None = 0.25,
    posture_gain: float = 0.005,
    enforce_loop_closure: bool = True,
    cancel_event=None,
    batch_size: int = 1024,
) -> TrajectoryIKResult:
    """Solve an ordered path with previous-solution warm starts and diagnostics."""
    check_cancelled(cancel_event)
    targets = np.asarray(_numpy(target_poses), dtype=float)
    if targets.ndim != 3 or targets.shape[1:] != (4, 4) or len(targets) == 0:
        raise ValueError("target_poses must have shape (N, 4, 4) with N > 0")
    require_integer(failure_restarts, "failure_restarts", minimum=1)
    require_integer(batch_size, "batch_size", minimum=1)
    if dt is not None and (not np.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be finite and positive")
    if dt is not None and timestamps is not None:
        raise ValueError("provide either dt or timestamps, not both")
    if jump_repair_threshold is not None and (
        not np.isfinite(jump_repair_threshold) or jump_repair_threshold <= 0
    ):
        raise ValueError("jump_repair_threshold must be positive or None")
    if not np.isfinite(posture_gain) or posture_gain < 0:
        raise ValueError("posture_gain must be finite and non-negative")
    intervals = None
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float)
        if (
            timestamps.shape != (len(targets),)
            or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise ValueError(
                "timestamps must have shape (N,), be finite, and strictly increase"
            )
        intervals = np.diff(timestamps)
    elif dt is not None:
        intervals = np.full(len(targets) - 1, dt)
    current = (
        solver.joint_limits.mean(axis=1)
        if seed is None
        else np.asarray(seed, dtype=float)
    )
    if current.shape != (solver.dof,):
        raise ValueError(f"seed must have shape ({solver.dof},)")

    positions = np.empty((len(targets), solver.dof))
    success = np.empty(len(targets), dtype=bool)
    residual = np.empty(len(targets))
    posture_reference = None
    for index, target in enumerate(targets):
        check_cancelled(cancel_event)
        result = solver.inverse(
            target,
            seed=current,
            position_only=position_only,
            posture_reference=posture_reference,
            posture_gain=posture_gain,
            cancel_event=cancel_event,
        )
        solved = bool(_numpy(result.success))
        used_restarts = False
        if not solved and failure_restarts > 1:
            used_restarts = True
            result = solver.inverse(
                target,
                seed=current,
                position_only=position_only,
                restarts=failure_restarts,
                prefer_seed=True,
                posture_reference=posture_reference,
                posture_gain=posture_gain,
                cancel_event=cancel_event,
            )
            solved = bool(_numpy(result.success))
        candidate = np.asarray(_numpy(result.positions), dtype=float)
        if (
            solved
            and not used_restarts
            and failure_restarts > 1
            and jump_repair_threshold is not None
            and _joint_max_delta(solver, candidate, current) > jump_repair_threshold
        ):
            repaired = solver.inverse(
                target,
                seed=current,
                position_only=position_only,
                restarts=failure_restarts,
                prefer_seed=True,
                posture_reference=posture_reference,
                posture_gain=posture_gain,
                cancel_event=cancel_event,
            )
            repaired_success = bool(_numpy(repaired.success))
            repaired_q = np.asarray(_numpy(repaired.positions), dtype=float)
            if repaired_success and _joint_distance(
                solver, repaired_q, current
            ) < _joint_distance(solver, candidate, current):
                result, candidate = repaired, repaired_q
        if solved:
            current = candidate
            if posture_reference is None:
                posture_reference = candidate.copy()
        positions[index] = candidate
        success[index] = solved
        residual[index] = float(_numpy(result.residual))

    if enforce_loop_closure:
        positions, success, residual = _refine_closed_loop(
            solver,
            targets,
            positions,
            success,
            residual,
            position_only=position_only,
            posture_gain=posture_gain,
            cancel_event=cancel_event,
            batch_size=batch_size,
        )
    check_cancelled(cancel_event)
    positions = _unwrap_continuous(solver, positions, success)
    delta = np.diff(positions, axis=0)
    valid_segments = success[:-1] & success[1:]
    delta[~valid_segments] = np.nan
    velocity = None if intervals is None else delta / intervals[:, None]
    acceleration = None
    if velocity is not None and len(velocity) >= 2:
        acceleration_dt = (intervals[:-1] + intervals[1:]) * 0.5
        acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
    violation = None
    if velocity is not None:
        limits = np.asarray(
            [
                joint.limit.velocity if joint.limit.velocity is not None else np.inf
                for joint in solver.active_joints
            ]
        )
        violation = np.abs(velocity) > limits
    minimum_singular_value = np.full(len(positions), np.nan, dtype=solver.config.dtype)
    joint_limit_margin = np.full_like(minimum_singular_value, np.nan)
    for start in range(0, len(positions), batch_size):
        check_cancelled(cancel_event)
        indices = start + np.flatnonzero(success[start : start + batch_size])
        if len(indices):
            quality = solver.dexterity(positions[indices], task="position")
            minimum_singular_value[indices] = _numpy(quality.minimum_singular_value)
            joint_limit_margin[indices] = _numpy(quality.joint_limit_margin)
    check_cancelled(cancel_event)
    return TrajectoryIKResult(
        targets,
        positions,
        success,
        residual,
        delta,
        velocity,
        acceleration,
        violation,
        minimum_singular_value,
        joint_limit_margin,
        np.maximum(solver.joint_limits[:, 1] - solver.joint_limits[:, 0], 1e-12),
    )


def _joint_distance(solver, q, reference):
    delta = np.asarray(q) - np.asarray(reference)
    for index, joint in enumerate(solver.active_joints):
        if joint.kind == "continuous":
            delta[index] = (delta[index] + np.pi) % (2 * np.pi) - np.pi
    ranges = np.maximum(solver.joint_limits[:, 1] - solver.joint_limits[:, 0], 1e-12)
    return float(np.linalg.norm(delta / ranges))


def _refine_closed_loop(
    solver,
    targets,
    positions,
    success,
    residual,
    *,
    position_only,
    posture_gain,
    cancel_event=None,
    batch_size=1024,
):
    """Distribute null-space drift and reproject a periodic path onto its targets."""
    if (
        len(targets) < 3
        or not success.all()
        or not np.allclose(targets[0], targets[-1], atol=1e-8, rtol=0.0)
    ):
        return positions, success, residual
    closure = positions[0] - positions[-1]
    for index, joint in enumerate(solver.active_joints):
        if joint.kind == "continuous":
            closure[index] = (closure[index] + np.pi) % (2 * np.pi) - np.pi
    refined = np.empty_like(positions)
    refined_residual = np.empty_like(residual)
    for start in range(0, len(targets), batch_size):
        check_cancelled(cancel_event)
        stop = min(start + batch_size, len(targets))
        phase = (np.arange(start, stop) / (len(targets) - 1))[:, None]
        seeds = solver._bound_configuration(
            solver._array(positions[start:stop] + phase * closure)
        )
        result = solver.inverse(
            targets[start:stop],
            seed=seeds,
            position_only=position_only,
            posture_reference=positions[0],
            posture_gain=posture_gain,
            cancel_event=cancel_event,
        )
        if not bool(_numpy(result.success).all()):
            return positions, success, residual
        refined[start:stop] = _numpy(result.positions)
        refined_residual[start:stop] = _numpy(result.residual)
    # Reuse the first solution only for an identical target. Approximately closed
    # paths still need their own last pose and its independently measured residual.
    if np.array_equal(targets[0], targets[-1]):
        refined[-1] = refined[0]
        refined_residual[-1] = refined_residual[0]
    return refined, success.copy(), refined_residual


def _joint_max_delta(solver, q, reference):
    delta = np.asarray(q) - np.asarray(reference)
    for index, joint in enumerate(solver.active_joints):
        if joint.kind == "continuous":
            delta[index] = (delta[index] + np.pi) % (2 * np.pi) - np.pi
    return float(np.max(np.abs(delta)))


def _finite_abs_max(values):
    if values is None:
        return None
    finite = np.abs(values[np.isfinite(values)])
    return None if not len(finite) else float(np.max(finite))


def _finite_rms(values):
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    return None if not len(finite) else float(np.sqrt(np.mean(finite**2)))


def _closure_error(result):
    if len(result.positions) < 2 or not (result.success[0] and result.success[-1]):
        return None
    if not np.allclose(
        result.target_poses[0], result.target_poses[-1], atol=1e-8, rtol=0.0
    ):
        return None
    delta = result.positions[-1] - result.positions[0]
    return float(np.linalg.norm(delta / result.joint_ranges))


def _unwrap_continuous(solver, positions, success=None):
    result = positions.copy()
    valid = np.ones(len(positions), dtype=bool) if success is None else success
    for index, joint in enumerate(solver.active_joints):
        if joint.kind == "continuous":
            # Failed configurations must not introduce fictitious turns into
            # the successful path. Invalid adjacent segments are masked later.
            result[valid, index] = np.unwrap(result[valid, index])
    return result


def _numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )
