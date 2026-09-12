"""Refine observed IK success/failure brackets along translation segments."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np

from ._cancellation import check_cancelled
from ._validation import require_integer
from .analyzer import AnalysisResult
from .kinematics import _validate_homogeneous_rows, _validate_rotation_matrices
from .reachability import ReachabilityConfig, _real_array, analyze_targets


@dataclass(frozen=True)
class BoundaryConfig:
    tolerance_m: float = 1e-4
    max_rounds: int = 16

    def __post_init__(self):
        if (
            isinstance(self.tolerance_m, (bool, np.bool_))
            or not isinstance(self.tolerance_m, Real)
            or not np.isfinite(self.tolerance_m)
            or self.tolerance_m <= 0
        ):
            raise ValueError("tolerance_m must be finite and positive")
        require_integer(self.max_rounds, "max_rounds")
        object.__setattr__(self, "tolerance_m", float(self.tolerance_m))
        object.__setattr__(self, "max_rounds", int(self.max_rounds))


@dataclass
class BoundaryResult:
    measurements: AnalysisResult
    segment_ids: np.ndarray
    parameters: np.ndarray
    stages: np.ndarray
    lower_indices: np.ndarray
    upper_indices: np.ndarray
    statuses: np.ndarray
    rounds: np.ndarray
    initial_brackets: np.ndarray

    def to_dict(self) -> dict:
        m = self.measurements
        rows = []
        for index, (lower, upper) in enumerate(
            zip(self.lower_indices, self.upper_indices)
        ):
            rows.append(
                {
                    "segment_id": index,
                    "status": str(self.statuses[index]),
                    "rounds": int(self.rounds[index]),
                    "initial_bracket": bool(self.initial_brackets[index]),
                    "width_m": float(_distance(m.points[upper], m.points[lower])),
                    "lower": self._endpoint(lower, index),
                    "upper": self._endpoint(upper, index),
                }
            )
        return {
            "metadata": deepcopy(m.metadata),
            "summary": {
                "segments": len(rows),
                "initial_brackets": int(self.initial_brackets.sum()),
                "refined": int(np.count_nonzero(self.statuses == "refined")),
                "target_evaluations": len(m.points),
                "status_counts": {
                    str(status): int(np.count_nonzero(self.statuses == status))
                    for status in np.unique(self.statuses)
                },
            },
            "segments": rows,
        }

    def _endpoint(self, sample, segment):
        m = self.measurements
        residual = float(m.residual[sample])
        return {
            "sample_index": int(sample),
            "parameter": float(self.parameters[sample]),
            "offset_m": float(_distance(m.points[sample], m.points[segment])),
            "point": m.points[sample].tolist(),
            "ik_success": bool(m.reachable[sample]),
            "residual": residual if np.isfinite(residual) else None,
            "quality_pass": bool(m.metrics["quality_pass"][sample]),
        }

    def save(self, directory: str | Path) -> None:
        """Save measurements, sample-to-segment mapping, and strict JSON report."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.measurements.save(directory / "measurements.npz")
        np.savez_compressed(
            directory / "trace.npz",
            segment_ids=self.segment_ids,
            parameters=self.parameters,
            stages=self.stages,
            lower_indices=self.lower_indices,
            upper_indices=self.upper_indices,
            statuses=self.statuses,
            rounds=self.rounds,
            initial_brackets=self.initial_brackets,
        )
        (directory / "report.json").write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def refine_translation_boundary(
    solver,
    reference_poses,
    end_positions,
    config=None,
    *,
    reachability=None,
    seed=None,
    cache=None,
    cancel_event=None,
    progress_callback=None,
):
    """Bisect independently observed IK brackets at fixed reference orientation.

    A bracket requires a successful reference and an unsuccessful end. Midpoints
    use the current successful configuration as seed. The final failed endpoint
    is checked again with the latest successful seed. This local numerical
    bracket is not a proof of a geometric boundary or monotonic reachability.
    """
    check_cancelled(cancel_event)
    config = config or BoundaryConfig()
    reachability = reachability or ReachabilityConfig(position_only=False)
    poses = _real_array(reference_poses, "reference_poses", dtype=solver.config.dtype)
    if poses.shape == (4, 4):
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError("reference_poses must have nonempty shape (N, 4, 4)")
    _validate_homogeneous_rows(poses, "numpy")
    _validate_rotation_matrices(poses[:, :3, :3], "numpy")
    count = len(poses)
    ends = _real_array(end_positions, "end_positions", dtype=solver.config.dtype)
    if ends.shape == (3,):
        ends = ends[None]
    if ends.shape != (count, 3):
        raise ValueError("end_positions must have shape (N, 3)")
    start_points = poses[:, :3, 3].copy()
    # Parameters use float64, while actual evaluated points respect solver dtype.
    displacement = ends.astype(np.float64) - start_points.astype(np.float64)
    lengths = np.linalg.norm(displacement, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError("translation segments must have finite nonzero length")
    chunks, segment_chunks, parameter_chunks, stage_chunks = [], [], [], []
    batch_records = []
    total = 0

    def evaluate(ids, parameters, seeds, stage):
        nonlocal total
        check_cancelled(cancel_event)
        targets = poses[ids].copy()
        targets[:, :3, 3] = start_points[ids] + displacement[ids] * parameters[:, None]
        # Keep endpoints exactly equal to the input after conversion to solver dtype.
        targets[parameters == 1, :3, 3] = ends[ids[parameters == 1]]
        measured = analyze_targets(
            solver,
            targets,
            reachability,
            seed=seeds,
            cache=cache,
            cancel_event=cancel_event,
        )
        indices = np.arange(total, total + len(ids))
        batch_records.append(
            {
                "start_row": total,
                "count": len(ids),
                "stage": stage,
                "cache_hit": measured.metadata.get("cache_hit", False),
                "cache_key": measured.metadata.get("cache_key"),
            }
        )
        total += len(ids)
        chunks.append(measured)
        segment_chunks.append(ids.copy())
        parameter_chunks.append(parameters.copy())
        stage_chunks.append(np.full(len(ids), stage))
        return measured, indices

    ids = np.arange(count)
    start, lower_indices = evaluate(ids, np.zeros(count), seed, "reference")
    end, upper_indices = evaluate(ids, np.ones(count), start.joint_positions, "end")
    initial = start.reachable & ~end.reachable
    statuses = np.full(count, "budget_exhausted", dtype="U32")
    statuses[~start.reachable] = "start_failed"
    statuses[start.reachable & end.reachable] = "end_succeeded"
    lower, upper = np.zeros(count), np.ones(count)
    lower_points, upper_points = start.points.copy(), end.points.copy()
    lower_q = start.joint_positions.copy()
    rounds = np.zeros(count, dtype=np.int64)

    def update_converged():
        width = _distance(upper_points, lower_points)
        statuses[initial & (width <= config.tolerance_m)] = "refined"

    update_converged()
    for iteration in range(config.max_rounds):
        check_cancelled(cancel_event)
        active = np.flatnonzero(statuses == "budget_exhausted")
        if not len(active):
            break
        middle = (lower[active] + upper[active]) / 2
        middle_points = (
            start_points[active] + displacement[active] * middle[:, None]
        ).astype(start_points.dtype)
        stuck = np.all(middle_points == lower_points[active], axis=1) | np.all(
            middle_points == upper_points[active], axis=1
        )
        statuses[active[stuck]] = "precision_limit"
        active, middle = active[~stuck], middle[~stuck]
        if len(active):
            measured, indices = evaluate(active, middle, lower_q[active], "midpoint")
            good, bad = measured.reachable, ~measured.reachable
            passed, failed = active[good], active[bad]
            lower[passed], upper[failed] = middle[good], middle[bad]
            lower_points[passed], upper_points[failed] = (
                measured.points[good],
                measured.points[bad],
            )
            lower_indices[passed], upper_indices[failed] = indices[good], indices[bad]
            lower_q[passed] = measured.joint_positions[good]
            rounds[active] += 1
            update_converged()
        if progress_callback is not None:
            progress_callback((iteration + 1) / (config.max_rounds + 1))

    # Failure is provisional: a nearer seed can recover the same endpoint.
    active = np.flatnonzero(initial)
    if len(active):
        checked, indices = evaluate(
            active, upper[active], lower_q[active], "end_recheck"
        )
        upper_indices[active] = indices
        statuses[active[checked.reachable]] = "end_recovered"
    check_cancelled(cancel_event)
    metadata = deepcopy(start.metadata)
    metadata["joint_names"] = list(metadata["joint_names"])
    for name in ("assessment", "cache_key", "cache_hit"):
        metadata.pop(name, None)
    metadata.update(
        mode="translation_boundary",
        boundary_config=vars(config).copy(),
        measurement_batches=batch_records,
        classification="observed_ik_success_failure",
        seed_policy="reference_input_then_current_successful_endpoint",
        fixed_orientation=True,
        orientation_constrained=not reachability.position_only,
        geometric_boundary_proven=False,
    )
    fields = {
        name: np.concatenate([getattr(chunk, name) for chunk in chunks])
        for name in (
            "points",
            "joint_positions",
            "manipulability",
            "reachable",
            "residual",
            "target_poses",
        )
    }
    measurements = AnalysisResult(
        **fields,
        metadata=metadata,
        metrics={
            name: np.concatenate([chunk.metrics[name] for chunk in chunks])
            for name in start.metrics
        },
    )
    result = BoundaryResult(
        measurements,
        np.concatenate(segment_chunks),
        np.concatenate(parameter_chunks),
        np.concatenate(stage_chunks),
        lower_indices,
        upper_indices,
        statuses,
        rounds,
        initial,
    )
    if progress_callback is not None:
        progress_callback(1.0)
    check_cancelled(cancel_event)
    return result


def _distance(first, second):
    return np.linalg.norm(
        np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64),
        axis=-1,
    )
