"""Aggregate sampled full-pose outcomes by explicit task-position identifiers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

import numpy as np

from ._cancellation import check_cancelled
from .reachability import _assessment_arrays, _validated_quality_flags


@dataclass(frozen=True)
class OrientationCoverage:
    position_ids: np.ndarray
    points: np.ndarray
    sample_counts: np.ndarray
    ik_success_counts: np.ndarray
    quality_pass_counts: np.ndarray
    ik_coverage: np.ndarray
    quality_coverage: np.ndarray
    weighted_ik_coverage: np.ndarray
    weighted_quality_coverage: np.ndarray
    metadata: dict

    def summary(self) -> dict:
        return {
            "positions": len(self.points),
            "pose_samples": int(self.sample_counts.sum()),
            "positions_with_any_ik": int(np.count_nonzero(self.ik_success_counts)),
            "positions_with_all_ik": int(
                np.count_nonzero(self.ik_success_counts == self.sample_counts)
            ),
            "positions_with_any_quality": int(
                np.count_nonzero(self.quality_pass_counts)
            ),
            "positions_with_all_quality": int(
                np.count_nonzero(self.quality_pass_counts == self.sample_counts)
            ),
            "mean_ik_coverage": float(self.ik_coverage.mean()),
            "mean_quality_coverage": float(self.quality_coverage.mean()),
        }

    def to_dict(self) -> dict:
        """Return JSON-ready data; zero-weight groups have null weighted rates."""
        return {
            "metadata": deepcopy(self.metadata),
            "summary": self.summary(),
            "position_ids": self.position_ids.tolist(),
            "points": self.points.tolist(),
            "sample_counts": self.sample_counts.tolist(),
            "ik_success_counts": self.ik_success_counts.tolist(),
            "quality_pass_counts": self.quality_pass_counts.tolist(),
            "ik_coverage": self.ik_coverage.tolist(),
            "quality_coverage": self.quality_coverage.tolist(),
            "weighted_ik_coverage": [
                float(x) if np.isfinite(x) else None for x in self.weighted_ik_coverage
            ],
            "weighted_quality_coverage": [
                float(x) if np.isfinite(x) else None
                for x in self.weighted_quality_coverage
            ],
        }


def summarize_orientation_coverage(
    result, position_ids, *, position_tolerance=1e-8, cancel_event=None
):
    """Group tested poses in first-occurrence order without running kinematics.

    All targets in a group must share a position within the given distance in
    metres. Coverage counts supplied samples, including repeated orientations;
    it is not a statement about all of SO(3) or collision-free motion.
    """
    check_cancelled(cancel_event)
    reachable, metrics = _assessment_arrays(result)
    if result.metadata["position_only"] or result.target_poses is None:
        raise ValueError("orientation coverage requires a full-pose assessment")
    if (
        isinstance(position_tolerance, (bool, np.bool_))
        or not isinstance(position_tolerance, Real)
        or not np.isfinite(position_tolerance)
        or position_tolerance < 0
    ):
        raise ValueError("position_tolerance must be finite and non-negative")
    ids = np.asarray(position_ids)
    if ids.shape != reachable.shape or ids.dtype.kind not in "iuU":
        raise ValueError("position_ids must contain N integer or string identifiers")
    points = np.asarray(result.points)
    if points.shape != (len(ids), 3) or not np.isfinite(points).all():
        raise ValueError("coverage points must have shape (N, 3) and be finite")
    poses = np.asarray(result.target_poses)
    if (
        poses.shape != (len(ids), 4, 4)
        or not np.isfinite(poses).all()
        or np.any(np.linalg.norm(poses[:, :3, 3] - points, axis=1) > position_tolerance)
    ):
        raise ValueError("target_poses must match the assessment points")
    accepted = _validated_quality_flags(result, reachable, metrics)
    weights = metrics["task_weight"]
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("task weights must be finite and non-negative")
    check_cancelled(cancel_event)
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    check_cancelled(cancel_event)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    inverse = remap[inverse]
    first = first[order]
    representative = points[first].copy()
    if np.any(
        np.linalg.norm(points - representative[inverse], axis=1) > position_tolerance
    ):
        raise ValueError("all poses with the same position_id must share a position")
    groups = len(first)
    check_cancelled(cancel_event)
    count = np.bincount(inverse, minlength=groups)
    solved = np.bincount(inverse[reachable], minlength=groups)
    passed = np.bincount(inverse[accepted], minlength=groups)
    # Scale separately within each group, retaining tiny weights in groups whose
    # overall mass is negligible compared with the rest of the task set.
    maximum = np.zeros(groups)
    np.maximum.at(maximum, inverse, weights)
    scaled = np.divide(
        weights, maximum[inverse], out=np.zeros(len(ids)), where=maximum[inverse] > 0
    )
    mass = np.bincount(inverse, weights=scaled, minlength=groups)
    weighted = []
    for mask in (reachable, accepted):
        check_cancelled(cancel_event)
        numerator = np.bincount(inverse[mask], weights=scaled[mask], minlength=groups)
        weighted.append(
            np.divide(numerator, mass, out=np.full(groups, np.nan), where=mass > 0)
        )
    result = OrientationCoverage(
        position_ids=ids[first].copy(),
        points=representative,
        sample_counts=count,
        ik_success_counts=solved,
        quality_pass_counts=passed,
        ik_coverage=solved / count,
        quality_coverage=passed / count,
        weighted_ik_coverage=weighted[0],
        weighted_quality_coverage=weighted[1],
        metadata={
            "coordinate_frame": result.metadata.get(
                "coordinate_frame", result.metadata.get("base_link")
            ),
            "position_tolerance": float(position_tolerance),
            "sampled_orientations_only": True,
            "quality_scope": result.metadata.get(
                "quality_scope", "selected_ik_solution"
            ),
            "quality_thresholds": deepcopy(
                result.metadata.get("quality_thresholds", {})
            ),
            "collision_checked": result.metadata.get("collision_checked", False),
        },
    )
    check_cancelled(cancel_event)
    return result
