"""Select measured IK candidates without rerunning kinematics."""

from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np

from ._cancellation import check_cancelled
from .analyzer import AnalysisResult, _numeric_array
from .cache import _normalize

SELECTION_OBJECTIVES = (
    "joint_limit_margin",
    "isotropy",
    "minimum_singular_value",
)


def select_ik_solutions(study, *, objective="joint_limit_margin", cancel_event=None):
    """Choose one observed candidate per target from an IK stability study.

    Prefer quality-accepted candidates, then IK successes, then failures. Within
    either successful group maximize the requested finite metric, breaking ties
    by lower finite residual and then original trial order. Failed candidates
    are ranked only by residual. A failure remains a failure in the result.

    The returned AnalysisResult owns its arrays. Its candidate_selection metadata
    is an audit snapshot at selection time; reassessing that result only changes
    gates on the chosen configurations. Reassess the study and select again to
    reconsider every candidate under different gates. No IK/FK is performed.
    """
    check_cancelled(cancel_event)
    if objective not in SELECTION_OBJECTIVES:
        raise ValueError(f"objective must be one of {SELECTION_OBJECTIVES}")
    ik, quality = _validated_candidates(study, cancel_event=cancel_event)
    baseline = study.results[0]
    thresholds = baseline.metadata["quality_thresholds"]
    count = len(baseline.points)
    indices = np.zeros(count, dtype=np.int64)
    best_group = np.full(count, -1, dtype=np.int8)
    best_score = np.full(count, -np.inf)
    best_residual = np.full(count, np.inf)
    for index, result in enumerate(study.results):
        check_cancelled(cancel_event)
        residual = np.asarray(result.residual)
        group = ik[index].astype(np.int8) + quality[index]
        metric = result.metrics[objective]
        score = np.where(ik[index] & np.isfinite(metric), metric, -np.inf)
        error = np.where(np.isfinite(residual) & (residual >= 0), residual, np.inf)
        better = (group > best_group) | (
            (group == best_group)
            & ((score > best_score) | ((score == best_score) & (error < best_residual)))
        )
        indices[better] = index
        best_group[better] = group[better]
        best_score[better] = score[better]
        best_residual[better] = error[better]

    fields = ("joint_positions", "manipulability", "reachable", "residual")
    gathered = {name: np.empty_like(getattr(baseline, name)) for name in fields}
    metrics = {name: np.empty_like(values) for name, values in baseline.metrics.items()}
    selected_counts = np.bincount(indices, minlength=len(study.results))
    for index, result in enumerate(study.results):
        check_cancelled(cancel_event)
        if not selected_counts[index]:
            continue
        # Reuse each row index across all fields instead of scanning all targets
        # for every field and trial, including trials with no selected candidate.
        rows = np.flatnonzero(indices == index)
        for name in fields:
            gathered[name][rows] = getattr(result, name)[rows]
        for name in metrics:
            metrics[name][rows] = result.metrics[name][rows]
    metrics["selected_trial_index"] = indices.copy()
    metadata = {
        key: deepcopy(baseline.metadata[key])
        for key in (
            "mode",
            "robot",
            "base_link",
            "tip_link",
            "joint_names",
            "joint_kinds",
            "coordinate_frame",
            "base_from_targets",
            "backend",
            "device",
            "dtype",
            "position_only",
            "batch_size",
            "dexterity_task",
            "dexterity_weights",
            "quality_thresholds",
            "stability_model_digest",
            "collision_checked",
        )
        if key in baseline.metadata
    }
    metadata["solver_settings"] = {
        key: deepcopy(value)
        for key, value in baseline.metadata.get("solver_settings", {}).items()
        if key not in {"max_iterations", "random_seed"}
    }
    metadata["quality_scope"] = "selected_observed_ik_candidate"
    selected = AnalysisResult(
        points=baseline.points.copy(),
        target_poses=None
        if baseline.target_poses is None
        else baseline.target_poses.copy(),
        **gathered,
        metrics=metrics,
        metadata=metadata,
    ).reassess_quality(**thresholds, cancel_event=cancel_event)
    selected.metadata.pop("quality_reassessed", None)
    # Store a selection-time audit separately from the current quality assessment.
    summary = {"targets": count, "trials": len(study.results)}
    for label, before, after in (
        ("ik", ik[0], selected.reachable),
        ("quality", quality[0], selected.metrics["quality_pass"]),
    ):
        gained = np.flatnonzero(~before & after).tolist()
        lost = np.flatnonzero(before & ~after).tolist()
        summary.update(
            {
                f"{label}_gained_count": len(gained),
                f"{label}_lost_count": len(lost),
                f"{label}_gained_indices": gained,
                f"{label}_lost_indices": lost,
            }
        )
    before = baseline.metrics[objective]
    after = selected.metrics[objective]
    comparable = ik[0] & selected.reachable & np.isfinite(before) & np.isfinite(after)
    summary.update(
        {
            "objective_comparable_count": int(comparable.sum()),
            "objective_improved_indices": np.flatnonzero(
                comparable & (after > before)
            ).tolist(),
            "objective_decreased_indices": np.flatnonzero(
                comparable & (after < before)
            ).tolist(),
            "changed_trial_indices": np.flatnonzero(indices != 0).tolist(),
        }
    )
    selected.metadata["candidate_selection"] = {
        "selection_version": 1,
        "scope": "independent per-target selection among observed candidates",
        "objective": objective,
        "priority": [
            "quality_pass",
            "ik_success",
            "finite_objective_descending",
            "finite_residual_ascending",
            "trial_order",
        ],
        "selection_time_quality_thresholds": deepcopy(thresholds),
        "selection_time_summary": summary,
        "ik_candidate_counts": ik.sum(axis=0).tolist(),
        "quality_candidate_counts_at_selection": quality.sum(axis=0).tolist(),
        "trials": [
            {
                "name": name,
                "selected_count": int(selected_counts[index]),
                "metadata": _normalize(result.metadata),
                "initialization": study._initialization(index),
            }
            for index, (name, result) in enumerate(zip(study.names, study.results))
        ],
    }
    check_cancelled(cancel_event)
    return selected


def _validated_candidates(study, *, cancel_event=None):
    """Shared measurement checks for candidate selection and diversity reports."""
    from .stability import IKStabilityResult

    check_cancelled(cancel_event)
    if not isinstance(study, IKStabilityResult):
        raise ValueError("candidate selection requires an IKStabilityResult")
    ik, quality = study._flags(cancel_event=cancel_event)
    baseline = study.results[0]
    if _numeric_array(baseline.points, "points").shape != (ik.shape[1], 3):
        raise ValueError("candidate points must have shape (N, 3)")
    if baseline.target_poses is not None and _numeric_array(
        baseline.target_poses, "target_poses"
    ).shape != (ik.shape[1], 4, 4):
        raise ValueError("candidate target poses must have shape (N, 4, 4)")
    tolerance = baseline.metadata.get("solver_settings", {}).get("tolerance")
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, Real)
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("candidate selection requires a finite positive IK tolerance")
    for index, result in enumerate(study.results):
        check_cancelled(cancel_event)
        # Measurement arrays are public and may have changed since construction.
        # Validate their current schema before indexing them or comparing gates.
        joints = _numeric_array(result.joint_positions, "joint_positions")
        if joints.ndim != 2 or len(joints) != len(baseline.points):
            raise ValueError("candidate joint configurations must have shape (N, DoF)")
        if result.joint_positions.shape != baseline.joint_positions.shape:
            raise ValueError("candidate joint configurations must share shape")
        if result.metrics.keys() != baseline.metrics.keys():
            raise ValueError("candidate metric schemas must match")
        for name, values in result.metrics.items():
            if _numeric_array(values, name).shape != ik[index].shape:
                raise ValueError(f"candidate metric {name!r} must have shape (N,)")
        residual = _numeric_array(result.residual, "residual")
        if residual.shape != ik[index].shape:
            raise ValueError("candidate residuals must have shape (N,)")
        if np.any(
            ik[index]
            & (~np.isfinite(residual) | (residual < 0) | (residual > tolerance))
        ) or not (np.isfinite(result.joint_positions[ik[index]]).all()):
            raise ValueError(
                "successful candidates require finite joints and residuals "
                "within IK tolerance"
            )
    check_cancelled(cancel_event)
    return ik, quality
