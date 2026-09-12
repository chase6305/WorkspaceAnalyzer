"""Reproducible joint seeds and tolerance-based IK configuration diversity."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from ._cancellation import check_cancelled
from ._validation import require_integer
from .selection import _validated_candidates
from .stability import IKTrial


@dataclass(frozen=True)
class IKDiversityConfig:
    angular_tolerance_rad: float = 1e-3
    linear_tolerance_m: float = 1e-4

    def __post_init__(self):
        for name in ("angular_tolerance_rad", "linear_tolerance_m"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))


def make_ik_seed_trials(solver, count, *, random_seed=42, joint_fraction=0.5):
    """Create independent uniform joint initial guesses without running IK.

    Each (DoF,) seed is broadcast to every target when used in a study. Budgets
    and IK restart seeds inherit the common configuration. Increasing count
    preserves the prefix of the generated sequence. The seed generator has its
    own RNG stream, independent of FK target sampling and solver restart RNGs.
    """
    require_integer(count, "count", minimum=1)
    require_integer(random_seed, "random_seed")
    if (
        isinstance(joint_fraction, (bool, np.bool_))
        or not isinstance(joint_fraction, Real)
        or not np.isfinite(joint_fraction)
        or not 0 < joint_fraction <= 0.5
    ):
        raise ValueError("joint_fraction must lie in (0, 0.5]")
    limits = np.asarray(solver.joint_limits, dtype=float)
    if (
        limits.shape != (solver.dof, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("seed generation requires finite ordered joint limits")
    rng = np.random.default_rng(np.random.SeedSequence([random_seed, 0x494B]))
    fraction = rng.uniform(-joint_fraction, joint_fraction, (count, solver.dof)) + 0.5
    seeds = ((1 - fraction) * limits[:, 0] + fraction * limits[:, 1]).astype(
        solver.config.dtype
    )
    return tuple(
        IKTrial(f"joint_seed_{index:03d}", seed=seed)
        for index, seed in enumerate(seeds)
    )


def summarize_ik_diversity(study, config=None, *, joint_kinds=None, cancel_event=None):
    """Group successful observed configurations around ordered representatives.

    A candidate joins the earliest representative within tolerance in EVERY joint.
    Only continuous joints use shortest circular distance. Failed trials receive
    representative index -1. Grouping is order dependent and is not IK branch
    enumeration; it never discards measurements or alters candidate selection.
    """
    check_cancelled(cancel_event)
    config = IKDiversityConfig() if config is None else config
    if not isinstance(config, IKDiversityConfig):
        raise ValueError("config must be an IKDiversityConfig")
    ik, quality = _validated_candidates(study, cancel_event=cancel_event)
    stored = study.results[0].metadata.get("joint_kinds")
    kinds = stored if joint_kinds is None else joint_kinds
    dof = study.results[0].joint_positions.shape[1]
    try:
        kinds = [] if isinstance(kinds, str) else list(kinds)
    except TypeError:
        kinds = []
    if len(kinds) != dof or any(
        not isinstance(kind, str) or kind not in {"revolute", "prismatic", "continuous"}
        for kind in kinds
    ):
        raise ValueError(
            "joint_kinds must identify each revolute/prismatic/continuous joint"
        )
    if stored is not None and kinds != list(stored):
        raise ValueError("joint_kinds disagree with the saved model metadata")
    tolerances = np.array(
        [
            config.linear_tolerance_m
            if kind == "prismatic"
            else config.angular_tolerance_rad
            for kind in kinds
        ]
    )
    periodic = np.array([kind == "continuous" for kind in kinds])
    has_periodic = bool(periodic.any())
    trials, targets = ik.shape
    labels = np.full((trials, targets), -1, dtype=np.int64)
    accepted_groups = np.zeros((trials, targets), dtype=bool)
    unique = np.zeros(targets, dtype=np.int64)
    by_trial = []
    representative_trials = []
    for index, result in enumerate(study.results):
        check_cancelled(cancel_event)
        remaining = ik[index].copy()
        remaining_count = int(remaining.sum())
        for prior in representative_trials:
            check_cancelled(cancel_event)
            if remaining_count == 0:
                break
            rows = np.flatnonzero(remaining & (labels[prior] == prior))
            if not len(rows):
                continue
            current = result.joint_positions[rows].astype(np.float64, copy=False)
            reference = (
                study.results[prior]
                .joint_positions[rows]
                .astype(np.float64, copy=False)
            )
            # Very distant finite coordinates can overflow subtraction; inf is
            # correctly outside tolerance. Periodic subtraction stays bounded.
            with np.errstate(over="ignore"):
                distance = np.abs(current - reference)
            if has_periodic:
                wrapped = np.abs(
                    np.remainder(current[:, periodic], 2 * np.pi)
                    - np.remainder(reference[:, periodic], 2 * np.pi)
                )
                distance[:, periodic] = np.minimum(wrapped, 2 * np.pi - wrapped)
            matched = rows[np.all(distance <= tolerances, axis=1)]
            labels[index, matched] = prior
            remaining[matched] = False
            remaining_count -= len(matched)
        labels[index, remaining] = index
        # Trials without any new representative cannot match later candidates.
        # Keep their measurements and quality contributions in the report.
        if remaining_count:
            representative_trials.append(index)
        unique += remaining
        accepted = np.flatnonzero(quality[index])
        representatives = labels[index, accepted]
        new_quality = ~accepted_groups[representatives, accepted]
        accepted_groups[representatives, accepted] = True
        by_trial.append(
            {
                "trial_index": index,
                "name": study.names[index],
                "ik_success_count": int(ik[index].sum()),
                "new_configuration_count": remaining_count,
                "duplicate_candidate_count": int(
                    np.count_nonzero(ik[index] & ~remaining)
                ),
                "new_quality_configuration_count": int(new_quality.sum()),
                "cumulative_configuration_count": int(unique.sum()),
            }
        )
    quality_unique = accepted_groups.sum(axis=0)
    ik_counts = ik.sum(axis=0)
    check_cancelled(cancel_event)
    return {
        "diversity_version": 1,
        "scope": "observed configurations at joint tolerances; not branch enumeration",
        "method": "ordered_greedy_representatives",
        "joint_kinds": kinds,
        "joint_tolerances": tolerances.tolist(),
        "configuration": vars(config).copy(),
        "trial_names": list(study.names),
        "summary": {
            "targets": targets,
            "trials": trials,
            "ik_candidate_count": int(ik_counts.sum()),
            "configuration_count": int(unique.sum()),
            "duplicate_candidate_count": int((ik_counts - unique).sum()),
            "quality_candidate_count": int(quality.sum()),
            "quality_configuration_count": int(quality_unique.sum()),
            "targets_with_multiple_configurations": int(np.count_nonzero(unique > 1)),
            "targets_without_ik": int(np.count_nonzero(unique == 0)),
        },
        "per_target": {
            "ik_candidate_counts": ik_counts.tolist(),
            "configuration_counts": unique.tolist(),
            "quality_configuration_counts": quality_unique.tolist(),
        },
        "representative_trial_indices": labels.tolist(),
        "by_trial": by_trial,
    }
