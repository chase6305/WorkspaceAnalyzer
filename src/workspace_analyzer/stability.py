"""Paired IK budget and initialization studies on an unchanged target set."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from ._cancellation import check_cancelled
from ._validation import require_integer
from .analyzer import AnalysisResult, _numpy
from .cache import _model_digest, _normalize
from .kinematics import KinematicsSolver
from .reachability import (
    ReachabilityConfig,
    _assessment_arrays,
    _real_array,
    _validated_quality_flags,
    analyze_targets,
)


@dataclass(frozen=True)
class IKTrial:
    """Named overrides of the common solver budget and initialization.

    None inherits the common budget setting. seed=None uses the solver's default
    initial configuration; it never inherits a solution from another trial.
    """

    name: str
    max_iterations: int | None = None
    random_seed: int | None = None
    restarts: int | None = None
    rescue_restarts: int | None = None
    rescue_rounds: int | None = None
    seed: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("trial name must be a nonempty string")
        for name in (
            "max_iterations",
            "random_seed",
            "restarts",
            "rescue_restarts",
            "rescue_rounds",
        ):
            value = getattr(self, name)
            if value is not None:
                require_integer(
                    value, name, minimum=int(name in {"max_iterations", "restarts"})
                )
                object.__setattr__(self, name, int(value))
        if self.seed is not None:
            raw = np.asarray(_numpy(self.seed))
            dtype = np.float32 if raw.dtype == np.float32 else np.float64
            seed = _real_array(raw, "trial seed", dtype=dtype)
            if seed.ndim not in (1, 2) or not seed.size:
                raise ValueError(
                    "trial seed must have nonempty shape (DoF,) or (N, DoF)"
                )
            seed.setflags(write=False)
            object.__setattr__(self, "seed", seed)


@dataclass
class IKStabilityResult:
    names: tuple[str, ...]
    results: tuple[AnalysisResult, ...]
    initial_seeds: tuple[np.ndarray | None, ...] | None = None

    def _flags(self, *, cancel_event=None):
        check_cancelled(cancel_event)
        if len(self.results) < 2 or len(self.names) != len(self.results):
            raise ValueError("stability requires at least two named results")
        if any(not isinstance(name, str) or not name.strip() for name in self.names):
            raise ValueError("trial names must be nonempty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("trial names must be unique")
        if self.initial_seeds is not None and len(self.initial_seeds) != len(
            self.names
        ):
            raise ValueError("initial seeds must match the trial count")
        baseline = self.results[0]
        flags, qualities = [], []
        for index, result in enumerate(self.results):
            check_cancelled(cancel_event)
            reachable, metrics = _assessment_arrays(result)
            if self.initial_seeds is not None and self.initial_seeds[index] is not None:
                seed = np.asarray(self.initial_seeds[index])
                joints = np.asarray(result.joint_positions)
                if joints.ndim != 2 or len(joints) != len(reachable):
                    raise ValueError("joint measurements must have shape (N, DoF)")
                dof = joints.shape[1]
                if (
                    seed.dtype.kind not in "iuf"
                    or not seed.size
                    or not np.isfinite(seed).all()
                    or seed.shape not in {(dof,), (1, dof), (len(reachable), dof)}
                ):
                    raise ValueError(
                        "initial seed must be finite with compatible target/joint shape"
                    )
            quality = _validated_quality_flags(result, reachable, metrics)
            for key in (
                "robot",
                "base_link",
                "tip_link",
                "coordinate_frame",
                "dtype",
                "position_only",
                "dexterity_task",
                "dexterity_weights",
                "quality_thresholds",
                "backend",
                "device",
                "batch_size",
                "stability_model_digest",
                "joint_names",
                "joint_kinds",
                "base_from_targets",
                "collision_checked",
            ):
                if _normalize(result.metadata.get(key)) != _normalize(
                    baseline.metadata.get(key)
                ):
                    raise ValueError(f"stability results must share {key}")

            def settings(item):
                return {
                    key: value
                    for key, value in item.metadata.get("solver_settings", {}).items()
                    if key not in {"max_iterations", "random_seed"}
                }

            if settings(result) != settings(baseline):
                raise ValueError(
                    "stability results must share non-budget solver settings"
                )
            if result.residual is None:
                raise ValueError("stability results must include IK residuals")
            residual = np.asarray(result.residual)
            if residual.shape != reachable.shape or residual.dtype.kind not in "iuf":
                raise ValueError(
                    "stability residuals must be real arrays of shape (N,)"
                )
            if (
                not np.array_equal(result.points, baseline.points)
                or not np.array_equal(result.target_poses, baseline.target_poses)
                or not np.array_equal(
                    metrics["task_weight"], baseline.metrics["task_weight"]
                )
            ):
                raise ValueError(
                    "stability results must use identical targets, order, "
                    "and task weights"
                )
            flags.append(reachable)
            qualities.append(quality)
        check_cancelled(cancel_event)
        return np.stack(flags), np.stack(qualities)

    def to_dict(self, *, cancel_event=None) -> dict:
        """Build a report with incremental metric reductions and cancellation."""
        ik, quality = self._flags(cancel_event=cancel_event)
        summary = {"trials": len(self.results), "targets": ik.shape[1]}
        per_target = {"target_indices": list(range(ik.shape[1]))}
        comparisons = []
        for label, values in (("ik", ik), ("quality", quality)):
            check_cancelled(cancel_event)
            counts = values.sum(axis=0)
            disagreement = (counts > 0) & (counts < len(values))
            summary.update(
                {
                    f"{label}_all_success_count": int(
                        np.count_nonzero(counts == len(values))
                    ),
                    f"{label}_no_success_count": int(np.count_nonzero(counts == 0)),
                    f"{label}_disagreement_count": int(disagreement.sum()),
                    f"{label}_disagreement_rate": float(disagreement.mean()),
                }
            )
            per_target[f"{label}_success_counts"] = counts.tolist()
            per_target[f"{label}_disagreement_indices"] = np.flatnonzero(
                disagreement
            ).tolist()
        for index in range(1, len(self.names)):
            check_cancelled(cancel_event)
            row = {"baseline": self.names[0], "trial": self.names[index]}
            for label, values in (("ik", ik), ("quality", quality)):
                gained = np.flatnonzero(~values[0] & values[index])
                lost = np.flatnonzero(values[0] & ~values[index])
                row[f"{label}_gained_indices"] = gained.tolist()
                row[f"{label}_lost_indices"] = lost.tolist()
                row[f"{label}_gained_count"] = len(gained)
                row[f"{label}_lost_count"] = len(lost)
            comparisons.append(row)
        for metric in ("residual", "isotropy", "joint_limit_margin"):
            # Only target-sized numeric working arrays are needed. In particular,
            # do not stack all trials and create another float array per reduction.
            arrays = [
                np.asarray(
                    result.residual if metric == "residual" else result.metrics[metric]
                )
                for result in self.results
            ]
            dtype = np.result_type(np.float64, *(values.dtype for values in arrays))
            counts = np.zeros(ik.shape[1], dtype=np.int64)
            minimum = np.full(ik.shape[1], np.inf, dtype=dtype)
            maximum = np.full(ik.shape[1], -np.inf, dtype=dtype)
            for index, values in enumerate(arrays):
                check_cancelled(cancel_event)
                valid = np.isfinite(values)
                if metric != "residual":
                    valid &= ik[index]
                counts += valid
                finite = np.where(valid, values, np.nan)
                np.fmin(minimum, finite, out=minimum)
                np.fmax(maximum, finite, out=maximum)
            per_target[f"{metric}_finite_counts"] = counts.tolist()
            for name, reduced in (("minimum", minimum), ("maximum", maximum)):
                serialized = reduced.astype(float, copy=False).astype(object)
                serialized[~np.isfinite(reduced)] = None
                per_target[f"{metric}_{name}"] = serialized.tolist()
        report = {
            "study_version": 1,
            "scope": "observed IK and selected-solution quality across trials",
            "rate_weighting": "equal_targets",
            "summary": summary,
            "trials": [
                {
                    "name": name,
                    "file": f"trial_{index:03d}.npz",
                    "metadata": _normalize(result.metadata),
                    "initialization": self._initialization(index),
                }
                for index, (name, result) in enumerate(zip(self.names, self.results))
            ],
            "baseline_comparisons": comparisons,
            "per_target": per_target,
        }
        check_cancelled(cancel_event)
        return report

    def _initialization(self, index):
        if self.initial_seeds is None:
            return {"kind": "unknown"}
        seed = self.initial_seeds[index]
        return (
            {"kind": "solver_default"}
            if seed is None
            else {
                "kind": "provided",
                "array": f"trial_{index:03d}",
                "shape": list(np.shape(seed)),
            }
        )

    def reassess_quality(self, **kwargs):
        """Replace every trial's quality gates using existing measurements."""
        return IKStabilityResult(
            self.names,
            tuple(result.reassess_quality(**kwargs) for result in self.results),
            self.initial_seeds,
        )

    def select_solutions(self, *, objective="joint_limit_margin", cancel_event=None):
        """Select candidates offline, retaining quality gates and source trials."""
        from .selection import select_ik_solutions

        return select_ik_solutions(self, objective=objective, cancel_event=cancel_event)

    def summarize_diversity(self, config=None, *, joint_kinds=None, cancel_event=None):
        """Count tolerance-separated configurations without discarding candidates."""
        from .diversity import summarize_ik_diversity

        return summarize_ik_diversity(
            self, config, joint_kinds=joint_kinds, cancel_event=cancel_event
        )

    def save(self, directory: str | Path) -> None:
        report = self.to_dict()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for index, result in enumerate(self.results):
            result.save(directory / f"trial_{index:03d}.npz")
        np.savez_compressed(
            directory / "initial_seeds.npz",
            **{
                f"trial_{index:03d}": seed
                for index, seed in enumerate(self.initial_seeds or ())
                if seed is not None
            },
        )
        (directory / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if (
            not isinstance(report, dict)
            or type(report.get("study_version")) is not int
            or report["study_version"] != 1
            or not isinstance(report.get("trials"), list)
            or len(report["trials"]) < 2
        ):
            raise ValueError("unsupported stability study report")
        for trial in report["trials"]:
            if (
                not isinstance(trial, dict)
                or not isinstance(trial.get("name"), str)
                or not trial["name"].strip()
                or not isinstance(trial.get("initialization"), dict)
                or trial["initialization"].get("kind")
                not in ("unknown", "provided", "solver_default")
            ):
                raise ValueError(
                    "invalid stability trial name or initialization metadata"
                )
        names = tuple(trial["name"] for trial in report["trials"])
        if len(set(names)) != len(names):
            raise ValueError("trial names must be unique")
        modes = [
            trial.get("initialization", {}).get("kind") for trial in report["trials"]
        ]
        seeds = None
        if not all(mode == "unknown" for mode in modes):
            if any(mode not in {"provided", "solver_default"} for mode in modes):
                raise ValueError("invalid trial initialization metadata")
            seeds = []
            try:
                archive = np.load(directory / "initial_seeds.npz", allow_pickle=False)
                if isinstance(archive, np.ndarray):
                    raise ValueError("initial seeds must be an NPZ archive")
                with archive:
                    for index, mode in enumerate(modes):
                        seed = (
                            archive[f"trial_{index:03d}"]
                            if mode == "provided"
                            else None
                        )
                        seed = IKTrial(names[index], seed=seed).seed
                        if seed is not None and list(seed.shape) != report["trials"][
                            index
                        ]["initialization"].get("shape"):
                            raise ValueError(
                                "saved seed shape does not match the report"
                            )
                        seeds.append(seed)
            except (BadZipFile, EOFError, KeyError, zlib.error) as exc:
                raise ValueError(f"invalid initial seed archive: {exc}") from exc
            seeds = tuple(seeds)
        result = cls(
            names,
            tuple(
                AnalysisResult.load(directory / f"trial_{i:03d}.npz")
                for i in range(len(names))
            ),
            seeds,
        )
        # Rebuild derived statistics from the actual files, not saved summaries.
        result._flags()
        return result


def analyze_ik_stability(
    solver,
    targets,
    trials,
    config=None,
    *,
    weights=None,
    base_from_targets=None,
    cache=None,
    cancel_event=None,
    progress_callback=None,
):
    """Evaluate a fixed target snapshot under independently initialized IK trials.

    Solver instances share the already loaded model, preserving chain, backend,
    device, dtype and non-budget solver settings. The input solver is unchanged.
    Trial names do not affect measurement cache keys.
    """
    check_cancelled(cancel_event)
    trials = tuple(trials)
    if len(trials) < 2 or any(not isinstance(trial, IKTrial) for trial in trials):
        raise ValueError("stability requires at least two IKTrial configurations")
    if len({trial.name for trial in trials}) != len(trials):
        raise ValueError("trial names must be unique")
    config = config or ReachabilityConfig()
    common_runtime = replace(
        solver.config, backend=solver.backend, device=solver.device
    )
    model, base_link, tip_link = solver.model, solver.base_link, solver.tip_link
    joint_kinds = [joint.kind for joint in solver.active_joints]
    targets = _real_array(targets, "targets", dtype=common_runtime.dtype)
    if targets.shape in {(3,), (4, 4)}:
        targets = targets[None]
    if not (
        targets.ndim == 2
        and targets.shape[1:] == (3,)
        or targets.ndim == 3
        and targets.shape[1:] == (4, 4)
    ) or not len(targets):
        raise ValueError("targets must have nonempty shape (N, 3) or (N, 4, 4)")
    for trial in trials:
        check_cancelled(cancel_event)
        if trial.seed is not None and trial.seed.shape not in {
            (solver.dof,),
            (1, solver.dof),
            (len(targets), solver.dof),
        }:
            raise ValueError("trial seed must have shape (DoF,), (1, DoF), or (N, DoF)")
        if trial.seed is not None:
            # Check every trial in runtime precision before any solve/cache I/O.
            # Keep the original seed snapshot for provenance; analyze_targets
            # makes the effective copy when that individual trial is evaluated.
            _real_array(trial.seed, "trial seed", dtype=common_runtime.dtype)
    weights = None if weights is None else _real_array(weights, "weights", dtype=float)
    transform = (
        None
        if base_from_targets is None
        else _real_array(
            base_from_targets, "base_from_targets", dtype=common_runtime.dtype
        )
    )
    results = []
    model_digest = _model_digest(model)
    for index, trial in enumerate(trials):
        check_cancelled(cancel_event)
        overrides = {
            name: getattr(trial, name)
            for name in ("random_seed", "restarts", "rescue_restarts", "rescue_rounds")
            if getattr(trial, name) is not None
        }
        reachability = replace(config, **overrides)
        runtime = replace(
            common_runtime,
            max_iterations=trial.max_iterations or common_runtime.max_iterations,
            random_seed=reachability.random_seed,
        )
        candidate = KinematicsSolver(model, base_link, tip_link, runtime)
        callback = (
            None
            if progress_callback is None
            else (
                lambda value, index=index: progress_callback(
                    (index + value) / len(trials)
                )
            )
        )
        result = analyze_targets(
            candidate,
            targets,
            reachability,
            seed=trial.seed,
            weights=weights,
            base_from_targets=transform,
            cache=cache,
            cancel_event=cancel_event,
            progress_callback=callback,
        )
        results.append(
            replace(
                result,
                metadata={
                    **result.metadata,
                    "stability_model_digest": model_digest,
                    "joint_kinds": joint_kinds.copy(),
                },
            )
        )
    check_cancelled(cancel_event)
    study = IKStabilityResult(
        tuple(trial.name for trial in trials),
        tuple(results),
        tuple(trial.seed for trial in trials),
    )
    study._flags(cancel_event=cancel_event)
    check_cancelled(cancel_event)
    return study
