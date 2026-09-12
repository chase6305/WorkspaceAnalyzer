"""Task-target reachability and dexterity assessment without simulation coupling."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np

from ._cancellation import check_cancelled
from ._validation import require_integer
from .analyzer import AnalysisResult, _numpy
from .cache import analysis_cache_key
from .kinematics import _validate_homogeneous_rows, _validate_rotation_matrices
from .metrics import _dexterity_from_jacobian, _validate_task_weights

_QUALITY_FIELDS = (
    "minimum_singular_value",
    "minimum_isotropy",
    "minimum_joint_limit_margin",
    "require_full_rank",
)
_QUALITY_METRICS = (
    "manipulability",
    "minimum_singular_value",
    "isotropy",
    "condition_number",
    "joint_limit_margin",
)


@dataclass(frozen=True)
class ReachabilityConfig:
    """IK budget and quality gates for the returned configuration of each target.

    Quality gates evaluate the selected IK solution, without searching additional
    branches for a higher-quality solution. Weights scale Jacobian rows; pose
    metrics mix translation and rotation unless meaningful scaling is supplied.
    """

    position_only: bool = True
    batch_size: int = 1024
    restarts: int = 4
    rescue_restarts: int = 16
    rescue_rounds: int = 3
    random_seed: int = 42
    dexterity_task: str = "position"
    dexterity_weights: np.ndarray | None = None
    minimum_singular_value: float | None = None
    minimum_isotropy: float | None = None
    minimum_joint_limit_margin: float | None = None
    require_full_rank: bool = False

    def __post_init__(self):
        for name in ("batch_size", "restarts"):
            require_integer(getattr(self, name), name, minimum=1)
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("rescue_restarts", "rescue_rounds", "random_seed"):
            require_integer(getattr(self, name), name)
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("position_only", "require_full_rank"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be Boolean")
        weights = _validate_task_weights(self.dexterity_task, self.dexterity_weights)
        if weights is not None:
            weights = weights.copy()
            weights.setflags(write=False)
            object.__setattr__(self, "dexterity_weights", weights)
        for name in (
            "minimum_singular_value",
            "minimum_isotropy",
            "minimum_joint_limit_margin",
        ):
            value = getattr(self, name)
            if value is not None:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{name} must be finite and non-negative")
                if name != "minimum_singular_value" and value > 1:
                    raise ValueError(f"{name} must lie in [0, 1]")
                object.__setattr__(self, name, float(value))


def analyze_targets(
    solver,
    targets,
    config=None,
    *,
    seed=None,
    weights=None,
    base_from_targets=None,
    cache=None,
    cancel_event=None,
    progress_callback=None,
):
    check_cancelled(cancel_event)
    config = config or ReachabilityConfig()
    _validate_task_weights(
        config.dexterity_task, config.dexterity_weights, dtype=solver.config.dtype
    )
    targets = _real_array(targets, "targets", dtype=solver.config.dtype)
    if targets.shape in {(3,), (4, 4)}:
        targets = targets[None]
    poses = targets.ndim == 3 and targets.shape[1:] == (4, 4)
    if not poses and not (targets.ndim == 2 and targets.shape[1:] == (3,)):
        raise ValueError("targets must have shape (N, 3) or (N, 4, 4)")
    if len(targets) == 0:
        raise ValueError("targets must contain at least one target")
    if not config.position_only and not poses:
        raise ValueError("full-pose assessment requires (N, 4, 4) targets")
    if poses:
        _validate_homogeneous_rows(targets, "numpy")
        _validate_rotation_matrices(targets[:, :3, :3], "numpy")
    transform = _real_array(
        np.eye(4) if base_from_targets is None else base_from_targets,
        "base_from_targets",
        dtype=solver.config.dtype,
    )
    if transform.shape != (4, 4):
        raise ValueError("base_from_targets must have shape (4, 4)")
    _validate_homogeneous_rows(transform[None], "numpy")
    _validate_rotation_matrices(transform[None, :3, :3], "numpy")
    count = len(targets)
    task_weights = _real_array(
        np.ones(count) if weights is None else weights, "weights", dtype=float
    )
    if (
        task_weights.shape != (count,)
        or np.any(task_weights < 0)
        or not np.any(task_weights > 0)
    ):
        raise ValueError(
            "weights must have shape (N,), be non-negative, "
            "and include a positive value"
        )
    if seed is not None:
        seed = _real_array(seed, "seed", dtype=solver.config.dtype)
        if seed.shape not in {(solver.dof,), (1, solver.dof), (count, solver.dof)}:
            raise ValueError("seed must have shape (DoF,), (1, DoF), or (N, DoF)")
    key = None
    if cache is not None:
        key = analysis_cache_key(
            solver,
            "target_measurements",
            {
                "measurement_schema": 1,
                "config": {
                    name: value
                    for name, value in vars(config).items()
                    if name not in _QUALITY_FIELDS
                },
                "targets": _digest(targets),
                "seed": None if seed is None else _digest(seed),
                "base_from_targets": transform,
            },
        )
        cached = cache.get(key)
        if cached is not None:
            try:
                cached = reassess_quality(
                    cached,
                    weights=task_weights,
                    cancel_event=cancel_event,
                    **{name: getattr(config, name) for name in _QUALITY_FIELDS},
                )
            except ValueError:
                # A loadable archive may still lack required measurement arrays.
                cached = None
            if cached is not None:
                cached.metadata["cache_hit"] = True
                if progress_callback is not None:
                    progress_callback(1.0)
                check_cancelled(cancel_event)
                return cached
    if poses:
        targets = transform @ targets
        points = targets[:, :3, 3].copy()
    else:
        targets = targets @ transform[:3, :3].T + transform[:3, 3]
        points = targets
    if not np.isfinite(targets).all():
        raise ValueError("transformed targets must be finite")
    joints = np.empty((count, solver.dof), dtype=solver.config.dtype)
    reachable = np.empty(count, dtype=bool)
    residual = np.empty(count, dtype=solver.config.dtype)
    metric_names = (
        "manipulability",
        "minimum_singular_value",
        "isotropy",
        "condition_number",
        "joint_limit_margin",
        "position_error",
        "rotation_error",
    )
    metrics = {
        name: np.full(count, np.nan, dtype=solver.config.dtype) for name in metric_names
    }
    rank = np.full(count, -1, dtype=np.int64)
    task_dimension = 6 if config.dexterity_task == "pose" else 3
    for start in range(0, count, config.batch_size):
        check_cancelled(cancel_event)
        stop = min(start + config.batch_size, count)
        selection = slice(start, stop)
        batch = targets[selection]
        if not poses:
            batch = np.broadcast_to(
                np.eye(4, dtype=solver.config.dtype), (stop - start, 4, 4)
            ).copy()
            batch[:, :3, 3] = points[selection]
        batch_seed = (
            seed[selection]
            if seed is not None and seed.ndim == 2 and len(seed) > 1
            else seed
        )
        ik = solver.inverse(
            batch,
            seed=batch_seed,
            position_only=config.position_only,
            restarts=config.restarts,
            rescue_restarts=config.rescue_restarts,
            rescue_rounds=config.rescue_rounds,
            random_seed=config.random_seed + start,
            cancel_event=cancel_event,
        )
        check_cancelled(cancel_event)
        joints[selection] = _numpy(ik.positions)
        reachable[selection] = _numpy(ik.success)
        residual[selection] = _numpy(ik.residual)
        good = np.flatnonzero(reachable[selection])
        achieved = np.empty((stop - start, 4, 4), dtype=solver.config.dtype)
        if len(good):
            q = solver._configuration_array(joints[start + good])
            fk, jacobian = solver.forward_with_jacobian(q)
            quality = _dexterity_from_jacobian(
                solver,
                q,
                jacobian,
                task=config.dexterity_task,
                weights=config.dexterity_weights,
            )
            achieved[good] = _numpy(fk)
            for name in metric_names[:5]:
                metrics[name][start + good] = _numpy(getattr(quality, name))
            singular = _numpy(quality.singular_values)
            tolerance = (
                np.finfo(singular.dtype).eps
                * max(task_dimension, solver.dof)
                * singular[:, :1]
            )
            rank[start + good] = np.count_nonzero(singular > tolerance, axis=1)
        bad = np.flatnonzero(~reachable[selection])
        if len(bad):
            achieved[bad] = _numpy(solver.forward(joints[start + bad]))
        metrics["position_error"][selection] = np.linalg.norm(
            achieved[:, :3, 3] - points[selection], axis=1
        )
        if not config.position_only:
            metrics["rotation_error"][selection] = _rotation_error(
                batch[:, :3, :3], achieved[:, :3, :3]
            )
        check_cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback(stop / count)
    check_cancelled(cancel_event)
    metrics["task_rank"] = rank
    summary, quality_pass = _assessment_summary(
        reachable, metrics, task_weights, config
    )
    manipulability = metrics.pop("manipulability")
    metrics.update(quality_pass=quality_pass, task_rank=rank, task_weight=task_weights)
    result = AnalysisResult(
        points=points,
        joint_positions=joints,
        manipulability=manipulability,
        reachable=reachable,
        residual=residual,
        metrics=metrics,
        target_poses=targets if poses else None,
        metadata={
            "mode": "targets",
            "robot": solver.model.name,
            "base_link": solver.base_link,
            "tip_link": solver.tip_link,
            "joint_names": solver.joint_names,
            "coordinate_frame": solver.base_link,
            "base_from_targets": transform.tolist(),
            "backend": solver.backend,
            "device": solver.device,
            "dtype": solver.config.dtype,
            "position_only": config.position_only,
            "batch_size": config.batch_size,
            "random_seed": config.random_seed,
            "restarts": config.restarts,
            "rescue_restarts": config.rescue_restarts,
            "rescue_rounds": config.rescue_rounds,
            "solver_settings": vars(solver.config).copy(),
            "dexterity_task": config.dexterity_task,
            "dexterity_weights": None
            if config.dexterity_weights is None
            else config.dexterity_weights.tolist(),
            "quality_thresholds": {
                name: getattr(config, name) for name in _QUALITY_FIELDS
            },
            "quality_scope": "selected_ik_solution",
            "collision_checked": False,
            "assessment": summary,
        },
    )
    check_cancelled(cancel_event)
    if cache is not None:
        result.metadata["cache_hit"] = False
        cache.put(key, result)
    return result


def reassess_quality(
    result,
    *,
    minimum_singular_value=None,
    minimum_isotropy=None,
    minimum_joint_limit_margin=None,
    require_full_rank=False,
    weights=None,
    cancel_event=None,
):
    """Apply a complete set of quality gates to saved target measurements.

    Unspecified thresholds are disabled. IK, FK and Jacobians are not recomputed.
    The source is not modified; returned results share its measurement arrays.
    """
    check_cancelled(cancel_event)
    reachable, metrics, task_weights, config = _quality_inputs(
        result,
        weights=weights,
        minimum_singular_value=minimum_singular_value,
        minimum_isotropy=minimum_isotropy,
        minimum_joint_limit_margin=minimum_joint_limit_margin,
        require_full_rank=require_full_rank,
    )
    task_weights = task_weights.copy()
    summary, accepted = _assessment_summary(reachable, metrics, task_weights, config)
    metadata = deepcopy(result.metadata)
    metadata.update(
        quality_thresholds={name: getattr(config, name) for name in _QUALITY_FIELDS},
        assessment=summary,
        quality_reassessed=True,
    )
    # Derived summaries, if attached by a caller, must be rebuilt for new gates.
    metadata.pop("orientation_coverage", None)
    metrics = {**result.metrics, "quality_pass": accepted, "task_weight": task_weights}
    check_cancelled(cancel_event)
    return replace(result, metadata=metadata, metrics=metrics)


def _quality_inputs(result, *, weights=None, **thresholds):
    """Validate quality inputs without copying measurements or building reports."""
    reachable, metrics = _assessment_arrays(result)
    config = ReachabilityConfig(
        position_only=result.metadata["position_only"],
        dexterity_task=result.metadata["dexterity_task"],
        **thresholds,
    )
    task_weights = np.asarray(
        _numpy(metrics["task_weight"] if weights is None else weights)
    )
    if task_weights.dtype.kind not in "iuf" or not np.isfinite(task_weights).all():
        raise ValueError("weights must contain finite real numeric values")
    if (
        task_weights.shape != reachable.shape
        or np.any(task_weights < 0)
        or not np.any(task_weights > 0)
    ):
        raise ValueError(
            "weights must have shape (N,), be non-negative, "
            "and include a positive value"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        task_weights = np.asarray(task_weights, dtype=float)
    if not np.isfinite(task_weights).all() or not np.any(task_weights > 0):
        raise ValueError("weights must remain finite with positive mass in float64")
    return reachable, metrics, task_weights, config


def _assessment_arrays(result):
    if (
        not isinstance(result, AnalysisResult)
        or result.metadata.get("mode") != "targets"
    ):
        raise ValueError("a target assessment result is required")
    count = len(result.points)
    if count == 0 or result.reachable is None or result.metrics is None:
        raise ValueError("target assessment measurements are missing")
    reachable = np.asarray(result.reachable)
    if reachable.shape != (count,) or reachable.dtype.kind != "b":
        raise ValueError("target assessment reachable flags must have shape (N,)")
    if result.metadata.get("dexterity_task") not in {
        "position",
        "rotation",
        "pose",
    } or not isinstance(result.metadata.get("position_only"), bool):
        raise ValueError("target assessment task metadata is invalid")
    arrays = {**result.metrics, "manipulability": result.manipulability}
    for name in (
        *_QUALITY_METRICS,
        "position_error",
        "rotation_error",
        "task_rank",
        "task_weight",
    ):
        value = np.asarray(arrays.get(name))
        if value.shape != (count,) or value.dtype.kind not in "iuf":
            raise ValueError(f"target assessment metric {name!r} must have shape (N,)")
        arrays[name] = value
    return reachable, arrays


def _validated_quality_flags(result, reachable, metrics):
    """Check saved acceptance against its current measurements and complete gates."""
    accepted = np.asarray(metrics.get("quality_pass"))
    if (
        accepted.shape != reachable.shape
        or accepted.dtype.kind != "b"
        or np.any(accepted & ~reachable)
    ):
        raise ValueError("quality_pass must be Boolean and imply IK success")
    thresholds = result.metadata.get("quality_thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(_QUALITY_FIELDS):
        raise ValueError("complete quality thresholds are required")
    _, _, _, config = _quality_inputs(result, **thresholds)
    expected, _ = _quality_gates(reachable, metrics, config)
    if not np.array_equal(accepted, expected):
        raise ValueError(
            "quality flags disagree with current measurements; reassess quality first"
        )
    return accepted


def _quality_gates(reachable, metrics, config):
    """Apply the same gates for reports and stale-candidate validation."""
    rank = metrics["task_rank"]
    dimension = 6 if config.dexterity_task == "pose" else 3
    quality_pass = reachable.copy()
    failures = {}
    for setting, metric in (
        ("minimum_singular_value", "minimum_singular_value"),
        ("minimum_isotropy", "isotropy"),
        ("minimum_joint_limit_margin", "joint_limit_margin"),
    ):
        threshold = getattr(config, setting)
        if threshold is not None:
            # Overflowed or unavailable measurements cannot establish a finite
            # quality threshold, even though +inf compares above that threshold.
            values = metrics[metric]
            passed = np.isfinite(values) & (values >= threshold)
            failures[setting] = int(np.count_nonzero(reachable & ~passed))
            quality_pass &= passed
    if config.require_full_rank:
        passed = rank == dimension
        failures["full_task_rank"] = int(np.count_nonzero(reachable & ~passed))
        quality_pass &= passed
    return quality_pass, failures


def _assessment_summary(reachable, metrics, task_weights, config):
    quality_pass, failures = _quality_gates(reachable, metrics, config)
    rank = metrics["task_rank"]
    dimension = 6 if config.dexterity_task == "pose" else 3
    normalized = task_weights / np.max(task_weights)
    total = np.sum(normalized)
    return {
        "samples": len(reachable),
        "ik_success_count": int(reachable.sum()),
        "ik_failure_count": int(np.count_nonzero(~reachable)),
        "quality_pass_count": int(quality_pass.sum()),
        "quality_rejected_count": int(np.count_nonzero(reachable & ~quality_pass)),
        "ik_success_rate": float(reachable.mean()),
        "quality_pass_rate": float(quality_pass.mean()),
        "weighted_ik_success_rate": float(np.sum(normalized[reachable]) / total),
        "weighted_quality_pass_rate": float(np.sum(normalized[quality_pass]) / total),
        "threshold_failures": failures,
        "full_task_rank_count": int(np.count_nonzero(rank == dimension)),
        "task_dimension": dimension,
        "quality_statistics": {
            name: _statistics(metrics[name][reachable]) for name in _QUALITY_METRICS
        },
        "position_error": _statistics(metrics["position_error"]),
        "rotation_error": None
        if config.position_only
        else _statistics(metrics["rotation_error"]),
    }, quality_pass


def _real_array(value, name, *, dtype):
    array = np.asarray(_numpy(value))
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.array(array, dtype=dtype, copy=True, order="C")
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must remain finite in {np.dtype(dtype).name}")
    return converted


def _digest(array):
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _rotation_error(target, achieved):
    rotation = target @ achieved.swapaxes(1, 2)
    skew = np.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        axis=1,
    )
    return np.arctan2(
        np.linalg.norm(skew, axis=1) / 2, (np.trace(rotation, axis1=1, axis2=2) - 1) / 2
    )


def _statistics(values):
    finite = values[np.isfinite(values)]
    p05, median, p95 = (
        (None, None, None)
        if not len(finite)
        else map(float, np.percentile(finite, [5, 50, 95]))
    )
    return {
        "count": len(values),
        "finite_count": len(finite),
        "infinite_count": int(np.isinf(values).sum()),
        "min": None if not len(finite) else float(np.min(finite)),
        "p05": p05,
        "median": median,
        "p95": p95,
        "max": None if not len(finite) else float(np.max(finite)),
    }
