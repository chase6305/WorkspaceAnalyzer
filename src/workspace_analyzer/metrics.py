"""Kinematic dexterity metrics for single configurations and batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class DexterityResult:
    """Per-configuration Jacobian and joint-limit quality measures."""

    singular_values: object
    manipulability: object
    minimum_singular_value: object
    isotropy: object
    condition_number: object
    joint_limit_margin: object


def dexterity_metrics(
    solver,
    q,
    *,
    task: Literal["position", "rotation", "pose"] = "position",
    weights=None,
) -> DexterityResult:
    """Evaluate dexterity while preserving the solver's NumPy/Torch backend."""
    _validate_task_weights(task, weights, dtype=solver.config.dtype)
    configuration = solver._configuration_array(q)
    single = configuration.ndim == 1
    if single:
        configuration = configuration[None, :]
    result = _dexterity_from_jacobian(
        solver,
        configuration,
        solver._jacobian_batch(configuration),
        task=task,
        weights=weights,
    )
    if single:
        return DexterityResult(
            *(getattr(result, name)[0] for name in result.__dataclass_fields__)
        )
    return result


def _validate_task_weights(task, weights, *, dtype=float):
    if task not in {"position", "rotation", "pose"}:
        raise ValueError("task must be 'position', 'rotation', or 'pose'")
    rows = 6 if task == "pose" else 3
    if weights is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            weights_array = np.asarray(weights, dtype=float)
        if (
            weights_array.shape != (rows,)
            or not np.isfinite(weights_array).all()
            or np.any(weights_array < 0)
            or not np.any(weights_array > 0)
        ):
            raise ValueError(
                f"weights must contain {rows} finite non-negative values "
                f"with at least one positive value in {np.dtype(dtype).name}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(weights_array, dtype=dtype)
        if not np.isfinite(converted).all() or not np.any(converted > 0):
            raise ValueError(
                "weights must remain finite with a positive value in "
                f"{np.dtype(dtype).name}"
            )
        return converted
    return None


def _dexterity_from_jacobian(solver, configuration, jacobian, *, task, weights=None):
    """Reuse a batched geometric Jacobian already computed alongside FK."""
    weights_array = _validate_task_weights(task, weights, dtype=solver.config.dtype)
    if task == "position":
        jacobian = jacobian[:, :3]
    elif task == "rotation":
        jacobian = jacobian[:, 3:]
    if weights is not None:
        # Configurations may keep immutable weight snapshots; Torch expects a
        # writable NumPy buffer when constructing its tensor view.
        jacobian = jacobian * solver._array(weights_array.copy())[None, :, None]

    if solver.backend == "numpy":
        singular = np.linalg.svd(jacobian, compute_uv=False)
        minimum = singular[..., -1]
        maximum = singular[..., 0]
        isotropy = np.divide(
            minimum,
            maximum,
            out=np.zeros_like(minimum),
            where=maximum > 0,
        )
        condition = np.divide(
            maximum,
            minimum,
            out=np.full_like(maximum, np.inf),
            where=minimum
            > np.finfo(maximum.dtype).eps * max(jacobian.shape[1:]) * maximum,
        )
        manipulability = np.prod(singular, axis=-1)
        margin = _joint_limit_margin_numpy(solver, configuration)
    else:
        import torch

        singular = torch.linalg.svdvals(jacobian)
        minimum, maximum = singular[..., -1], singular[..., 0]
        nonzero = maximum > 0
        # torch.where evaluates both branches. Guard the denominator before
        # division so an unused 0/0 or x/0 cannot poison backward with NaNs.
        isotropy = torch.where(
            nonzero, minimum / torch.where(nonzero, maximum, 1.0), 0.0
        )
        well_conditioned = (
            minimum > torch.finfo(minimum.dtype).eps * max(jacobian.shape[1:]) * maximum
        )
        condition = torch.where(
            well_conditioned,
            maximum / torch.where(well_conditioned, minimum, 1.0),
            torch.full_like(maximum, torch.inf),
        )
        manipulability = torch.prod(singular, dim=-1)
        margin = _joint_limit_margin_torch(solver, configuration)

    return DexterityResult(
        singular, manipulability, minimum, isotropy, condition, margin
    )


def _joint_limit_margin_numpy(solver, q):
    limits = solver.joint_limits
    half_range = (limits[:, 1] - limits[:, 0]) * 0.5
    margin = np.minimum(q - limits[:, 0], limits[:, 1] - q) / half_range
    continuous = np.asarray(
        [joint.kind == "continuous" for joint in solver.active_joints]
    )
    margin[:, continuous] = 1.0
    return np.clip(margin, 0.0, 1.0).min(axis=-1)


def _joint_limit_margin_torch(solver, q):
    import torch

    limits = solver._array(solver.joint_limits)
    half_range = (limits[:, 1] - limits[:, 0]) * 0.5
    margin = torch.minimum(q - limits[:, 0], limits[:, 1] - q) / half_range
    continuous = torch.as_tensor(
        [joint.kind == "continuous" for joint in solver.active_joints],
        dtype=torch.bool,
        device=q.device,
    )
    margin[:, continuous] = 1.0
    return margin.clamp(0.0, 1.0).amin(dim=-1)
