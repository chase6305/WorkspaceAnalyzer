"""Deterministic, paired pose perturbations for numerical robustness studies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from ._cancellation import check_cancelled
from .analyzer import _numpy
from .kinematics import _validate_homogeneous_rows, _validate_rotation_matrices
from .reachability import _assessment_arrays, _real_array, _validated_quality_flags


@dataclass(frozen=True)
class PosePerturbations:
    """Reference plus independent +/- XYZ translations and rotations per pose.

    Inputs and output targets are in the solver base frame. Translation axes may
    be expressed in the base or reference tool frame; rotation offsets change
    orientation about the TCP, never rotating the TCP position around the base.
    Zero amplitudes disable that family. Samples are not an error distribution.
    """

    reference_poses: np.ndarray
    translation_m: float = 0.01
    rotation_rad: float = np.pi / 36
    translation_frame: str = "base"
    rotation_frame: str = "tool"
    targets: np.ndarray = field(init=False, repr=False)
    variant_names: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        for name in ("translation_m", "rotation_rad"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if self.rotation_rad >= np.pi:
            raise ValueError("rotation_rad must lie in [0, pi)")
        for name in ("translation_frame", "rotation_frame"):
            if getattr(self, name) not in ("base", "tool"):
                raise ValueError(f"{name} must be 'base' or 'tool'")
        raw = np.asarray(_numpy(self.reference_poses))
        dtype = np.float32 if raw.dtype == np.float32 else np.float64
        poses = _real_array(raw, "reference_poses", dtype=dtype)
        if poses.shape == (4, 4):
            poses = poses[None]
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
            raise ValueError("reference_poses must have nonempty shape (N, 4, 4)")
        _validate_homogeneous_rows(poses, "numpy")
        _validate_rotation_matrices(poses[:, :3, :3], "numpy")
        names = ["reference"]
        for family, amplitude in (
            ("translation", self.translation_m),
            ("rotation", self.rotation_rad),
        ):
            if amplitude > 0:
                names.extend(
                    f"{family}_{axis}{sign}" for axis in "xyz" for sign in "+-"
                )
        targets = np.repeat(poses[:, None], len(names), axis=1)
        index = 1
        for axis in range(3):
            if self.translation_m == 0:
                break
            direction = np.eye(3, dtype=dtype)[axis]
            if self.translation_frame == "tool":
                direction = poses[:, :3, :3] @ direction
            for sign in (1, -1):
                targets[:, index, :3, 3] += sign * self.translation_m * direction
                index += 1
        for axis in range(3):
            if self.rotation_rad == 0:
                break
            for sign in (1, -1):
                offset = np.eye(3, dtype=dtype)
                a, b = (axis + 1) % 3, (axis + 2) % 3
                c, s = np.cos(self.rotation_rad), sign * np.sin(self.rotation_rad)
                offset[a, a] = offset[b, b] = c
                offset[a, b], offset[b, a] = -s, s
                rotation = poses[:, :3, :3]
                targets[:, index, :3, :3] = (
                    rotation @ offset
                    if self.rotation_frame == "tool"
                    else offset @ rotation
                )
                index += 1
        if not np.isfinite(targets).all():
            raise ValueError("perturbed targets must be finite")
        targets = targets.reshape(-1, 4, 4)
        poses.setflags(write=False)
        targets.setflags(write=False)
        object.__setattr__(self, "reference_poses", poses)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "variant_names", tuple(names))

    def summarize(self, result, *, cancel_event=None) -> dict:
        """Summarize aligned full-pose results without further kinematics calls.

        Pairing is checked against the generated base-frame targets, including
        their order. Rates use equal sample counts, not result task weights.
        Undefined conditional rates and missing finite metrics become JSON null.
        """
        check_cancelled(cancel_event)
        reachable, metrics = _assessment_arrays(result)
        if result.metadata["position_only"] or result.target_poses is None:
            raise ValueError("perturbation summaries require a full-pose assessment")
        poses = np.asarray(result.target_poses)
        if (
            poses.shape != self.targets.shape
            or not np.array_equal(poses, self.targets.astype(poses.dtype, copy=False))
            or not np.array_equal(result.points, poses[:, :3, 3])
        ):
            raise ValueError(
                "result targets must match the perturbations in base-frame order"
            )
        accepted = _validated_quality_flags(result, reachable, metrics)
        shape = (len(self.reference_poses), len(self.variant_names))
        ik, quality = reachable.reshape(shape), accepted.reshape(shape)
        by_variant = []
        for index, name in enumerate(self.variant_names):
            check_cancelled(cancel_event)
            row = {"name": name}
            for metric, values in (("ik", ik), ("quality", quality)):
                row.update(
                    {
                        f"{metric}_success_count": int(values[:, index].sum()),
                        f"{metric}_lost_from_reference": int(
                            (values[:, 0] & ~values[:, index]).sum()
                        ),
                        f"{metric}_gained_from_reference": int(
                            (~values[:, 0] & values[:, index]).sum()
                        ),
                    }
                )
            by_variant.append(row)
        per_reference = {"reference_ids": list(range(shape[0]))}
        for name, values in (("ik", ik), ("quality", quality)):
            check_cancelled(cancel_event)
            per_reference[f"reference_{name}"] = values[:, 0].tolist()
            per_reference[f"all_variants_{name}"] = values.all(axis=1).tolist()
            per_reference[f"perturbed_{name}_rate"] = (
                values[:, 1:].mean(axis=1).tolist()
                if shape[1] > 1
                else [None] * shape[0]
            )
        for name in ("minimum_singular_value", "isotropy", "joint_limit_margin"):
            check_cancelled(cancel_event)
            values = metrics[name].reshape(shape)
            if values.dtype.kind != "f":
                values = values.astype(float)
            worst = np.min(
                values, axis=1, where=ik & np.isfinite(values), initial=np.inf
            )
            serialized = worst.astype(float, copy=False).astype(object)
            serialized[~np.isfinite(worst)] = None
            per_reference[f"worst_solved_{name}"] = serialized.tolist()
        report = {
            "metadata": {
                "translation_m": self.translation_m,
                "rotation_rad": self.rotation_rad,
                "translation_frame": self.translation_frame,
                "rotation_frame": self.rotation_frame,
                "coordinate_frame": result.metadata.get("coordinate_frame"),
                "single_axis_samples_only": True,
                "rate_weighting": "equal_samples",
                "quality_scope": result.metadata.get("quality_scope"),
                "quality_thresholds": deepcopy(
                    result.metadata.get("quality_thresholds", {})
                ),
                "collision_checked": result.metadata.get("collision_checked", False),
            },
            "summary": {
                "references": shape[0],
                "variants_per_reference": shape[1],
                "perturbed_samples": shape[0] * (shape[1] - 1),
                "ik": _paired_summary(ik),
                "quality": _paired_summary(quality),
            },
            "by_variant": by_variant,
            "per_reference": per_reference,
        }
        check_cancelled(cancel_event)
        return report


def _paired_summary(values):
    baseline = values[:, 0]
    perturbed = values[:, 1:]
    successes = perturbed.sum(axis=1)
    reference_count = int(baseline.sum())
    success_count = int(successes.sum())
    retained_count = int(successes[baseline].sum())
    retained_samples = reference_count * perturbed.shape[1]
    all_count = int(np.count_nonzero(baseline & (successes == perturbed.shape[1])))
    return {
        "reference_success_count": reference_count,
        "perturbed_success_count": success_count,
        "perturbed_success_rate": success_count / perturbed.size
        if perturbed.size
        else None,
        "all_variants_pass_count": all_count,
        "all_variants_pass_rate": all_count / len(values) if perturbed.size else None,
        "retained_success_rate": retained_count / retained_samples
        if retained_samples
        else None,
        "lost_pairs": retained_samples - retained_count,
        "gained_pairs": success_count - retained_count,
    }
