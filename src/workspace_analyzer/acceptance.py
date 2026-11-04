"""Explicit, reproducible pass/fail criteria for measured reachability runs."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class AcceptanceConfig:
    min_success_rate: float | None = None
    min_quality_pass_rate: float | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)
                    or not np.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    raise ValueError(f"{name} must be finite and between 0 and 1")
                object.__setattr__(self, name, float(value))

    @property
    def enabled(self):
        return (
            self.min_success_rate is not None or self.min_quality_pass_rate is not None
        )


def add_acceptance_arguments(parser):
    parser.add_argument(
        "--min-success-rate",
        type=float,
        help="headless acceptance: minimum IK success fraction (0..1)",
    )
    parser.add_argument(
        "--min-quality-pass-rate",
        type=float,
        help="headless acceptance: minimum fraction passing quality gates (0..1)",
    )


def reassess_acceptance(result, overrides=None, *, reset=False):
    """Recompute saved criteria, preserving thresholds not explicitly overridden.

    Reset discards the saved policy before applying overrides. Measurement arrays
    are shared; source metadata and stored verdicts are never modified in place.
    """
    requirements = {}
    if not reset and "acceptance" in result.metadata:
        record = result.metadata["acceptance"]
        try:
            if not isinstance(record, dict) or not isinstance(
                record.get("requirements"), dict
            ):
                raise ValueError("requirements must be an object")
            saved = AcceptanceConfig(**record["requirements"])
            if not saved.enabled:
                raise ValueError("at least one requirement must be enabled")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid saved acceptance: {exc}") from exc
        requirements.update(asdict(saved))
    if overrides is not None:
        requirements.update(
            {
                key: value
                for key, value in asdict(overrides).items()
                if value is not None
            }
        )
    config = AcceptanceConfig(**requirements)
    if reset:
        metadata = deepcopy(result.metadata)
        metadata.pop("acceptance", None)
        if "plane_scan" in metadata:
            metadata["plane_scan"].pop("acceptance", None)
        result = replace(result, metadata=metadata)
    return apply_acceptance(result, config)


def apply_acceptance(result, config):
    """Copy metadata and attach acceptance, leaving shared measurement arrays intact."""
    if not config.enabled:
        return result
    if result.reachable is None:
        raise ValueError("acceptance requires a reachability result")
    count = len(result.points)
    success_rate = float(result.reachable.mean()) if count else None
    quality_rate = None
    if config.min_quality_pass_rate is not None:
        quality = (result.metrics or {}).get("quality_pass")
        if quality is None:
            raise ValueError(
                "quality acceptance requires saved target quality measurements"
            )
        quality = np.asarray(quality)
        if (
            quality.shape != (count,)
            or quality.dtype.kind != "b"
            or np.any(quality & ~result.reachable)
        ):
            raise ValueError("quality_pass must be Boolean and imply IK success")
        quality_rate = float(quality.mean()) if count else None
    failures = []
    if not count:
        failures.append("no targets were evaluated")
    else:
        for name, actual, minimum in (
            ("IK success rate", success_rate, config.min_success_rate),
            ("quality pass rate", quality_rate, config.min_quality_pass_rate),
        ):
            if minimum is not None and actual < minimum:
                failures.append(f"{name} {actual:.6g} is below {minimum:.6g}")
    acceptance = {
        "passed": not failures,
        "requirements": asdict(config),
        "observed": {"success_rate": success_rate, "quality_pass_rate": quality_rate},
        "failures": failures,
    }
    metadata = deepcopy(result.metadata)
    metadata["acceptance"] = acceptance
    if "plane_scan" in metadata:
        metadata["plane_scan"]["acceptance"] = acceptance
    return replace(result, metadata=metadata)
