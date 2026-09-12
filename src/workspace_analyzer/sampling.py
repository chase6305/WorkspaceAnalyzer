"""Deterministic joint-space samplers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._validation import require_integer


class SamplingStrategy(str, Enum):
    RANDOM = "random"
    UNIFORM = "uniform"
    HALTON = "halton"
    SOBOL = "sobol"
    LATIN_HYPERCUBE = "lhs"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class SamplingConfig:
    strategy: SamplingStrategy = SamplingStrategy.RANDOM
    num_samples: int = 10_000
    batch_size: int = 4096
    seed: int = 42

    def __post_init__(self):
        require_integer(self.num_samples, "num_samples", minimum=1)
        require_integer(self.batch_size, "batch_size", minimum=1)
        require_integer(self.seed, "seed")
        SamplingStrategy(self.strategy)


def sample(
    limits: np.ndarray, config: SamplingConfig, *, allow_fixed: bool = False
) -> np.ndarray:
    """Sample intervals, optionally preserving fixed axes without sampling them.

    Fixed axes do not consume random dimensions or Cartesian grid levels. An
    entirely fixed region returns the requested number of identical points.
    """
    limits = np.asarray(limits, dtype=float)
    if limits.ndim != 2 or limits.shape[1] != 2 or len(limits) == 0:
        raise ValueError("limits must have shape (dimensions, 2) with dimensions > 0")
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("limits must be finite with lower <= upper")
    fixed = limits[:, 0] == limits[:, 1]
    if fixed.any():
        if not allow_fixed:
            raise ValueError("limits must be finite with lower < upper")
        points = np.broadcast_to(limits[:, 0], (config.num_samples, len(limits))).copy()
        if not fixed.all():
            points[:, ~fixed] = sample(limits[~fixed], config)
        return points
    lo, hi = limits[:, 0], limits[:, 1]
    rng = np.random.default_rng(config.seed)
    strategy = SamplingStrategy(config.strategy)
    if strategy == SamplingStrategy.RANDOM:
        unit = rng.random((config.num_samples, len(limits)))
    elif strategy == SamplingStrategy.UNIFORM:
        side = max(2, int(np.ceil(config.num_samples ** (1 / len(limits)))))
        while side ** len(limits) < config.num_samples:
            side += 1
        # Decode only the requested prefix of the Cartesian grid. A full mesh
        # would allocate side**dimensions points before truncating to this size.
        indices = np.arange(config.num_samples)
        levels = np.linspace(0, 1, side)
        unit = np.empty((config.num_samples, len(limits)))
        for dimension in range(len(limits) - 1, -1, -1):
            indices, digit = np.divmod(indices, side)
            unit[:, dimension] = levels[digit]
    elif strategy == SamplingStrategy.GAUSSIAN:
        unit = rng.normal(0.5, 1 / 6, (config.num_samples, len(limits)))
        np.clip(unit, 0, 1, out=unit)
    else:
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise ImportError(
                "quasi-random sampling requires workspace-analyzer[sampling]"
            ) from exc
        engines = {
            SamplingStrategy.HALTON: qmc.Halton,
            SamplingStrategy.SOBOL: qmc.Sobol,
            SamplingStrategy.LATIN_HYPERCUBE: qmc.LatinHypercube,
        }
        engine = engines[strategy](d=len(limits), seed=config.seed)
        if strategy == SamplingStrategy.SOBOL:
            power = int(np.ceil(np.log2(config.num_samples)))
            unit = engine.random_base2(power)[: config.num_samples]
            # Do not retain the unrequested tail of the power-of-two allocation.
            if config.num_samples != 2**power:
                unit = unit.copy()
        else:
            unit = engine.random(config.num_samples)
    # Every sampler creates a private array; reuse it for the affine mapping.
    with np.errstate(over="ignore"):
        width = hi - lo
    wide = ~np.isfinite(width)
    if wide.any():
        # A convex combination stays finite even when the interval width does
        # not fit in float64. Keep the usual mapping unchanged for other axes.
        fraction = unit[:, wide]
        unit[:, wide] = (1 - fraction) * lo[wide] + fraction * hi[wide]
        unit[:, ~wide] = unit[:, ~wide] * width[~wide] + lo[~wide]
    else:
        unit *= width
        unit += lo
    return unit
