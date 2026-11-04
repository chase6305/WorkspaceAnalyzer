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
    count, dimensions = config.num_samples, len(limits)
    fixed = limits[:, 0] == limits[:, 1]
    if fixed.any():
        if not allow_fixed:
            raise ValueError("limits must be finite with lower < upper")
        points = np.broadcast_to(limits[:, 0], (count, dimensions)).copy()
        if not fixed.all():
            points[:, ~fixed] = sample(limits[~fixed], config)
        return points
    lower, upper = limits[:, 0], limits[:, 1]
    strategy = SamplingStrategy(config.strategy)
    if strategy == SamplingStrategy.RANDOM:
        rng = np.random.default_rng(config.seed)
        unit = rng.random((count, dimensions))
    elif strategy == SamplingStrategy.UNIFORM:
        if dimensions == 1:
            unit = np.linspace(0, 1, count)[:, None]
        else:
            side = max(2, int(np.ceil(count ** (1 / dimensions))))
            while side**dimensions < count:
                side += 1
            # Decode only the requested prefix rather than the full Cartesian grid.
            indices = np.arange(count)
            levels = np.linspace(0, 1, side)
            unit = np.empty((count, dimensions))
            for dimension in range(dimensions - 1, -1, -1):
                indices, digits = np.divmod(indices, side)
                unit[:, dimension] = levels[digits]
                # Quotients stay sorted; a zero maximum leaves only leading zeros.
                if indices[-1] == 0:
                    unit[:, :dimension] = 0
                    break
    elif strategy == SamplingStrategy.GAUSSIAN:
        rng = np.random.default_rng(config.seed)
        unit = rng.normal(0.5, 1 / 6, (count, dimensions))
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
        engine = engines[strategy](d=dimensions, seed=config.seed)
        if strategy == SamplingStrategy.SOBOL:
            power = int(np.ceil(np.log2(count)))
            unit = engine.random_base2(power)[:count]
            # Do not retain the unrequested tail of the power-of-two allocation.
            if count != 2**power:
                unit = unit.copy()
        else:
            unit = engine.random(count)
    # Every sampler creates a private array; reuse it for the affine mapping.
    with np.errstate(over="ignore"):
        width = upper - lower
    wide = ~np.isfinite(width)
    if wide.any():
        # A convex combination stays finite even when the interval width does
        # not fit in float64. Keep the usual mapping unchanged for other axes.
        fraction = unit[:, wide]
        unit[:, wide] = (1 - fraction) * lower[wide] + fraction * upper[wide]
        unit[:, ~wide] = unit[:, ~wide] * width[~wide] + lower[~wide]
    else:
        unit *= width
        unit += lower
    return unit
