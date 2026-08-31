"""Deterministic joint-space samplers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


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


def sample(limits: np.ndarray, config: SamplingConfig) -> np.ndarray:
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    lo, hi = limits[:, 0], limits[:, 1]
    rng = np.random.default_rng(config.seed)
    strategy = SamplingStrategy(config.strategy)
    if strategy == SamplingStrategy.RANDOM:
        unit = rng.random((config.num_samples, len(limits)))
    elif strategy == SamplingStrategy.UNIFORM:
        side = max(2, int(np.ceil(config.num_samples ** (1 / len(limits)))))
        grid = np.meshgrid(*([np.linspace(0, 1, side)] * len(limits)), indexing="ij")
        unit = np.stack(grid, axis=-1).reshape(-1, len(limits))[: config.num_samples]
    elif strategy == SamplingStrategy.GAUSSIAN:
        unit = np.clip(rng.normal(0.5, 1 / 6, (config.num_samples, len(limits))), 0, 1)
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
        else:
            unit = engine.random(config.num_samples)
    return lo + unit * (hi - lo)
