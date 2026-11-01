"""Module 3: the Law of Large Numbers.

Notation: ``X_1, X_2, ...`` i.i.d. with mean ``mu`` and variance ``sigma^2``;
the sample mean is ``Xbar_n = (X_1 + ... + X_n) / n``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

Draw = Callable[[np.random.Generator, tuple], np.ndarray]


@dataclass(frozen=True)
class Sampler:
    """A named distribution with its theoretical mean and variance (``None`` if undefined)."""

    name: str
    draw: Draw
    mean: float | None
    var: float | None


def _bernoulli(rng, size):
    return (rng.random(size) < 0.3).astype(float)


SAMPLERS: dict[str, Sampler] = {
    s.name: s
    for s in [
        Sampler("Bernoulli(0.3)", _bernoulli, 0.3, 0.21),
        Sampler("Uniform(0,1)", lambda rng, size: rng.random(size), 0.5, 1 / 12),
        Sampler("Exponential(1)", lambda rng, size: rng.exponential(1.0, size), 1.0, 1.0),
        Sampler(
            "Lognormal(0,1)",
            lambda rng, size: rng.lognormal(0.0, 1.0, size),
            float(np.exp(0.5)),
            float((np.e - 1.0) * np.e),
        ),
        Sampler("Student t(3)", lambda rng, size: rng.standard_t(3, size), 0.0, 3.0),
        Sampler("Cauchy", lambda rng, size: rng.standard_cauchy(size), None, None),
    ]
}


def running_mean(x: np.ndarray) -> np.ndarray:
    """Running mean along the last axis: ``Xbar_k = (x_1 + ... + x_k) / k``."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x, axis=-1) / np.arange(1, x.shape[-1] + 1)


def running_mean_paths(sampler: Sampler, n: int, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_paths, n)`` array; row ``i`` is the running mean of an independent sequence."""
    return running_mean(sampler.draw(rng, (n_paths, n)))


def sample_means_at(
    sampler: Sampler, n_grid, reps: int, rng: np.random.Generator, chunk: int = 250
) -> np.ndarray:
    """Sample means ``Xbar_n`` for each ``n`` in ``n_grid`` over ``reps`` independent replications.

    One long sequence per replication is generated and read at the checkpoints,
    so the columns are the *same* sequences observed at different sample sizes.
    Memory is bounded by processing ``chunk`` replications at a time.
    """
    n_grid = np.asarray(n_grid, dtype=int)
    big_n = int(n_grid.max())
    idx = n_grid - 1
    out = np.empty((reps, n_grid.size))
    for start in range(0, reps, chunk):
        m = min(chunk, reps - start)
        cs = np.cumsum(sampler.draw(rng, (m, big_n)), axis=1)[:, idx]
        out[start : start + m] = cs / n_grid
    return out


def chebyshev_bound(var: float, n: int, eps: float) -> float:
    """Chebyshev: ``P(|Xbar_n - mu| >= eps) <= var / (n eps^2)`` (capped at 1)."""
    return min(1.0, var / (n * eps**2))


def clt_approximation(var: float, n: int, eps: float) -> float:
    """CLT approximation ``P(|Xbar_n - mu| > eps) ~ 2 (1 - Phi(eps sqrt(n) / sigma))``."""
    return float(2.0 * stats.norm.sf(eps * np.sqrt(n) / np.sqrt(var)))


def exceedance_probability(means: np.ndarray, mu: float, eps: float) -> float:
    """Fraction of replications with ``|Xbar_n - mu| > eps``."""
    return float(np.mean(np.abs(means - mu) > eps))


def loglog_slope(n_grid, errors) -> float:
    """Least-squares slope of ``log(error)`` against ``log(n)``. A ``1/sqrt(n)`` rate gives -0.5."""
    return float(np.polyfit(np.log(np.asarray(n_grid, dtype=float)), np.log(np.asarray(errors, dtype=float)), 1)[0])
