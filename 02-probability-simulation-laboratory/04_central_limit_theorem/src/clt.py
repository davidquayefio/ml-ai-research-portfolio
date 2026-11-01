"""Module 4: the Central Limit Theorem.

For i.i.d. ``X_i`` with mean ``mu`` and finite variance ``sigma^2``,
``Z_n = sqrt(n) (Xbar_n - mu) / sigma`` converges in distribution to ``N(0, 1)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

Draw = Callable[[np.random.Generator, tuple], np.ndarray]


def from_frozen(dist) -> tuple[Draw, float, float]:
    """Return ``(draw, mu, sigma)`` for a frozen scipy distribution with finite variance."""
    mu, var = (float(v) for v in dist.stats(moments="mv"))

    def draw(rng, size):
        return dist.rvs(size=size, random_state=rng)

    return draw, mu, float(np.sqrt(var))


def sample_means(draw: Draw, n: int, reps: int, rng: np.random.Generator, max_block: int = 4_000_000) -> np.ndarray:
    """``reps`` independent sample means of size ``n``.

    Chunked so that at most ``max_block`` random numbers are held in memory.
    """
    rows_per_block = max(1, max_block // n)
    out = np.empty(reps)
    for start in range(0, reps, rows_per_block):
        m = min(rows_per_block, reps - start)
        out[start : start + m] = draw(rng, (m, n)).mean(axis=1)
    return out


def standardized_means(draw: Draw, mu: float, sigma: float, n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """``Z_n = sqrt(n) (Xbar_n - mu) / sigma`` for ``reps`` replications."""
    return np.sqrt(n) * (sample_means(draw, n, reps, rng) - mu) / sigma


def ks_distance_to_normal(z: np.ndarray) -> float:
    """Kolmogorov distance ``sup_t |F_n(t) - Phi(t)|`` between the sample and N(0,1)."""
    return float(stats.kstest(z, "norm").statistic)


def student_t_coverage(
    draw: Draw, mu: float, n: int, reps: int, rng: np.random.Generator, level: float = 0.95, max_block: int = 4_000_000
) -> float:
    """Empirical coverage of the usual t-interval ``Xbar +/- t_{n-1} s / sqrt(n)`` for the mean."""
    tcrit = stats.t.ppf(0.5 + level / 2.0, df=n - 1)
    rows_per_block = max(1, max_block // n)
    covered = 0
    for start in range(0, reps, rows_per_block):
        m = min(rows_per_block, reps - start)
        x = draw(rng, (m, n))
        xbar = x.mean(axis=1)
        s = x.std(axis=1, ddof=1)
        half = tcrit * s / np.sqrt(n)
        covered += int(np.sum((xbar - half <= mu) & (mu <= xbar + half)))
    return covered / reps


def prob_beyond(z: np.ndarray, k: float = 3.0) -> float:
    """Empirical ``P(|Z| > k)``; compare with ``2 (1 - Phi(k))`` for a standard normal."""
    return float(np.mean(np.abs(z) > k))


def iqr(x: np.ndarray) -> float:
    """Interquartile range (a robust scale measure that exists even without a mean)."""
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)
