"""Small statistical helpers used across the laboratory."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


def standard_error(x) -> float:
    """Standard error of the sample mean: s / sqrt(n), with s the sample SD (ddof=1)."""
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def mean_ci(x, level: float = 0.95) -> tuple[float, float, float]:
    """Normal-approximation confidence interval for the mean.

    Returns ``(mean, lower, upper)``. The interval is
    ``mean +/- z * s / sqrt(n)`` with ``z`` the ``(1+level)/2`` normal quantile.
    """
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    se = standard_error(x)
    z = _st.norm.ppf(0.5 + level / 2.0)
    return m, m - z * se, m + z * se


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Better behaved than the textbook Wald interval when ``n`` is small or the
    proportion is close to 0 or 1.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    z = _st.norm.ppf(0.5 + level / 2.0)
    phat = successes / n
    denom = 1.0 + z**2 / n
    centre = (phat + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(phat * (1.0 - phat) / n + z**2 / (4.0 * n**2)) / denom
    return float(centre - half), float(centre + half)


def ecdf(x) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF. Returns sorted values and ``F_n`` evaluated at them.

    ``F_n(t) = (1/n) * #{i : x_i <= t}``.
    """
    xs = np.sort(np.asarray(x, dtype=float))
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys
