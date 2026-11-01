"""Module 2: random variables and probability distributions.

Frozen ``scipy.stats`` distributions provide the *theory* (pmf/pdf, cdf, moments);
NumPy/SciPy sampling provides the *experiment*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# The eleven distributions of the laboratory
# ----------------------------------------------------------------------------
def default_distributions() -> dict:
    """Name -> frozen ``scipy.stats`` distribution, with fixed illustrative parameters.

    Parameterisation notes (scipy uses ``loc``/``scale``):
    * Exponential(rate) -> ``expon(scale=1/rate)``
    * Gamma(shape k, scale theta) -> ``gamma(k, scale=theta)``; mean k*theta
    * Lognormal(mu, sigma): ``lognorm(s=sigma, scale=exp(mu))`` -- ``X = exp(N(mu, sigma^2))``
    * Geometric counts the trial of the first success, support {1, 2, ...}
    """
    return {
        "Bernoulli(0.3)": stats.bernoulli(0.3),
        "Binomial(10, 0.3)": stats.binom(10, 0.3),
        "Geometric(0.2)": stats.geom(0.2),
        "Poisson(4)": stats.poisson(4),
        "Uniform(0, 1)": stats.uniform(0, 1),
        "Exponential(rate 1.5)": stats.expon(scale=1 / 1.5),
        "Normal(1, 2^2)": stats.norm(1, 2),
        "Gamma(2, scale 1.5)": stats.gamma(2, scale=1.5),
        "Beta(2, 5)": stats.beta(2, 5),
        "Lognormal(0, 0.5^2)": stats.lognorm(s=0.5, scale=1.0),
        "Student t(5)": stats.t(5),
    }


def is_discrete(dist) -> bool:
    """True for frozen discrete distributions."""
    return isinstance(dist.dist, stats.rv_discrete)


def draw_samples(dist, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` i.i.d. samples from a frozen scipy distribution using ``rng``."""
    return dist.rvs(size=n, random_state=rng)


# ----------------------------------------------------------------------------
# Moments
# ----------------------------------------------------------------------------
def theoretical_moments(dist) -> dict:
    """Mean, variance, skewness and *excess* kurtosis from theory."""
    m, v, s, k = dist.stats(moments="mvsk")
    return {"mean": float(m), "var": float(v), "skew": float(s), "ex_kurt": float(k)}


def sample_moments(x: np.ndarray) -> dict:
    """Sample mean, variance (ddof=1), skewness and excess kurtosis."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(x.mean()),
        "var": float(x.var(ddof=1)),
        "skew": float(stats.skew(x)),
        "ex_kurt": float(stats.kurtosis(x)),
    }


def moment_table(dists: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Compare theoretical and simulated moments for every distribution."""
    rows = []
    for name, d in dists.items():
        th = theoretical_moments(d)
        sm = sample_moments(draw_samples(d, n, rng))
        row = {"distribution": name}
        for key in ("mean", "var", "skew", "ex_kurt"):
            row[f"{key}_theory"] = th[key]
            row[f"{key}_sim"] = sm[key]
        rows.append(row)
    return pd.DataFrame(rows).set_index("distribution")


# ----------------------------------------------------------------------------
# Goodness of fit
# ----------------------------------------------------------------------------
def tv_distance_discrete(x: np.ndarray, dist) -> float:
    """Total variation distance ``0.5 * sum_k |empirical pmf(k) - pmf(k)|`` for integer data."""
    x = np.asarray(x).astype(int)
    support = np.arange(x.min(), x.max() + 1)
    emp = np.bincount(x - x.min(), minlength=support.size) / x.size
    th = dist.pmf(support)
    tail_mass = 1.0 - th.sum()  # theoretical mass outside the observed range
    return float(0.5 * (np.abs(emp - th).sum() + max(tail_mass, 0.0)))


def ks_test(x: np.ndarray, dist) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic ``sup_t |F_n(t) - F(t)|`` and p-value (continuous dists)."""
    res = stats.kstest(x, dist.cdf)
    return float(res.statistic), float(res.pvalue)


# ----------------------------------------------------------------------------
# Moment-generating functions  M(t) = E[exp(tX)]
# ----------------------------------------------------------------------------
def mgf_catalogue() -> dict:
    """Name -> (frozen distribution, closed-form MGF ``M(t)``)."""
    return {
        "Bernoulli(0.3)": (stats.bernoulli(0.3), lambda t: 1 - 0.3 + 0.3 * np.exp(t)),
        "Binomial(10, 0.3)": (stats.binom(10, 0.3), lambda t: (1 - 0.3 + 0.3 * np.exp(t)) ** 10),
        "Poisson(4)": (stats.poisson(4), lambda t: np.exp(4 * (np.exp(t) - 1))),
        "Exponential(rate 1.5)": (stats.expon(scale=1 / 1.5), lambda t: 1.5 / (1.5 - t)),  # t < 1.5
        "Normal(1, 2^2)": (stats.norm(1, 2), lambda t: np.exp(1 * t + 0.5 * 4 * t**2)),
        "Gamma(2, scale 1.5)": (stats.gamma(2, scale=1.5), lambda t: (1 - 1.5 * t) ** (-2.0)),  # t < 1/1.5
        "Uniform(0, 1)": (
            stats.uniform(0, 1),
            lambda t: np.where(np.abs(t) < 1e-12, 1.0, np.expm1(t) / np.where(np.abs(t) < 1e-12, 1.0, t)),
        ),
    }


def empirical_mgf(x: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Monte Carlo estimate of ``M(t) = E[exp(tX)]`` on a grid."""
    x = np.asarray(x, dtype=float)
    return np.array([np.exp(t * x).mean() for t in t_grid])


def mgf_moments_finite_difference(mgf, h: float = 1e-3) -> tuple[float, float]:
    """Mean and variance from ``M'(0)`` and ``M''(0)`` by central finite differences.

    ``E[X] = M'(0)``, ``E[X^2] = M''(0)``, ``Var X = M''(0) - M'(0)^2``.
    """
    m0 = float(mgf(0.0))
    mp, mm = float(mgf(h)), float(mgf(-h))
    first = (mp - mm) / (2 * h)
    second = (mp - 2 * m0 + mm) / h**2
    return first, second - first**2


# ----------------------------------------------------------------------------
# Transformations and sampling
# ----------------------------------------------------------------------------
def inverse_transform_sample(ppf, n: int, rng: np.random.Generator) -> np.ndarray:
    """If ``U ~ Uniform(0,1)`` and ``F`` is a continuous CDF, then ``F^{-1}(U) ~ F``."""
    return ppf(rng.random(n))


def inverse_transform_exponential(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Exponential(rate) via ``X = -ln(1 - U) / rate`` (``F^{-1}(u) = -ln(1-u)/rate``)."""
    return -np.log1p(-rng.random(n)) / rate


def inverse_transform_geometric(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Geometric(p) on {1, 2, ...} via ``X = ceil(ln(1-U) / ln(1-p))``."""
    x = np.ceil(np.log1p(-rng.random(n)) / np.log1p(-p)).astype(int)
    return np.maximum(x, 1)


# ----------------------------------------------------------------------------
# Joint distributions: correlation, Cholesky, uncorrelated-but-dependent
# ----------------------------------------------------------------------------
def correlated_normals(n: int, cov: np.ndarray, rng: np.random.Generator, mean=None) -> np.ndarray:
    """Draw ``n`` vectors from ``N(mean, cov)`` using the Cholesky factor ``cov = L L^T``.

    If ``Z ~ N(0, I)`` then ``Y = L Z`` has covariance ``L L^T = cov``.
    """
    cov = np.asarray(cov, dtype=float)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, cov.shape[0]))
    y = z @ L.T
    if mean is not None:
        y = y + np.asarray(mean, dtype=float)
    return y


def uncorrelated_but_dependent(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``X ~ N(0,1)`` and ``Y = X^2``: ``Cov(X, Y) = E[X^3] = 0`` yet ``Y`` is a function of ``X``."""
    x = rng.standard_normal(n)
    return x, x**2
