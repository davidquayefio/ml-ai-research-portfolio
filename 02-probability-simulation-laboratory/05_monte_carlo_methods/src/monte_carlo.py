"""Module 5: Monte Carlo methods.

Core idea: to estimate ``theta = E[g(X)]`` draw ``X_1..X_n`` i.i.d. and use
``theta_hat = (1/n) sum g(X_i)``. It is unbiased, has standard error
``sd(g(X)) / sqrt(n)`` and (by the CLT) an approximate normal sampling distribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import gamma, pi
from typing import Callable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MCResult:
    """A Monte Carlo estimate with its uncertainty."""

    estimate: float
    std_error: float
    ci_low: float
    ci_high: float
    n: int

    def covers(self, truth: float) -> bool:
        return self.ci_low <= truth <= self.ci_high


def summarize(samples: np.ndarray, level: float = 0.95) -> MCResult:
    """Mean, standard error and normal-approximation CI of i.i.d. per-sample values."""
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    m = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(0.5 + level / 2.0)
    return MCResult(m, se, m - z * se, m + z * se, n)


# ----------------------------------------------------------------------------
# Constants and probabilities
# ----------------------------------------------------------------------------
def estimate_pi(n: int, rng: np.random.Generator) -> MCResult:
    """Hit-or-miss: ``pi = 4 P(U1^2 + U2^2 <= 1)`` for independent uniforms on [0,1]."""
    xy = rng.random((n, 2))
    return summarize(4.0 * (np.sum(xy**2, axis=1) <= 1.0))


def estimate_real_root_probability(n: int, rng: np.random.Generator) -> MCResult:
    """``P(x^2 + b x + c has real roots)`` for ``b, c ~ U(0,1)``; exact value ``1/12``.

    Real roots iff ``b^2 >= 4c``, i.e. ``c <= b^2/4``; so ``P = int_0^1 b^2/4 db = 1/12``.
    """
    b, c = rng.random(n), rng.random(n)
    return summarize(b * b >= 4.0 * c)


def ball_volume_hit_or_miss(d: int, n: int, rng: np.random.Generator) -> MCResult:
    """Volume of the unit d-ball by hit-or-miss in ``[-1, 1]^d`` (cube volume ``2^d``)."""
    x = rng.uniform(-1.0, 1.0, size=(n, d))
    return summarize((2.0**d) * (np.sum(x**2, axis=1) <= 1.0))


def ball_volume_exact(d: int) -> float:
    """``V_d = pi^{d/2} / Gamma(d/2 + 1)``."""
    return pi ** (d / 2.0) / gamma(d / 2.0 + 1.0)


# ----------------------------------------------------------------------------
# Numerical integration:  I = int_a^b f(x) dx = (b - a) E[f(U)],  U ~ Unif(a, b)
# ----------------------------------------------------------------------------
def mc_integral(f: Callable, a: float, b: float, n: int, rng: np.random.Generator) -> MCResult:
    """Plain Monte Carlo integration."""
    u = a + (b - a) * rng.random(n)
    return summarize((b - a) * f(u))


def antithetic_integral(f: Callable, a: float, b: float, n: int, rng: np.random.Generator) -> MCResult:
    """Antithetic variates using ``n`` function evaluations (``n/2`` pairs).

    Each sample is ``(b-a) * (f(U) + f(a+b-U)) / 2``. Since ``a + b - U`` is also uniform,
    the estimator is unbiased, and its variance is smaller whenever
    ``Cov(f(U), f(a+b-U)) < 0`` (e.g. monotone ``f``).
    """
    m = n // 2
    u = a + (b - a) * rng.random(m)
    return summarize((b - a) * 0.5 * (f(u) + f(a + b - u)))


def control_variate_integral(
    f: Callable,
    a: float,
    b: float,
    n: int,
    rng: np.random.Generator,
    control: Callable | None = None,
    control_mean: float | None = None,
) -> tuple[MCResult, float]:
    """Control variate estimator ``Y - beta (C - E[C])`` with ``beta = Cov(Y, C) / Var(C)``.

    Default control: ``C = U`` (the sampled point) with known mean ``(a+b)/2``.
    Returns the result and the fitted ``beta``.
    """
    u = a + (b - a) * rng.random(n)
    y = (b - a) * f(u)
    c = u if control is None else control(u)
    mean_c = (a + b) / 2.0 if control_mean is None else control_mean
    beta = float(np.cov(y, c)[0, 1] / np.var(c, ddof=1))
    return summarize(y - beta * (c - mean_c)), beta


# ----------------------------------------------------------------------------
# Rare events and importance sampling
# ----------------------------------------------------------------------------
def naive_normal_tail(c: float, n: int, rng: np.random.Generator) -> MCResult:
    """Plain Monte Carlo for ``P(Z > c)``, ``Z ~ N(0,1)``."""
    return summarize(rng.standard_normal(n) > c)


def importance_sampling_normal_tail(c: float, n: int, rng: np.random.Generator) -> MCResult:
    """Importance sampling for ``P(Z > c)`` using the proposal ``N(c, 1)``.

    ``P(Z > c) = E_q[1{X > c} w(X)]`` with weight ``w(x) = phi(x)/phi(x-c) = exp(-c x + c^2/2)``.
    """
    x = rng.normal(c, 1.0, size=n)
    w = np.exp(-c * x + 0.5 * c * c)
    return summarize((x > c) * w)


# ----------------------------------------------------------------------------
# Experiment utilities
# ----------------------------------------------------------------------------
def ci_coverage(estimator: Callable[[np.random.Generator], MCResult], truth: float, reps: int, rng: np.random.Generator) -> float:
    """Fraction of ``reps`` independent runs whose CI contains ``truth``."""
    return float(np.mean([estimator(rng).covers(truth) for _ in range(reps)]))


def timed(fn: Callable, *args, **kwargs):
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
