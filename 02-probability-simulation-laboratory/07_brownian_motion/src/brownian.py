"""Module 7: Brownian motion, quadratic variation and geometric Brownian motion.

The path simulators live in the shared library (``plab.processes``); this file adds
the experiment-specific tools.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from plab.processes import (  # noqa: F401  (re-exported for notebooks)
    brownian_increments,
    brownian_paths,
    gbm_euler_paths,
    gbm_exact_paths,
    gbm_from_brownian,
    gbm_terminal,
)


# ----------------------------------------------------------------------------
# Random walk -> Brownian motion (Donsker's invariance principle)
# ----------------------------------------------------------------------------
def scaled_walk_terminal(n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """``S_n / sqrt(n)`` for a symmetric +/-1 walk, via ``S_n = 2 Binomial(n, 1/2) - n``."""
    return (2.0 * rng.binomial(n, 0.5, size=reps) - n) / np.sqrt(n)


def scaled_walk_path(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One path of ``W^{(n)}(k/n) = S_k / sqrt(n)``, ``k = 0..n`` on ``[0, 1]``."""
    steps = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    return np.arange(n + 1) / n, s / np.sqrt(n)


# ----------------------------------------------------------------------------
# Variation of Brownian paths on nested dyadic partitions
# ----------------------------------------------------------------------------
def coarsen_increments(dW_fine: np.ndarray, n_coarse: int) -> np.ndarray:
    """Sum blocks of fine increments to obtain the increments on a coarser grid.

    Because ``W`` is additive, this gives exactly the increments of the *same* path observed on
    ``n_coarse`` equally spaced points. ``n_coarse`` must divide the number of fine steps.
    """
    n_paths, n_fine = dW_fine.shape
    if n_fine % n_coarse:
        raise ValueError("n_coarse must divide n_fine")
    return dW_fine.reshape(n_paths, n_coarse, n_fine // n_coarse).sum(axis=2)


def quadratic_variation(dW: np.ndarray) -> np.ndarray:
    """``sum_k (W_{t_{k+1}} - W_{t_k})^2`` for each path."""
    return np.sum(dW**2, axis=1)


def total_variation(dW: np.ndarray) -> np.ndarray:
    """``sum_k |W_{t_{k+1}} - W_{t_k}|`` for each path."""
    return np.sum(np.abs(dW), axis=1)


# ----------------------------------------------------------------------------
# Maximum of Brownian motion (reflection principle)
# ----------------------------------------------------------------------------
def reflection_max_probability(a: float, T: float) -> float:
    """``P(max_{s<=T} W_s >= a) = 2 P(W_T >= a) = 2 (1 - Phi(a / sqrt(T)))`` for ``a > 0``."""
    return float(2.0 * stats.norm.sf(a / np.sqrt(T)))


def discrete_max_exceedance(a: float, T: float, n_steps: int, n_paths: int, rng: np.random.Generator, chunk: int = 2000) -> float:
    """Fraction of simulated paths whose *grid* maximum reaches ``a`` (biased low for finite ``n_steps``)."""
    hits = 0
    for start in range(0, n_paths, chunk):
        m = min(chunk, n_paths - start)
        _, W = brownian_paths(n_steps, T, m, rng)
        hits += int(np.sum(W.max(axis=1) >= a))
    return hits / n_paths


# ----------------------------------------------------------------------------
# Geometric Brownian motion
# ----------------------------------------------------------------------------
def gbm_theoretical_moments(S0: float, mu: float, sigma: float, T: float) -> dict:
    """Mean, variance and median of ``S_T`` under GBM.

    ``E[S_T] = S0 e^{mu T}``, ``Var[S_T] = S0^2 e^{2 mu T} (e^{sigma^2 T} - 1)``,
    ``median = S0 e^{(mu - sigma^2/2) T}``.
    """
    return {
        "mean": S0 * np.exp(mu * T),
        "var": S0**2 * np.exp(2 * mu * T) * np.expm1(sigma**2 * T),
        "median": S0 * np.exp((mu - 0.5 * sigma**2) * T),
    }


def euler_strong_error(
    S0: float, mu: float, sigma: float, T: float, level_steps, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Strong error ``E|S_T^{Euler} - S_T^{exact}|`` for several step counts, using nested grids.

    Both schemes see the *same* Brownian path (the finest increments are summed to coarser grids),
    which is required for a strong (pathwise) comparison.
    """
    level_steps = list(level_steps)
    n_fine = max(level_steps)
    dW_fine = brownian_increments(n_fine, T, n_paths, rng)
    W_T = dW_fine.sum(axis=1)
    exact = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
    errors = []
    for n in level_steps:
        dW = coarsen_increments(dW_fine, n)
        euler_T = gbm_euler_paths(S0, mu, sigma, T / n, dW)[:, -1]
        errors.append(float(np.mean(np.abs(euler_T - exact))))
    return np.array(errors)
