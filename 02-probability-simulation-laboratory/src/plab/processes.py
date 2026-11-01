"""Brownian motion and geometric Brownian motion simulators (shared code).

Mathematical conventions
------------------------
* A standard Brownian motion (Wiener process) ``W`` satisfies ``W_0 = 0``,
  has independent increments, and ``W_t - W_s ~ N(0, t - s)`` for ``s < t``.
* On a grid of step ``dt = T / n_steps`` we simulate
  ``W_{k+1} - W_k = sqrt(dt) * Z_k`` with ``Z_k`` i.i.d. ``N(0, 1)``. This is
  *exact* in distribution at the grid points (no discretisation error in the
  law of ``W``); only the path between grid points is unobserved.
* Geometric Brownian motion (GBM) solves ``dS = mu S dt + sigma S dW`` and has
  the closed form ``S_t = S_0 exp((mu - sigma^2/2) t + sigma W_t)``.
"""

from __future__ import annotations

import numpy as np


def brownian_increments(n_steps: int, T: float, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Return an ``(n_paths, n_steps)`` array of Brownian increments ``~ N(0, T/n_steps)``."""
    dt = T / n_steps
    return rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)


def brownian_paths(n_steps: int, T: float, n_paths: int, rng: np.random.Generator):
    """Simulate Brownian paths on ``[0, T]``.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
    W : ndarray, shape (n_paths, n_steps + 1), with ``W[:, 0] = 0``.
    """
    dW = brownian_increments(n_steps, T, n_paths, rng)
    W = np.zeros((n_paths, n_steps + 1))
    np.cumsum(dW, axis=1, out=W[:, 1:])
    t = np.linspace(0.0, T, n_steps + 1)
    return t, W


def gbm_from_brownian(S0: float, mu: float, sigma: float, t: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Exact GBM values ``S_t = S0 exp((mu - sigma^2/2) t + sigma W_t)``."""
    return S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


def gbm_exact_paths(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int, rng: np.random.Generator):
    """Simulate GBM paths exactly at the grid points. Returns ``(t, S)``."""
    t, W = brownian_paths(n_steps, T, n_paths, rng)
    return t, gbm_from_brownian(S0, mu, sigma, t, W)


def gbm_terminal(S0: float, mu: float, sigma: float, T: float, z: np.ndarray) -> np.ndarray:
    """Terminal GBM value from standard normal draws ``z``: ``W_T = sqrt(T) z``."""
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def gbm_euler_paths(S0: float, mu: float, sigma: float, dt: float, dW: np.ndarray) -> np.ndarray:
    """Euler-Maruyama scheme ``S_{k+1} = S_k (1 + mu dt + sigma dW_k)``.

    The explicit product form is ``S_k = S_0 * prod_{j<k} (1 + mu dt + sigma dW_j)``.
    Returns an ``(n_paths, n_steps + 1)`` array including the starting value.
    """
    growth = 1.0 + mu * dt + sigma * dW
    S = np.empty((dW.shape[0], dW.shape[1] + 1))
    S[:, 0] = S0
    np.cumprod(growth, axis=1, out=S[:, 1:])
    S[:, 1:] *= S0
    return S
