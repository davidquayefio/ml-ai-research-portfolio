"""Covariance and mean estimators.

Research purpose: the central object of the project is the sensitivity of
optimal portfolios to estimated inputs. Different estimators trade bias for
variance in different ways; this module makes them interchangeable.
All functions take a (T x N) array/DataFrame of returns and return an
(N x N) covariance matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sample_covariance(returns) -> np.ndarray:
    """Unbiased sample covariance (ddof=1). MLE would divide by T instead."""
    return np.cov(_arr(returns), rowvar=False, ddof=1)


def sample_mean(returns) -> np.ndarray:
    return _arr(returns).mean(axis=0)


def ledoit_wolf_covariance(returns) -> np.ndarray:
    """Ledoit-Wolf (2004) linear shrinkage towards a scaled identity."""
    return LedoitWolf().fit(_arr(returns)).covariance_


def ledoit_wolf_shrinkage_intensity(returns) -> float:
    return float(LedoitWolf().fit(_arr(returns)).shrinkage_)


def oas_covariance(returns) -> np.ndarray:
    """Oracle Approximating Shrinkage (Chen et al., 2010)."""
    return OAS().fit(_arr(returns)).covariance_


def constant_correlation_shrinkage(returns, delta: float = 0.5) -> np.ndarray:
    """Shrink the sample covariance towards a constant-correlation target.

    delta in [0, 1] is a user-chosen intensity (not estimated), which keeps
    the estimator transparent for sensitivity analysis.
    """
    S = sample_covariance(returns)
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    n = len(sd)
    rbar = (corr.sum() - n) / (n * (n - 1))
    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, sd ** 2)
    return (1 - delta) * S + delta * target


def ewma_covariance(returns, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics exponentially weighted covariance (zero-mean convention)."""
    X = _arr(returns)
    T = X.shape[0]
    w = lam ** np.arange(T - 1, -1, -1)
    w = w / w.sum()
    Xc = X - X.mean(axis=0)
    return (Xc * w[:, None]).T @ Xc


def nearest_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping eigenvalues."""
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def to_correlation(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def condition_number(cov: np.ndarray) -> float:
    vals = np.linalg.eigvalsh((cov + cov.T) / 2)
    vmin = max(vals.min(), 1e-300)
    return float(vals.max() / vmin)


def shrunk_mean(returns, kappa: float = 0.5) -> np.ndarray:
    """Shrink asset means towards the grand mean (James-Stein flavour)."""
    mu = sample_mean(returns)
    return (1 - kappa) * mu + kappa * mu.mean()


ESTIMATORS = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "oas": oas_covariance,
    "const_corr_0.5": constant_correlation_shrinkage,
    "ewma_0.94": ewma_covariance,
}


def get_estimator(name: str):
    return ESTIMATORS[name]


def to_frame(cov: np.ndarray, columns) -> pd.DataFrame:
    return pd.DataFrame(cov, index=columns, columns=columns)
