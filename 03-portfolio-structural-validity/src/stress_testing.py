"""Controlled perturbation and stress tests (Test 7).

Research purpose: separate the model's *mathematical* response to input
changes (a property of the optimiser) from its *statistical* response to
sampling noise (a property of estimation). Perturbations are applied to the
estimated inputs, so differences in weights are caused by the perturbation
only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import covariance_estimators as ce
from .portfolio_models import TRADING_DAYS


# ----------------------------------------------------------------------------
# Perturbation operators on returns (so every model's own estimator is used)
# ----------------------------------------------------------------------------
def perturb_mean(returns: np.ndarray, sigma_mult: float, rng) -> np.ndarray:
    """Add a constant shift to each asset's mean of size N(0, (sigma_mult * SE)^2),
    where SE = sd/sqrt(T) is the standard error of the sample mean.
    sigma_mult=1 means 'one standard error of estimation noise'."""
    X = np.asarray(returns, dtype=float)
    T = X.shape[0]
    se = X.std(axis=0, ddof=1) / np.sqrt(T)
    return X + rng.normal(0, 1, X.shape[1]) * se * sigma_mult


def scale_volatility(returns: np.ndarray, factor: float, assets: list[int] | None = None) -> np.ndarray:
    """Scale deviations from the mean (inflates/deflates volatility)."""
    X = np.asarray(returns, dtype=float)
    mu = X.mean(axis=0)
    out = X.copy()
    idx = assets if assets is not None else list(range(X.shape[1]))
    out[:, idx] = mu[idx] + (X[:, idx] - mu[idx]) * factor
    return out


def shift_correlation(returns: np.ndarray, rho_shift: float) -> np.ndarray:
    """Impose a common-factor shock that raises average correlation.

    Adds a shared standardised factor f (the equal-weight cross-sectional
    mean, rescaled) with loading ~ rho_shift * sd_i.  Volatilities change
    slightly; correlations rise towards 1 as rho_shift grows.
    """
    X = np.asarray(returns, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    f = (X - mu).mean(axis=1)
    f = f / f.std(ddof=1)
    return X + rho_shift * sd[None, :] * f[:, None]


def inject_outliers(
    returns: np.ndarray, n_outliers: int, magnitude_sd: float, rng, common: bool = True
) -> np.ndarray:
    """Replace n random days with extreme joint (or idiosyncratic) shocks."""
    X = np.asarray(returns, dtype=float).copy()
    T, N = X.shape
    sd = X.std(axis=0, ddof=1)
    days = rng.choice(T, size=n_outliers, replace=False)
    for d in days:
        sign = rng.choice([-1, 1])
        if common:
            X[d] = X.mean(axis=0) + sign * magnitude_sd * sd
        else:
            j = rng.integers(N)
            X[d, j] = X[:, j].mean() + sign * magnitude_sd * sd[j]
    return X


def perturb_covariance_eigen(returns: np.ndarray, noise_level: float, rng) -> np.ndarray:
    """Return a covariance matrix whose eigenvalues are multiplicatively
    perturbed by exp(N(0, noise_level^2)) and re-projected to PSD."""
    S = ce.sample_covariance(returns)
    vals, vecs = np.linalg.eigh(S)
    vals = vals * np.exp(rng.normal(0, noise_level, len(vals)))
    return (vecs * vals) @ vecs.T


# ----------------------------------------------------------------------------
# Experiment runners
# ----------------------------------------------------------------------------
def weight_response(
    model, returns: np.ndarray, perturb, levels, n_rep: int = 50, seed: int = 0
) -> pd.DataFrame:
    """For each perturbation level, repeat n_rep times and summarise how far
    the weights move from the unperturbed ones.

    perturb(returns, level, rng) -> perturbed returns.
    """
    X = np.asarray(returns, dtype=float)
    w0 = model(X)
    rng = np.random.default_rng(seed)
    rows = []
    for lev in levels:
        for _ in range(n_rep):
            try:
                w = model(perturb(X, lev, rng))
            except Exception:
                continue
            S = ce.sample_covariance(X)
            rows.append(
                {
                    "level": lev,
                    "L1_distance": float(np.abs(w - w0).sum()),
                    "turnover_equiv": float(0.5 * np.abs(w - w0).sum()),
                    "vol_change": float(np.sqrt(w @ S @ w) / np.sqrt(w0 @ S @ w0) - 1),
                    "max_weight": float(w.max()),
                }
            )
    return pd.DataFrame(rows)


def compare_models_under_perturbation(
    models: dict, returns: np.ndarray, perturb, level: float, n_rep: int = 50, seed: int = 0
) -> pd.DataFrame:
    """One-level comparison across models: mean turnover-equivalent & variance
    inflation (risk of the perturbed portfolio evaluated under the ORIGINAL covariance)."""
    X = np.asarray(returns, dtype=float)
    S = ce.sample_covariance(X)
    out = {}
    for name, m in models.items():
        res = weight_response(m, X, perturb, [level], n_rep, seed)
        out[name] = {
            "mean_turnover_equiv": res["turnover_equiv"].mean(),
            "p95_turnover_equiv": res["turnover_equiv"].quantile(0.95),
            "mean_vol_change": res["vol_change"].mean(),
            "n_ok": len(res),
        }
    return pd.DataFrame(out).T


def historical_scenario_returns(
    weights: pd.Series, returns: pd.DataFrame, start: str, end: str
) -> dict:
    """Cumulative return and worst day of a fixed-weight portfolio inside a window."""
    sub = returns.loc[start:end]
    rp = sub @ weights
    return {
        "cum_return": float((1 + rp).prod() - 1),
        "worst_day": float(rp.min()),
        "n_obs": int(len(rp)),
    }


def estimation_error_risk_gap(
    w: np.ndarray, S_estimated: np.ndarray, S_realised: np.ndarray
) -> dict:
    """Predicted vs realised portfolio volatility (annualised).
    ratio > 1 means the optimiser under-estimated its own risk."""
    pred = float(np.sqrt(w @ S_estimated @ w * TRADING_DAYS))
    real = float(np.sqrt(w @ S_realised @ w * TRADING_DAYS))
    return {"predicted_vol": pred, "realised_vol": real, "ratio": real / pred if pred > 0 else np.nan}
