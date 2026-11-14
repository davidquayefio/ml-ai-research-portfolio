"""The ten portfolio construction models.

Every model exposes the same interface:

    weights = model(returns_window: np.ndarray, **kwargs) -> np.ndarray (N,)

so the backtester can swap them freely. Research purpose: isolate the
effect of the *construction rule* and of the *input estimators* on weight
stability and out-of-sample behaviour.

Conventions
-----------
* Returns are periodic simple returns (daily by default).
* Weights sum to 1. Long-only (w >= 0) unless stated, because unconstrained
  mean-variance weights are famously extreme and that is itself one of the
  phenomena under test (see `gmv_unconstrained`).
* Optimisation uses SciPy SLSQP (convex problems, small N). The same
  problems can be expressed in cvxpy; SciPy keeps the code dependency-light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from . import covariance_estimators as ce

TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _n(returns: np.ndarray) -> int:
    return np.asarray(returns).shape[1]


def _solve(
    objective: Callable[[np.ndarray], float],
    n: int,
    bounds: tuple[float, float] = (0.0, 1.0),
    x0: np.ndarray | None = None,
    extra_constraints: list | None = None,
    jac: Callable | None = None,
) -> np.ndarray:
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]
    if extra_constraints:
        cons += extra_constraints
    res = minimize(
        objective, x0, jac=jac, method="SLSQP",
        bounds=[bounds] * n, constraints=cons,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, bounds[0], bounds[1])
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


# ----------------------------------------------------------------------------
# 1. Equal weight
# ----------------------------------------------------------------------------
def equal_weight(returns, **_) -> np.ndarray:
    n = _n(returns)
    return np.full(n, 1.0 / n)


# ----------------------------------------------------------------------------
# 2. Global minimum variance (unconstrained, closed form) -- shorts allowed
#    Derivation: min w'Σw s.t. 1'w = 1  =>  w* = Σ^{-1}1 / (1'Σ^{-1}1)
# ----------------------------------------------------------------------------
def gmv_unconstrained(returns, cov_estimator=ce.sample_covariance, **_) -> np.ndarray:
    S = cov_estimator(returns)
    ones = np.ones(S.shape[0])
    x = np.linalg.solve(S, ones)
    return x / x.sum()


def gmv_long_only(returns, cov_estimator=ce.sample_covariance, **_) -> np.ndarray:
    S = cov_estimator(returns)
    n = S.shape[0]
    return _solve(lambda w: w @ S @ w, n, jac=lambda w: 2 * S @ w)


# ----------------------------------------------------------------------------
# 3. Mean-variance with a target return
# ----------------------------------------------------------------------------
def mean_variance_target(
    returns, target: float | None = None, cov_estimator=ce.sample_covariance,
    mean_estimator=ce.sample_mean, **_
) -> np.ndarray:
    """min w'Σw s.t. w'μ = target, 1'w = 1, w >= 0.

    If target is None, use the midpoint between the minimum-variance
    portfolio's expected return and the best single asset's mean (always
    feasible under long-only).
    """
    S = cov_estimator(returns)
    mu = mean_estimator(returns)
    n = len(mu)
    if target is None:
        w_g = gmv_long_only(returns, cov_estimator=cov_estimator)
        target = 0.5 * (w_g @ mu + mu.max())
    cons = [{"type": "eq", "fun": lambda w: w @ mu - target}]
    return _solve(lambda w: w @ S @ w, n, extra_constraints=cons, jac=lambda w: 2 * S @ w)


# ----------------------------------------------------------------------------
# 4. Maximum Sharpe (tangency) portfolio, long-only
# ----------------------------------------------------------------------------
def max_sharpe(
    returns, rf: float = 0.0, cov_estimator=ce.sample_covariance,
    mean_estimator=ce.sample_mean, **_
) -> np.ndarray:
    S = cov_estimator(returns)
    mu = mean_estimator(returns) - rf / TRADING_DAYS
    n = len(mu)
    if mu.max() <= 0:  # no positive excess mean: fall back to min variance
        return gmv_long_only(returns, cov_estimator=cov_estimator)

    def neg_sharpe(w):
        v = max(w @ S @ w, 1e-16)
        return -(w @ mu) / np.sqrt(v)

    best = None
    starts = [np.full(n, 1.0 / n)] + [np.eye(n)[i] * 0.6 + 0.4 / n for i in range(min(n, 4))]
    for x0 in starts:
        w = _solve(neg_sharpe, n, x0=x0 / x0.sum())
        val = neg_sharpe(w)
        if best is None or val < best[0]:
            best = (val, w)
    return best[1]


# ----------------------------------------------------------------------------
# 5. Minimum volatility: long-only with a concentration cap
# ----------------------------------------------------------------------------
def min_volatility_capped(
    returns, cap: float = 0.25, cov_estimator=ce.sample_covariance, **_
) -> np.ndarray:
    S = cov_estimator(returns)
    n = S.shape[0]
    cap = max(cap, 1.0 / n + 1e-9)
    return _solve(lambda w: w @ S @ w, n, bounds=(0.0, cap), jac=lambda w: 2 * S @ w)


# ----------------------------------------------------------------------------
# 6. Ledoit-Wolf shrinkage minimum-variance portfolio (long-only)
# ----------------------------------------------------------------------------
def ledoit_wolf_min_variance(returns, **_) -> np.ndarray:
    return gmv_long_only(returns, cov_estimator=ce.ledoit_wolf_covariance)


# ----------------------------------------------------------------------------
# 7. Robust mean-variance (ellipsoidal uncertainty set on the mean)
#    max_w  w'μ - κ sqrt(w'Ω w) - (γ/2) w'Σ w,   Ω = Σ / T
#    The worst-case mean over {m : (m-μ)'Ω^{-1}(m-μ) <= κ²} is w'μ - κ·sqrt(w'Ωw).
# ----------------------------------------------------------------------------
def robust_mean_variance(
    returns, kappa: float = 1.0, gamma: float = 5.0,
    cov_estimator=ce.ledoit_wolf_covariance, **_
) -> np.ndarray:
    X = np.asarray(returns, dtype=float)
    T, n = X.shape
    S = cov_estimator(X)
    mu = ce.sample_mean(X)
    Om = S / T
    scale = TRADING_DAYS

    def obj(w):
        pen = np.sqrt(max(w @ Om @ w, 1e-20))
        return -(scale * (w @ mu) - scale * kappa * pen) + 0.5 * gamma * scale * (w @ S @ w)

    return _solve(obj, n)


# ----------------------------------------------------------------------------
# 8. Risk parity (equal risk contribution)
#    RC_i = w_i (Σw)_i / sqrt(w'Σw);  target RC_i = RC_j for all i, j.
# ----------------------------------------------------------------------------
def risk_parity(returns, cov_estimator=ce.sample_covariance, **_) -> np.ndarray:
    S = cov_estimator(returns)
    n = S.shape[0]

    def obj(w):
        w = np.maximum(w, 1e-12)
        rc = w * (S @ w)
        rc = rc / rc.sum()
        return np.sum((rc - 1.0 / n) ** 2) * 1e4

    return _solve(obj, n, bounds=(1e-6, 1.0))


def risk_contributions(w: np.ndarray, S: np.ndarray) -> np.ndarray:
    rc = w * (S @ w)
    return rc / rc.sum()


# ----------------------------------------------------------------------------
# 9. Black-Litterman
#    Prior (equilibrium) returns: π = δ Σ w_mkt.  With views (P, Q, Ω):
#    μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]
#    NOTE: true market-cap weights are not downloaded here; the default prior
#    uses equal weights as a transparent proxy. This is a stated limitation.
# ----------------------------------------------------------------------------
def black_litterman_mu(
    S: np.ndarray, w_mkt: np.ndarray, delta: float = 2.5, tau: float = 0.05,
    P: np.ndarray | None = None, Q: np.ndarray | None = None,
    omega: np.ndarray | None = None,
) -> np.ndarray:
    pi = delta * S @ w_mkt
    if P is None or Q is None:
        return pi
    if omega is None:
        omega = np.diag(np.diag(P @ (tau * S) @ P.T))
    ts_inv = np.linalg.inv(tau * S)
    om_inv = np.linalg.inv(omega)
    A = ts_inv + P.T @ om_inv @ P
    b = ts_inv @ pi + P.T @ om_inv @ Q
    return np.linalg.solve(A, b)


def black_litterman(
    returns, w_mkt: np.ndarray | None = None, delta: float = 2.5, tau: float = 0.05,
    P=None, Q=None, omega=None, cov_estimator=ce.ledoit_wolf_covariance,
    data_driven_views: bool = True, view_confidence: float = 1.0, **_
) -> np.ndarray:
    """Black-Litterman portfolio (long-only mean-variance on the posterior mean).

    With NO views, reverse-optimisation returns the prior weights exactly
    (here: equal weight), so the model would be a duplicate of 1/N. To make it a
    distinct, testable model, `data_driven_views=True` (default) uses absolute
    views on every asset equal to the window's annualised sample mean, with view
    uncertainty omega = diag(tau*Sigma)/view_confidence. This makes the posterior
    mean a precision-weighted blend of the equilibrium prior and the sample
    mean, i.e. a Bayesian shrinkage of the means. This is a research design
    choice (documented in research/methodology.md), not the only way to specify views.
    """
    S = cov_estimator(returns) * TRADING_DAYS  # annualised for delta scaling
    n = S.shape[0]
    w_mkt = np.full(n, 1.0 / n) if w_mkt is None else np.asarray(w_mkt)
    if P is None and Q is None and data_driven_views:
        P = np.eye(n)
        Q = ce.sample_mean(returns) * TRADING_DAYS
        omega = np.diag(np.diag(tau * S)) / view_confidence
    mu_bl = black_litterman_mu(S, w_mkt, delta, tau, P, Q, omega)
    return _solve(lambda w: -(w @ mu_bl) + 0.5 * delta * (w @ S @ w), n)


# ----------------------------------------------------------------------------
# 10. Conditional-volatility portfolio: minimum variance using an EWMA
#     (conditional) covariance rather than the unconditional sample one.
# ----------------------------------------------------------------------------
def conditional_vol_min_variance(returns, lam: float = 0.94, **_) -> np.ndarray:
    return gmv_long_only(returns, cov_estimator=lambda r: ce.ewma_covariance(r, lam))


# ----------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    name: str
    fn: Callable
    needs_mean: bool
    description: str

    def __call__(self, returns, **kw):
        return self.fn(returns, **kw)


MODELS: dict[str, Model] = {
    "equal_weight": Model("equal_weight", equal_weight, False, "1/N"),
    "gmv_unconstrained": Model("gmv_unconstrained", gmv_unconstrained, False, "Closed-form GMV, shorts allowed"),
    "gmv_long_only": Model("gmv_long_only", gmv_long_only, False, "GMV, w>=0"),
    "mean_variance": Model("mean_variance", mean_variance_target, True, "Target-return MV, w>=0"),
    "max_sharpe": Model("max_sharpe", max_sharpe, True, "Tangency, w>=0"),
    "min_vol_capped": Model("min_vol_capped", min_volatility_capped, False, "Min vol, 0<=w<=25%"),
    "ledoit_wolf_minvar": Model("ledoit_wolf_minvar", ledoit_wolf_min_variance, False, "Min var with LW covariance"),
    "robust_mv": Model("robust_mv", robust_mean_variance, True, "Ellipsoidal robust MV"),
    "risk_parity": Model("risk_parity", risk_parity, False, "Equal risk contribution"),
    "black_litterman": Model("black_litterman", black_litterman, True, "BL, equal-weight prior + sample-mean views"),
    "cond_vol_minvar": Model("cond_vol_minvar", conditional_vol_min_variance, False, "Min var with EWMA covariance"),
}

# The ten models named in the proposal (long-only variant of GMV is used in
# backtests by default; the unconstrained version is analysed separately).
CORE_TEN = [
    "equal_weight", "gmv_long_only", "mean_variance", "max_sharpe",
    "min_vol_capped", "ledoit_wolf_minvar", "robust_mv", "risk_parity",
    "black_litterman", "cond_vol_minvar",
]


def portfolio_stats(w: np.ndarray, mu: np.ndarray, S: np.ndarray) -> dict:
    """Ex-ante annualised return, volatility and Sharpe (rf=0)."""
    r = float(w @ mu) * TRADING_DAYS
    v = float(np.sqrt(w @ S @ w * TRADING_DAYS))
    return {"exp_return": r, "volatility": v, "sharpe": r / v if v > 0 else np.nan}
