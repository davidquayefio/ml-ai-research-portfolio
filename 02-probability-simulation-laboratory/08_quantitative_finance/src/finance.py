"""Module 8: probability in quantitative finance.

IMPORTANT: everything here is a *model*. Geometric Brownian motion and Black-Scholes
assume constant volatility, continuous trading, no jumps, no transaction costs and
Gaussian log-returns. Real markets violate all of these; see the research notes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from plab.processes import gbm_terminal


class PriceEstimate(NamedTuple):
    price: float
    std_error: float


# ----------------------------------------------------------------------------
# Returns
# ----------------------------------------------------------------------------
def simple_returns(prices: np.ndarray) -> np.ndarray:
    """``R_t = S_t / S_{t-1} - 1``."""
    prices = np.asarray(prices, dtype=float)
    return prices[..., 1:] / prices[..., :-1] - 1.0


def log_returns(prices: np.ndarray) -> np.ndarray:
    """``r_t = ln(S_t / S_{t-1})``."""
    return np.diff(np.log(np.asarray(prices, dtype=float)), axis=-1)


def simple_from_log(r: np.ndarray) -> np.ndarray:
    return np.expm1(r)


def log_from_simple(R: np.ndarray) -> np.ndarray:
    return np.log1p(R)


# ----------------------------------------------------------------------------
# Black-Scholes closed forms
# ----------------------------------------------------------------------------
def _d1_d2(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call(S0, K, r, sigma, T):
    """European call: ``S0 Phi(d1) - K e^{-rT} Phi(d2)``."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def bs_put(S0, K, r, sigma, T):
    """European put: ``K e^{-rT} Phi(-d2) - S0 Phi(-d1)``."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)


def bs_call_delta(S0, K, r, sigma, T):
    return stats.norm.cdf(_d1_d2(S0, K, r, sigma, T)[0])


def bs_vega(S0, K, r, sigma, T):
    """``dC/dsigma = S0 phi(d1) sqrt(T)``."""
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return S0 * stats.norm.pdf(d1) * np.sqrt(T)


def implied_vol_call(price: float, S0: float, K: float, r: float, T: float) -> float:
    """Invert the Black-Scholes call formula for sigma (monotone in sigma, so root finding is safe)."""
    return float(brentq(lambda s: bs_call(S0, K, r, s, T) - price, 1e-6, 5.0))


def geometric_asian_call(S0, K, r, sigma, T, m):
    """Closed-form price of a discretely monitored geometric-average Asian call.

    Monitoring dates ``t_i = i T / m``, ``i = 1..m``. ``ln G`` is normal with
    ``M = ln S0 + (r - sigma^2/2) T (m+1)/(2m)`` and ``V = sigma^2 T (m+1)(2m+1)/(6 m^2)``;
    price ``= e^{-rT} [ e^{M + V/2} Phi(d1) - K Phi(d2) ]``.
    """
    M = np.log(S0) + (r - 0.5 * sigma**2) * T * (m + 1) / (2.0 * m)
    V = sigma**2 * T * (m + 1) * (2 * m + 1) / (6.0 * m**2)
    sV = np.sqrt(V)
    d2 = (M - np.log(K)) / sV
    d1 = d2 + sV
    return np.exp(-r * T) * (np.exp(M + 0.5 * V) * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))


# ----------------------------------------------------------------------------
# Monte Carlo pricing (risk-neutral: drift = r)
# ----------------------------------------------------------------------------
def mc_european_call(S0, K, r, sigma, T, n: int, rng: np.random.Generator, method: str = "plain") -> PriceEstimate:
    """Monte Carlo price of a European call.

    Under the risk-neutral measure the drift is ``r`` and
    ``price = e^{-rT} E[(S_T - K)^+]``.

    method
    ------
    ``plain``       i.i.d. normals.
    ``antithetic``  pairs ``(Z, -Z)``; ``n`` payoffs evaluated in total.
    ``control``     control variate ``e^{-rT} S_T`` whose expectation is exactly ``S0``.
    """
    disc = np.exp(-r * T)
    if method == "plain":
        z = rng.standard_normal(n)
        y = disc * np.maximum(gbm_terminal(S0, r, sigma, T, z) - K, 0.0)
    elif method == "antithetic":
        z = rng.standard_normal(n // 2)
        y = 0.5 * disc * (
            np.maximum(gbm_terminal(S0, r, sigma, T, z) - K, 0.0)
            + np.maximum(gbm_terminal(S0, r, sigma, T, -z) - K, 0.0)
        )
    elif method == "control":
        z = rng.standard_normal(n)
        st = gbm_terminal(S0, r, sigma, T, z)
        payoff = disc * np.maximum(st - K, 0.0)
        c = disc * st
        beta = np.cov(payoff, c)[0, 1] / np.var(c, ddof=1)
        y = payoff - beta * (c - S0)
    else:
        raise ValueError(f"unknown method {method!r}")
    return PriceEstimate(float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size)))


def mc_european_put_call_pair(S0, K, r, sigma, T, n: int, rng: np.random.Generator) -> tuple[PriceEstimate, PriceEstimate]:
    """Call and put priced from the *same* paths (to check put-call parity)."""
    disc = np.exp(-r * T)
    st = gbm_terminal(S0, r, sigma, T, rng.standard_normal(n))
    call = disc * np.maximum(st - K, 0.0)
    put = disc * np.maximum(K - st, 0.0)
    f = lambda y: PriceEstimate(float(y.mean()), float(y.std(ddof=1) / np.sqrt(n)))
    return f(call), f(put)


def mc_asian_call(
    S0, K, r, sigma, T, n_steps: int, n_paths: int, rng: np.random.Generator, control: bool = False
) -> PriceEstimate:
    """Arithmetic-average Asian call by Monte Carlo (no closed form exists).

    With ``control=True`` the geometric Asian call (known in closed form and extremely
    correlated with the arithmetic one) is used as a control variate.
    """
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_s = np.log(S0) + np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    disc = np.exp(-r * T)
    arith = disc * np.maximum(np.exp(log_s).mean(axis=1) - K, 0.0)
    if control:
        geo = disc * np.maximum(np.exp(log_s.mean(axis=1)) - K, 0.0)
        geo_exact = geometric_asian_call(S0, K, r, sigma, T, n_steps)
        beta = np.cov(arith, geo)[0, 1] / np.var(geo, ddof=1)
        y = arith - beta * (geo - geo_exact)
    else:
        y = arith
    return PriceEstimate(float(y.mean()), float(y.std(ddof=1) / np.sqrt(n_paths)))


# ----------------------------------------------------------------------------
# Risk measures (losses are positive numbers)
# ----------------------------------------------------------------------------
def historical_var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Empirical VaR (the alpha-quantile of losses) and ES (mean loss at or beyond VaR)."""
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, alpha))
    es = float(losses[losses >= var].mean())
    return var, es


def normal_var_es(mu: float, sigma: float, alpha: float = 0.99) -> tuple[float, float]:
    """VaR and ES for ``L ~ N(mu, sigma^2)``: ``mu + sigma z_a`` and ``mu + sigma phi(z_a)/(1-a)``."""
    z = stats.norm.ppf(alpha)
    return mu + sigma * z, mu + sigma * stats.norm.pdf(z) / (1.0 - alpha)


def student_t_var_es(mu: float, sigma: float, nu: float, alpha: float = 0.99) -> tuple[float, float]:
    """VaR and ES for a location-scale Student t with mean ``mu`` and standard deviation ``sigma`` (``nu > 2``).

    ``L = mu + s T_nu`` with ``s = sigma sqrt((nu-2)/nu)``; ``ES_T = f_nu(q) (nu + q^2) / ((nu-1)(1-alpha))``.
    """
    s = sigma * np.sqrt((nu - 2.0) / nu)
    q = stats.t.ppf(alpha, nu)
    es_t = stats.t.pdf(q, nu) * (nu + q**2) / ((nu - 1.0) * (1.0 - alpha))
    return mu + s * q, mu + s * es_t


def exceedance_rate(losses: np.ndarray, var: float) -> float:
    """Fraction of losses strictly above a VaR level (should equal ``1 - alpha`` if the model is right)."""
    return float(np.mean(np.asarray(losses) > var))


# ----------------------------------------------------------------------------
# Stylised facts: fat tails and volatility clustering
# ----------------------------------------------------------------------------
def garch11(n: int, omega: float, alpha: float, beta: float, rng: np.random.Generator, burn: int = 1000) -> np.ndarray:
    """Simulate GARCH(1,1) returns with Gaussian innovations.

    ``r_t = sigma_t z_t``, ``sigma_{t+1}^2 = omega + alpha r_t^2 + beta sigma_t^2``.
    Unconditional variance is ``omega / (1 - alpha - beta)``.
    """
    z = rng.standard_normal(n + burn)
    r = np.empty(n + burn)
    s2 = omega / (1.0 - alpha - beta)
    for t in range(n + burn):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r[burn:]


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations at lags ``1..nlags``."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    denom = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, nlags + 1)])


def normal_tail_probability(k: float) -> float:
    """``P(|X - mu| > k sigma)`` for a normal."""
    return float(2.0 * stats.norm.sf(k))


def student_t_tail_probability(k: float, nu: float) -> float:
    """``P(|X - mu| > k sigma)`` for a Student t scaled to unit variance (``nu > 2``)."""
    return float(2.0 * stats.t.sf(k * np.sqrt(nu / (nu - 2.0)), nu))
