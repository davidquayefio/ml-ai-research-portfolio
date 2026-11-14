"""Performance evaluation and statistical comparison.

Research purpose: performance metrics are reported but are NOT the definition
of validity (see research/methodology.md). Statistical comparisons use a
stationary block bootstrap because returns are serially dependent and
heavy-tailed, so iid standard errors are unreliable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualised_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (periods / len(r)) - 1)


def annualised_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(r: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = r - rf / periods
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def sortino_ratio(r: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    ex = r - rf / periods
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(ex.mean() / dd * np.sqrt(periods)) if dd > 0 else np.nan


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


def calmar_ratio(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(r))
    return annualised_return(r, periods) / mdd if mdd > 0 else np.nan


def historical_var(r: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR as a positive loss number."""
    return float(-np.quantile(r.dropna(), alpha))


def historical_cvar(r: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall: mean loss beyond the VaR threshold (positive)."""
    x = r.dropna()
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    return float(-tail.mean())


def effective_n(w: np.ndarray) -> float:
    """Inverse Herfindahl: the number of equally weighted assets with the same concentration."""
    w = np.asarray(w)
    return float(1.0 / np.sum(w ** 2))


def performance_table(
    returns: pd.DataFrame,
    turnover: pd.Series | None = None,
    avg_eff_n: pd.Series | None = None,
    rf: float = 0.0,
) -> pd.DataFrame:
    rows = {}
    for c in returns.columns:
        r = returns[c].dropna()
        rows[c] = {
            "ann_return": annualised_return(r),
            "ann_vol": annualised_vol(r),
            "sharpe": sharpe_ratio(r, rf),
            "sortino": sortino_ratio(r, rf),
            "max_drawdown": max_drawdown(r),
            "calmar": calmar_ratio(r),
            "VaR_5%": historical_var(r, 0.05),
            "CVaR_5%": historical_cvar(r, 0.05),
        }
        if turnover is not None and c in turnover.index:
            rows[c]["avg_turnover_per_rebalance"] = turnover[c]
        if avg_eff_n is not None and c in avg_eff_n.index:
            rows[c]["avg_effective_N"] = avg_eff_n[c]
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# Stationary bootstrap (Politis & Romano, 1994)
# ----------------------------------------------------------------------------
def stationary_bootstrap_indices(
    T: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    p = 1.0 / mean_block
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(T)
    new = rng.random(T) < p
    starts = rng.integers(T, size=T)
    for t in range(1, T):
        idx[t] = starts[t] if new[t] else (idx[t - 1] + 1) % T
    return idx


def bootstrap_metric_ci(
    r: pd.Series, metric=sharpe_ratio, n_boot: int = 2000, mean_block: float = 10,
    alpha: float = 0.05, seed: int = 0,
) -> dict:
    x = r.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        ix = stationary_bootstrap_indices(len(x), mean_block, rng)
        stats[b] = metric(pd.Series(x[ix]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"estimate": metric(pd.Series(x)), "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_sharpe_difference(
    r1: pd.Series, r2: pd.Series, n_boot: int = 2000, mean_block: float = 10,
    alpha: float = 0.05, seed: int = 0,
) -> dict:
    """CI and two-sided bootstrap p-value for SR(r1) - SR(r2).

    The p-value recentres the bootstrap distribution at zero (null of equal
    Sharpe ratios). Dependence is preserved by resampling paired blocks.
    This is in the spirit of Ledoit & Wolf (2008) but is a simplified
    implementation; treat p-values as indicative, not definitive.
    """
    df = pd.concat([r1, r2], axis=1).dropna()
    x = df.to_numpy()
    rng = np.random.default_rng(seed)
    obs = sharpe_ratio(df.iloc[:, 0]) - sharpe_ratio(df.iloc[:, 1])
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        ix = stationary_bootstrap_indices(len(x), mean_block, rng)
        s1 = sharpe_ratio(pd.Series(x[ix, 0]))
        s2 = sharpe_ratio(pd.Series(x[ix, 1]))
        diffs[b] = s1 - s2
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(obs)))
    return {"diff": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": p}


def economic_significance(
    r_candidate: pd.Series, r_benchmark: pd.Series, gamma: float = 5.0,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised certainty-equivalent return gain (mean-variance utility).

    CE = mean - (gamma/2) var, annualised. Positive means the candidate is
    economically preferred. Statistical significance is judged separately.
    """
    def ce(r):
        return periods * r.mean() - 0.5 * gamma * periods * r.var(ddof=1)

    return float(ce(r_candidate.dropna()) - ce(r_benchmark.dropna()))


def regime_conditional_performance(
    returns: pd.DataFrame, regimes: pd.Series, min_obs: int = 30
) -> pd.DataFrame:
    """Sharpe, vol and mean return by regime label for every model."""
    out = []
    reg = regimes.reindex(returns.index)
    for label in sorted(reg.dropna().unique()):
        sub = returns[reg == label]
        if len(sub) < min_obs:
            continue
        for c in sub.columns:
            out.append(
                {
                    "regime": label, "model": c, "n_obs": len(sub),
                    "ann_return": float(sub[c].mean() * TRADING_DAYS),
                    "ann_vol": annualised_vol(sub[c]),
                    "sharpe": sharpe_ratio(sub[c]),
                }
            )
    return pd.DataFrame(out)
