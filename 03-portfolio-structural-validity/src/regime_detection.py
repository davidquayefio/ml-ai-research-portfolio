"""Market regime labelling.

Research purpose: Test 4 asks whether model behaviour differs across
regimes. Two kinds of labels are provided and must not be confused:

* ex-post labels use the whole sample (e.g. full-sample volatility
  quantiles). They are fine for *describing* what happened, but a trader
  could not have used them -> never use them to drive portfolio decisions.
* ex-ante labels use only information available at each date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate historical stress windows. These are conventional, widely
# discussed episodes; exact start/end dates are a judgement call and should
# be verified/adjusted by the researcher (documented in research/limitations.md).
STRESS_WINDOWS = {
    "GFC": ("2008-09-01", "2009-03-31"),
    "Euro_crisis_2011": ("2011-07-15", "2011-10-31"),
    "Q4_2018_selloff": ("2018-10-01", "2018-12-31"),
    "COVID_crash": ("2020-02-15", "2020-04-30"),
    "Rate_shock_2022": ("2022-01-01", "2022-10-31"),
}


def market_proxy(returns: pd.DataFrame, column: str | None = None) -> pd.Series:
    """Equal-weight average of assets, or a named column (e.g. 'SPY')."""
    return returns[column] if column else returns.mean(axis=1)


def volatility_regimes_ex_post(
    market: pd.Series, window: int = 21, q_low: float = 0.33, q_high: float = 0.67
) -> pd.Series:
    """Label low / mid / high volatility using FULL-sample quantiles (ex-post)."""
    vol = market.rolling(window).std() * np.sqrt(252)
    lo, hi = vol.quantile([q_low, q_high])
    lab = pd.Series(np.where(vol <= lo, "low_vol", np.where(vol >= hi, "high_vol", "mid_vol")), index=market.index)
    lab[vol.isna()] = np.nan
    return lab


def volatility_regimes_ex_ante(
    market: pd.Series, window: int = 21, min_history: int = 252,
    q_low: float = 0.33, q_high: float = 0.67,
) -> pd.Series:
    """Same labels but thresholds use only data up to each date (expanding quantiles)."""
    vol = market.rolling(window).std() * np.sqrt(252)
    lo = vol.expanding(min_periods=min_history).quantile(q_low)
    hi = vol.expanding(min_periods=min_history).quantile(q_high)
    lab = pd.Series(np.where(vol <= lo, "low_vol", np.where(vol >= hi, "high_vol", "mid_vol")), index=market.index)
    lab[lo.isna() | vol.isna()] = np.nan
    return lab


def bull_bear_regimes(market: pd.Series, bear_threshold: float = -0.20) -> pd.Series:
    """Classic rule: a bear market begins when the index falls `bear_threshold`
    from its peak and ends when it recovers `+20%` from the trough (ex-ante:
    uses only past values at each date)."""
    wealth = (1 + market).cumprod()
    state = "bull"
    peak = wealth.iloc[0]
    trough = wealth.iloc[0]
    labels = []
    for w in wealth:
        if state == "bull":
            peak = max(peak, w)
            if w / peak - 1 <= bear_threshold:
                state = "bear"
                trough = w
        else:
            trough = min(trough, w)
            if w / trough - 1 >= 0.20:
                state = "bull"
                peak = w
        labels.append(state)
    return pd.Series(labels, index=market.index)


def stress_labels(index: pd.DatetimeIndex, windows: dict | None = None) -> pd.Series:
    """'stress' inside any listed window, otherwise 'normal'."""
    windows = windows or STRESS_WINDOWS
    lab = pd.Series("normal", index=index)
    for start, end in windows.values():
        lab[(index >= start) & (index <= end)] = "stress"
    return lab


def gmm_regimes(
    market: pd.Series, n_regimes: int = 2, window: int = 21, seed: int = 0
) -> pd.Series:
    """Gaussian-mixture regimes on (rolling mean, rolling vol). Ex-post fit.

    Regimes are re-labelled so 0 = lowest-volatility cluster. Unsupervised
    clusters need not correspond to economic regimes; interpret with care.
    """
    from sklearn.mixture import GaussianMixture

    feats = pd.DataFrame(
        {"mu": market.rolling(window).mean() * 252, "vol": market.rolling(window).std() * np.sqrt(252)}
    ).dropna()
    gm = GaussianMixture(n_regimes, covariance_type="full", random_state=seed, n_init=5).fit(feats)
    raw = gm.predict(feats)
    order = np.argsort(gm.means_[:, 1])
    remap = {old: new for new, old in enumerate(order)}
    lab = pd.Series([f"gmm_{remap[x]}" for x in raw], index=feats.index)
    return lab.reindex(market.index)


def markov_switching_regimes(market: pd.Series, k: int = 2):
    """Markov-switching mean/variance model (requires statsmodels).

    Returns (smoothed probability DataFrame, fitted results). The
    low-variance regime is placed first.
    """
    import statsmodels.api as sm

    mod = sm.tsa.MarkovRegression(market.dropna() * 100, k_regimes=k, switching_variance=True)
    res = mod.fit(disp=False)
    probs = res.smoothed_marginal_probabilities
    variances = [res.params[f"sigma2[{i}]"] for i in range(k)]
    probs = probs[np.argsort(variances)]
    probs.columns = [f"regime_{i}" for i in range(k)]
    return probs, res
