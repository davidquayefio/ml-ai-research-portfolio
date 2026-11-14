"""Walk-forward backtesting engine.

Research purpose: Test 6 (out-of-sample robustness). The engine is built so
that look-ahead bias is structurally impossible:

* weights chosen at rebalance index i use ONLY rows [: i] of the returns
  (i.e. information up to the previous close);
* they are applied to rows [i : i + rebalance_every];
* between rebalances weights drift with realised returns;
* transaction costs are charged on the turnover generated at each rebalance,
  taken out of the first return of the new holding period.

Survivorship bias is NOT removed by the engine: it depends on the asset
universe (ETFs that still exist today). See research/limitations.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import covariance_estimators as ce
from .portfolio_models import portfolio_stats
from .evaluation import effective_n

ModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class BacktestResult:
    net_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]          # target weights at rebalance dates
    turnover: pd.DataFrame                    # one-way turnover at each rebalance
    ex_ante: dict[str, pd.DataFrame]          # in-sample ex-ante stats at each rebalance
    failures: dict[str, int] = field(default_factory=dict)

    def mean_turnover(self) -> pd.Series:
        return self.turnover.mean()

    def mean_effective_n(self) -> pd.Series:
        return pd.Series({m: w.apply(lambda r: effective_n(r.values), axis=1).mean() for m, w in self.weights.items()})


def walk_forward(
    returns: pd.DataFrame,
    models: dict[str, ModelFn],
    window: int = 504,
    rebalance_every: int = 21,
    expanding: bool = False,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """Run every model through the same walk-forward schedule.

    window: lookback length in observations (initial length if expanding).
    cost_bps: proportional cost per unit of one-way turnover, in basis points.
    """
    R = returns.to_numpy(dtype=float)
    T, N = R.shape
    cols = list(returns.columns)
    if T <= window + rebalance_every:
        raise ValueError("Not enough observations for the chosen window.")

    names = list(models)
    net = {m: [] for m in names}
    gross = {m: [] for m in names}
    dates_out: list = []
    wts = {m: [] for m in names}
    wdates: list = []
    tov = {m: [] for m in names}
    exa = {m: [] for m in names}
    fails = {m: 0 for m in names}
    held = {m: np.full(N, 1.0 / N) for m in names}  # start from equal weight (cost of first trade charged)
    first = True

    for i in range(window, T, rebalance_every):
        lo = 0 if expanding else i - window
        est = R[lo:i]
        hold = R[i : min(i + rebalance_every, T)]
        wdates.append(returns.index[i])
        for m in names:
            try:
                w_new = np.asarray(models[m](est), dtype=float)
                if not np.all(np.isfinite(w_new)) or abs(w_new.sum() - 1) > 1e-6:
                    raise ValueError("bad weights")
            except Exception:
                fails[m] += 1
                w_new = held[m].copy()
            drifted = held[m]
            to = 0.0 if first else 0.5 * np.abs(w_new - drifted).sum()
            tov[m].append(to)
            wts[m].append(w_new)
            S = ce.sample_covariance(est)
            exa[m].append(portfolio_stats(w_new, est.mean(axis=0), S))

            w = w_new.copy()
            for k, row in enumerate(hold):
                rp = float(w @ row)
                cost = (cost_bps / 1e4) * 2 * to if k == 0 else 0.0  # 2*one-way: sell + buy legs
                gross[m].append(rp)
                net[m].append(rp - cost)
                w = w * (1 + row) / (1 + rp)
            held[m] = w
        dates_out.extend(returns.index[i : min(i + rebalance_every, T)])
        first = False

    net_df = pd.DataFrame(net, index=dates_out)
    gross_df = pd.DataFrame(gross, index=dates_out)
    weights = {m: pd.DataFrame(wts[m], index=wdates, columns=cols) for m in names}
    turnover = pd.DataFrame(tov, index=wdates)
    ex_ante = {m: pd.DataFrame(exa[m], index=wdates) for m in names}
    return BacktestResult(net_df, gross_df, weights, turnover, ex_ante, fails)


def in_sample_vs_out_of_sample(result: BacktestResult, sharpe_fn) -> pd.DataFrame:
    """Average ex-ante (in-sample) Sharpe versus realised out-of-sample Sharpe.

    The gap is a direct measure of optimisation 'overfitting': how much of
    the promised performance failed to materialise.
    """
    rows = {}
    for m in result.net_returns.columns:
        ins = result.ex_ante[m]["sharpe"].mean()
        oos = sharpe_fn(result.gross_returns[m])
        rows[m] = {"in_sample_sharpe": ins, "oos_sharpe_gross": oos, "gap": ins - oos}
    return pd.DataFrame(rows).T


def make_models(spec: dict[str, Callable], **fixed_kwargs) -> dict[str, ModelFn]:
    """Bind keyword arguments so every model has the signature f(window)->weights."""
    return {name: (lambda w, f=f: f(w, **fixed_kwargs)) for name, f in spec.items()}
