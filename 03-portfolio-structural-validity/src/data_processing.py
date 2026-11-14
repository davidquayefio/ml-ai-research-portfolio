"""Data acquisition and preparation.

Research purpose: every empirical claim in the project must trace back to
documented, reproducible data. This module downloads prices from Yahoo
Finance (via yfinance), caches the raw download, documents missing data,
and converts prices to returns.

Nothing here fabricates data. `simulate_returns` exists ONLY for unit tests
and is deliberately labelled; its output must never be written to results/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Asset universe. Chosen to span equity, fixed income, commodities, sectors,
# international markets. Inception dates are approximate and should be
# verified by the user in notebook 01 (see research/limitations.md).
# ---------------------------------------------------------------------------
UNIVERSE: dict[str, str] = {
    "SPY": "US large-cap equity",
    "IWM": "US small-cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging-market equity",
    "XLF": "US financials sector",
    "XLE": "US energy sector",
    "XLK": "US technology sector",
    "XLV": "US health-care sector",
    "VNQ": "US real estate (REITs)",
    "TLT": "US Treasuries 20y+",
    "IEF": "US Treasuries 7-10y",
    "LQD": "US investment-grade credit",
    "HYG": "US high-yield credit",
    "GLD": "Gold",
    "DBC": "Broad commodities",
}

DEFAULT_START = "2007-07-01"
DEFAULT_END = "2026-10-31"


def download_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_path: str | Path = "data/raw/prices_adjusted.csv",
    force: bool = False,
) -> pd.DataFrame:
    """Download adjusted close prices with yfinance and cache them to CSV.

    Requires internet access and `pip install yfinance`. If a cache exists
    and `force` is False, the cached file is used so results are
    reproducible even if Yahoo later revises history.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install yfinance: pip install yfinance") from exc

    tickers = tickers or list(UNIVERSE)
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data (network or ticker problem).")
    prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    return prices


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load a user-supplied CSV of prices (Date index, one column per asset)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def data_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset documentation of coverage and missing observations."""
    rows = []
    for c in prices.columns:
        s = prices[c]
        first, last = s.first_valid_index(), s.last_valid_index()
        inner = s.loc[first:last] if first is not None else s
        rows.append(
            {
                "asset": c,
                "description": UNIVERSE.get(c, ""),
                "first_obs": first,
                "last_obs": last,
                "n_obs": int(s.notna().sum()),
                "n_missing_inside_span": int(inner.isna().sum()),
                "pct_missing_inside_span": float(inner.isna().mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("asset")


def clean_prices(
    prices: pd.DataFrame, max_ffill_days: int = 3, drop_incomplete: bool = True
) -> pd.DataFrame:
    """Forward-fill short gaps only, then keep the common-history window.

    Long gaps are NOT filled (that would invent data). Rows where any asset
    is still missing are dropped so every model sees the same sample.
    """
    out = prices.sort_index().ffill(limit=max_ffill_days)
    if drop_incomplete:
        out = out.dropna(how="any")
    return out


def compute_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Convert prices to returns. kind: 'simple' or 'log'."""
    if kind == "log":
        r = np.log(prices).diff()
    elif kind == "simple":
        r = prices.pct_change()
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return r.dropna(how="all").dropna(how="any")


def to_frequency(returns: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """Aggregate simple returns to a lower frequency by compounding."""
    return (1.0 + returns).resample(freq).prod().sub(1.0).dropna(how="any")


def simulate_returns(
    n_obs: int = 1500, n_assets: int = 6, seed: int = 0, regime_shift: bool = True
) -> pd.DataFrame:
    """SIMULATED returns for unit tests only (never real, never for results).

    Multivariate Student-t innovations, a one-factor structure, and an
    optional mid-sample volatility/correlation shift.
    """
    rng = np.random.default_rng(seed)
    f = rng.standard_t(5, n_obs) * 0.006
    beta = np.linspace(0.6, 1.2, n_assets)
    idio = rng.standard_t(5, (n_obs, n_assets)) * 0.008
    r = f[:, None] * beta[None, :] + idio + 0.0003
    if regime_shift:
        half = n_obs // 2
        r[half:] = r[half:] * 1.8 + f[half:, None] * 0.004
    idx = pd.bdate_range("2015-01-01", periods=n_obs)
    return pd.DataFrame(r, index=idx, columns=[f"SIM{i+1}" for i in range(n_assets)])
