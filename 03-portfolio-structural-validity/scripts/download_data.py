"""Download the ETF universe from Yahoo Finance and cache it.

Run from the repository root (needs internet and `pip install yfinance`):

    python scripts/download_data.py

Creates:
    data/raw/prices_adjusted.csv        raw adjusted closes (cached download)
    data/processed/returns_daily.csv    cleaned simple daily returns
    results/tables/data_quality.csv     coverage / missing-data report
    results/reports/data_log.txt        provenance (date, versions, universe)
"""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_processing as dp  # noqa: E402


def main(force: bool = False) -> None:
    prices = dp.download_prices(
        cache_path=ROOT / "data/raw/prices_adjusted.csv", force=force
    )
    report = dp.data_quality_report(prices)
    report.to_csv(ROOT / "results/tables/data_quality.csv")
    clean = dp.clean_prices(prices, max_ffill_days=3)
    rets = dp.compute_returns(clean, "simple")
    rets.to_csv(ROOT / "data/processed/returns_daily.csv")

    try:
        import yfinance
        yf_version = yfinance.__version__
    except Exception:
        yf_version = "unknown"
    log = [
        f"Downloaded on: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"yfinance version: {yf_version}",
        f"Requested window: {dp.DEFAULT_START} to {dp.DEFAULT_END}",
        f"Universe: {', '.join(prices.columns)}",
        f"Raw rows: {len(prices)}",
        f"Common-history rows after cleaning: {len(clean)} "
        f"({clean.index[0].date()} to {clean.index[-1].date()})",
        f"Return rows: {len(rets)}",
        "Prices: auto-adjusted (splits + dividends). Returns: simple daily.",
        "Forward-fill limit: 3 days; remaining gaps -> row dropped.",
    ]
    (ROOT / "results/reports/data_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print("\nData quality report:\n", report.to_string())


if __name__ == "__main__":
    main(force="--force" in sys.argv)
