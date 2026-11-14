# Beyond Optimization: Testing the Structural Validity of Portfolio Models

**Author:** David Quayefio  
**Type:** Independent, exploratory quantitative-finance research project (Python)  
**Paper:** [`paper/paper.pdf`](paper/paper.pdf) (LaTeX source in `paper/paper.tex`)

## Overview

Mean-variance portfolio optimisation (Markowitz, 1952) depends on estimated expected returns and covariances. Those estimates are noisy, so a portfolio that is mathematically optimal for the estimated inputs may not be statistically reliable or economically useful (Michaud, 1989). This project asks:

> To what extent do conventional portfolio optimisation models retain their structural validity when their statistical assumptions, estimated parameters and underlying market conditions are challenged?

Rather than ranking rules by Sharpe ratio alone, it profiles ten portfolio construction rules on five stability components and relates that profile to net out-of-sample performance.

## Research objectives

1. Examine the mathematical foundations and assumptions underlying conventional portfolio optimisation.
2. Investigate the statistical reliability of expected-return, variance, covariance and correlation estimates.
3. Evaluate the sensitivity of optimal portfolio weights to estimation errors.
4. Test whether portfolio models remain stable across different market regimes.
5. Investigate the effects of non-stationarity, changing correlations, volatility clustering and extreme market observations.
6. Compare theoretical portfolio performance with genuinely out-of-sample performance.
7. Determine whether more sophisticated portfolio models provide meaningful improvements over simpler alternatives.
8. Establish a framework for evaluating portfolio models beyond conventional optimisation objectives.

## Hypotheses

| ID | Hypothesis |
|----|-----------|
| H1 | Optimised portfolios beat equal weight out of sample, net of costs |
| H2 | Shrinkage improves weight stability |
| H3 | Robust optimisation reduces input sensitivity |
| H4 | Weights are less stable in stress |
| H5 | Estimation error materially raises realised risk |
| H6 | Gains from optimisation are economically meaningful |
| H7 | Simple methods win after costs |

The criteria for judging each were written after the results were seen, so the verdicts in the paper are descriptive, not pre-registered tests.

## Data

- **Universe:** SPY, EFA, AGG, GLD, VNQ (US equity, international equity, bonds, gold, real estate)
- **Source:** Yahoo Finance via `yfinance` (auto-adjusted prices), downloaded 2026-11-11
- **Period:** 2012-01-04 to 2025-12-31, 3,519 daily simple returns
- `scripts/download_data.py` is written for a wider 15-ETF design; the notebooks and paper use the five-ETF sample above.

## Models (10)

Equal weight; global minimum variance (long-only; unconstrained closed form in notebook 02); mean-variance with target return; maximum Sharpe; minimum volatility with a 25% cap; Ledoit-Wolf minimum variance; robust (ellipsoidal) mean-variance; risk parity (equal risk contribution); Black-Litterman; conditional-volatility (EWMA, lambda = 0.94) minimum variance.

## Structural validity profile

Five components, each oriented so that lower means more valid, plus a weight-free composite (mean of ranks):

| Component | Measures |
|-----------|----------|
| S1 | Weight instability (turnover between rebalances) |
| S2 | Input sensitivity (bootstrap weight movement) |
| S3 | Overfitting gap (common-moment in-sample Sharpe minus gross out-of-sample Sharpe) |
| S4 | Regime inconsistency (dispersion of Sharpe across volatility regimes) |
| S5 | Perturbation fragility (weight movement under mean perturbation) |

## Evaluation design

- Walk-forward backtest: 504-day estimation window, rebalanced every 21 days, no look-ahead.
- Costs: 10 bps per dollar traded on each side (cost = 2c x turnover). No initial-trade cost is charged.
- Inference: stationary block bootstrap (1,000 resamples, mean block length 10); Bonferroni threshold 0.05/9 over nine comparisons with equal weight; certainty-equivalent gain at risk aversion 5.
- Robustness: cost levels, window length, rebalance interval, expanding window.

## Main findings (descriptive, five-ETF sample)

- Capped minimum volatility and risk parity had the highest net out-of-sample Sharpe ratios (0.912 and 0.909) against 0.798 for equal weight.
- Capped minimum volatility versus equal weight: Sharpe difference 0.114, 95% interval [0.011, 0.223], unadjusted p = 0.036, certainty-equivalent gain 0.008. No comparison passes the Bonferroni threshold.
- Composite structural ranks (lower is better): equal weight 1.4, capped minimum volatility 3.6, GMV and risk parity 4.0, Ledoit-Wolf 4.2, EWMA 5.8, mean-variance 6.4, Black-Litterman 7.2, robust 8.8, maximum Sharpe 9.6.
- Rules that depend on estimated means were the least structurally stable; mean-estimation error appears to dominate covariance error in this small universe.
- The composite score and net Sharpe are negatively associated but imprecisely (Spearman rho = -0.505, p = 0.137, n = 10).
- The ranking depends on design: with longer windows, some mean-dependent rules exceed equal weight on point estimates (untested).

### Hypothesis status

| Hypothesis | Verdict |
|-----------|---------|
| H1 | Not supported (baseline design only; variants untested) |
| H2 | Not supported |
| H3 | Inconclusive |
| H4 | Inconclusive (turnover by regime pending) |
| H5 | Not supported at T = 504 |
| H6 | Not supported |
| H7 | Inconclusive |

## Repository layout

```
paper/          paper.pdf, paper.tex, figs/
src/            data_processing, portfolio_models, covariance_estimators,
                stability_tests, regime_detection, stress_testing,
                backtesting, evaluation
notebooks/      01 data -> 02 maths -> 03 models -> 04 covariance -> 05 sensitivity
                -> 06 regimes and distributions -> 07 stress -> 08 out-of-sample -> 09 synthesis
scripts/        download_data.py, run_pipeline.py, extra_analyses.py
tests/          pytest suite (simulated data only)
research/       research_questions, methodology, mathematical_foundations,
                literature_review, limitations, paper_template
results/        figures/, tables/, reports/
data/           raw/ (git-ignored), processed/
```

## Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py    # fetches and cleans prices
pytest -q                          # unit tests on simulated data
python scripts/run_pipeline.py     # tables and figures into results/
python scripts/extra_analyses.py   # volatility scaling, normality, tail dependence,
                                   # turnover by regime, variant significance
jupyter notebook notebooks/        # interactive, notebooks 01 to 09
```

Downloaded prices come from a live source and may change slightly with later vendor adjustments, so re-running the download may not reproduce the reported numbers exactly. The processed returns file used for the paper is in `data/processed/`.

## Limitations

- Five survivor-selected ETFs and a single historical path; results do not generalise automatically.
- No initial-trade cost; bootstrap test is a simplified version, not the full Ledoit-Wolf (2008) studentised test.
- Regimes are ex-post volatility terciles; ex-ante labels give different counts.
- S5 has ties; the composite is a rank average with equal weights.
- Volatility is modelled with EWMA, not GARCH or DCC.
- Black-Litterman uses an equal-weight prior and data-driven views, not a market-capitalisation prior with independent views.
- No claim of novelty is made; a systematic literature search is still needed.
- Several references should be re-checked against the originals before citing (see `research/literature_review.md`).

Full discussion: `research/limitations.md` and Section 5 of the paper.

## Key references

- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77-91.
- Michaud, R. O. (1989). The Markowitz optimization enigma: Is "optimized" optimal? *Financial Analysts Journal*, 45(1), 31-42.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification. *Review of Financial Studies*, 22(5), 1915-1953.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.

The complete reference list is in the paper.

## Integrity statement

No data, results, tests or citations are fabricated. Every figure and table in the paper comes from the notebook and script outputs in `results/tables/`. Simulated data appears only in unit tests and smoke tests.

## License

Add a licence of your choice (for example MIT for code) before making the repository public.