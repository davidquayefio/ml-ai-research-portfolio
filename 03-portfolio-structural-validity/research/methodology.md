# Methodology

## 1. Study design and analysis snapshot

This study compares portfolio construction rules along two dimensions:
out-of-sample investment performance and structural stability under changes
in estimated inputs, market regimes, and modeling assumptions. Performance
and stability are reported separately; the composite stability score is an
exploratory summary, not a validated measure of portfolio quality.

**The current processed data and executed notebooks use a five-ETF sample**:
SPY, EFA, AGG, GLD, and VNQ. The aligned price data cover 2012-01-03 through
2025-12-31 (3,520 observations); the simple-return sample covers
2012-01-04 through 2025-12-31 (3,519 observations). The sample is daily.
These files do not match the downloader's configured 15-ETF universe or its
requested 2007-07-01 to 2026-10-31 window. Results from the current five-ETF
sample must not be described as results for the planned 15-ETF study.

There is also a file-path mismatch: the notebooks use
`data/processed/returns_simple_daily.csv`, whereas `scripts/run_pipeline.py`
expects `data/processed/returns_daily.csv`; `scripts/download_data.py`
generates the latter. The current artifacts and acquisition pipeline must be
reconciled before claiming that the existing results can be regenerated
end-to-end from the download script.

## 2. Data and preprocessing

### 2.1 Source, universe, and coverage

Adjusted daily closing prices were obtained from Yahoo Finance through
`yfinance` with `auto_adjust=True`. The archived source file is
`data/raw/yfinance_raw_20261111.csv`. The processed artifacts are
`data/processed/prices_adjusted_aligned.csv` and
`data/processed/returns_simple_daily.csv`. The recorded download time is
2026-11-11 UTC; the requested interval in the metadata is 2012-01-01 through
2026-01-01, with the end date exclusive.

The five funds represent U.S. large-cap equities (SPY), developed-market
international equities (EFA), U.S. investment-grade bonds (AGG), gold
(GLD), and U.S. real estate investment trusts (VNQ). This is a small,
researcher-selected universe of funds that exist today. It is not a
survivorship-free market universe, and results should not be generalized to
all investable assets.

### 2.2 Alignment, missing values, and returns

Prices are aligned to a common date index before returns are computed. In
the current data snapshot, the five price series have no missing
observations over the aligned sample; no rows were dropped, no missing values
were filled, and flagged outliers were retained. The metadata flags
observations exceeding an absolute five-standard-deviation threshold:
SPY 12 days, EFA 12, AGG 10, GLD 6, and VNQ 12. These flags are diagnostics,
not deletions or verified data errors.

The current preprocessing function can forward-fill gaps of up to three
observations before dropping incomplete rows, and the download script calls
that function. This differs from the executed data-exploration notebook,
which states that no forward filling is used. Because the current aligned
sample contains no missing prices, this discrepancy does not change the
present sample; it must nevertheless be resolved and documented before
processing a refreshed dataset.

Simple returns are used for portfolio aggregation:
\[
r_{i,t}=\frac{P_{i,t}}{P_{i,t-1}}-1.
\]
The first aligned price observation therefore does not produce a return
observation. Augmented Dickey–Fuller checks, where run, use simple returns
and log price levels; the project does not use log returns for portfolio
aggregation. Weekly simple returns can be formed by compounding daily
returns within each week.

## 3. Portfolio construction

Ten core construction rules are compared. All are long-only and fully
invested in the walk-forward experiment unless otherwise stated. The
unconstrained closed-form GMV solution is used as a theoretical reference,
not as one of the ten core backtest models.

1. **Equal weight:** \(w_i=1/N\).
2. **Long-only GMV:** minimize \(w^\top\hat\Sigma w\), subject to
   \(\mathbf 1^\top w=1\) and \(w_i\ge0\).
3. **Target-return mean–variance:** minimize estimated variance subject to
   the budget and long-only constraints and a target return. If no target is
   supplied, it is the midpoint between the long-only GMV sample-mean return
   and the highest individual sample mean in the estimation window.
4. **Maximum Sharpe:** maximize the long-only Sharpe ratio with a zero
   risk-free rate by default. If no asset has positive estimated excess
   return, the implementation falls back to long-only GMV.
5. **Capped minimum volatility:** long-only minimum variance with a 25%
   maximum asset weight. It differs from ordinary long-only GMV only when
   the cap binds.
6. **Ledoit–Wolf minimum variance:** long-only minimum variance using the
   scikit-learn Ledoit–Wolf covariance estimate, which shrinks toward a
   scaled-identity target.
7. **Robust mean–variance:** maximize estimated return less an ellipsoidal
   mean-uncertainty penalty and a variance penalty. The uncertainty
   covariance is \(\Omega=\hat\Sigma/T\); the baseline uses
   \(\kappa=1\) and risk aversion \(\gamma=5\). The covariance estimate is
   Ledoit–Wolf; the mean estimate is the sample mean. The objective is
   annualized using 252 trading days.
8. **Equal risk contribution:** solve numerically for long-only weights
   whose asset risk contributions are equal.
9. **Black–Litterman:** form the prior
   \(\pi=\delta\hat\Sigma w_{\rm ref}\), with \(\delta=2.5\),
   \(\tau=0.05\), and equal-weight reference weights by default. The
   implementation does not use market-cap weights. By default, it uses
   absolute views equal to annualized sample means, \(P=I\), and view
   uncertainty \(\Omega=\operatorname{diag}(\tau\hat\Sigma)\). The
   covariance and view means are annualized before optimization. This
   data-driven view specification is a project choice, not a market-view
   Black–Litterman implementation.
10. **EWMA conditional-volatility minimum variance:** long-only minimum
    variance using a demeaned, exponentially weighted covariance with
    decay \(\lambda=0.94\).

The models use different estimators and objectives, so differences in their
weights reflect both construction rules and input estimates. The
unconstrained GMV formula, which permits short positions, is analyzed
separately as a theoretical reference.

## 4. Walk-forward evaluation

### 4.1 Baseline protocol

The baseline uses a rolling estimation window of 504 daily observations and
rebalances every 21 observations (approximately monthly). At rebalance index
\(i\), weights are estimated using only returns before that date,
\[
\mathcal I_i=\{r_1,\ldots,r_{i-1}\},
\]
and applied to the next holding interval. Portfolio weights drift with
realized returns between rebalances. Unit tests check that changing future
returns does not change previously estimated weights.

The baseline cost parameter is 10 basis points per dollar traded on each
side. If one-way turnover is
\[
\operatorname{TO}_i=\frac12\lVert w_i^{\rm target}
-w_i^{\rm pretrade}\rVert_1,
\]
the implementation deducts \(2c\,\operatorname{TO}_i\) from the first
portfolio return after a rebalance, where \(c\) is the cost rate in decimal
units. Thus the cost model charges for both purchases and sales. It is
proportional only; it does not model market impact, capacity, taxes, or
asset-specific bid–ask spreads.

**Initial-trade caveat:** the backtester initializes holdings at equal
weights but sets turnover to zero at the first rebalance. It therefore does
not charge the cost of moving from equal-weight initial holdings to the
first model target. Reported net returns exclude that initial transition
cost.

If a model raises an exception or returns invalid weights, the backtester
retains its previous drifted holdings and increments a failure counter.
Failures should be reported by model; the fallback prevents the series from
stopping but does not make the failed optimization successful.

### 4.2 Design variants

The executed notebook compares the baseline with a 252-observation rolling
window rebalanced every 21 observations, a 756-observation window
rebalanced every 63 observations, and a 504-observation expanding-window
design rebalanced every 21 observations. Since the 756-observation variant
also changes rebalance frequency, its difference from baseline cannot be
attributed to window length alone.

The notebook evaluates transaction-cost rates of 0, 5, 10, 25, and 50
basis points per traded dollar on each side. The pipeline script currently
evaluates 0, 10, 25, and 50 basis points. These settings should be
harmonized before producing a single final set of tables.

## 5. Structural-stability analysis

The analysis reports five complementary components. Lower component values
are oriented as more stable, but the components are not assumed to measure a
single underlying construct.

| Component | Operational definition in the current analysis |
|---|---|
| **S1: Weight instability** | Mean one-way \(L_1\) distance between consecutive target-weight vectors. This summary does not account for within-period weight drift. |
| **S2: Input sensitivity** | Mean one-way distance between the baseline weights estimated on the latest 504-observation window and weights estimated on stationary-bootstrap resamples of that window. The standard analysis uses 100 resamples and mean block length 10. |
| **S3: Ex-ante/OOS gap** | Mean in-sample Sharpe based on the common sample mean and sample covariance of each estimation window, less realized gross out-of-sample Sharpe. This is a common-moment descriptive gap, not each model's own forecast Sharpe. |
| **S4: Regime inconsistency** | Sample standard deviation of net out-of-sample regime-conditional Sharpe ratios across ex-post volatility terciles, based on a 21-observation rolling volatility measure. |
| **S5: Perturbation fragility** | Mean one-way weight movement after adding a random one-standard-error shift to each asset's estimated mean; the composite uses 30 perturbations. |

The composite is the arithmetic mean of the within-component ranks, with
rank 1 denoting the lowest value. This avoids combining quantities with
different units, but it imposes equal importance on each component and
discards differences in magnitude. Raw component values must therefore be
reported with the composite, and rank sensitivity to omitting each component
should be shown. The composite is exploratory and has not been validated as
a general measure of portfolio quality.

Additional analyses examine:

- **Covariance structure:** rolling 252-observation covariance estimates,
  updated every 21 observations; condition numbers, leading-eigenvalue
  shares, average pairwise correlations, and changes in principal
  directions. Marchenko–Pastur bounds, Box's \(M\), and Frobenius-norm
  changes are supplementary diagnostics; their assumptions and
  non-normality sensitivity must be considered.
- **Market regimes:** ex-post volatility terciles, a 20% drawdown/recovery
  bull–bear rule, and prespecified stress windows: GFC (2008-09-01 to
  2009-03-31), Euro crisis (2011-07-15 to 2011-10-31), Q4 2018 selloff
  (2018-10-01 to 2018-12-31), COVID crash (2020-02-15 to 2020-04-30), and
  2022 rate shock (2022-01-01 to 2022-10-31). The first two precede the
  current sample. Regime performance is omitted when a regime has fewer
  than 30 observations. Expanding ex-ante volatility thresholds are
  available as a separate label. GMM and Markov-switching labels are
  optional analyses, not part of the baseline pipeline. Ex-post labels
  describe realized conditions and are not investable signals.
- **Return distributions:** univariate skewness, kurtosis and
  Jarque–Bera statistics; Mardia multivariate normality statistics;
  empirical tail co-exceedance; ARCH-LM tests; and, where run, ADF tests
  on returns and log price levels. A rejection of a statistical null is
  not by itself evidence of an economically important portfolio effect.
- **Controlled perturbations:** one-standard-error mean shifts, volatility
  scaling, correlation shifts, outlier injection, covariance-eigenvalue
  perturbations, and historical scenario replay. Only the mean-shift
  response enters S5 in the current composite; the other perturbations are
  reported separately.

## 6. Statistical inference and performance measures

### 6.1 Bootstrap comparisons

Sharpe-ratio differences are evaluated using paired stationary-bootstrap
resamples of daily strategy returns, with mean block length 10. Pairing
resamples preserves the contemporaneous relationship between each candidate
strategy and the equal-weight benchmark. Confidence intervals are
percentile intervals. The two-sided p-value is based on a bootstrap
difference distribution recentred at zero. This is a simplified procedure,
not the full studentized Ledoit–Wolf test; p-values are treated as
indicative. The executed out-of-sample notebook uses 1,000 resamples, while
the pipeline runner defaults to 500. The number of resamples must be fixed
and recorded for the final analysis.

The nine model-versus-equal-weight comparisons are evaluated at a
Bonferroni-adjusted familywise threshold of \(0.05/9\approx0.0056\) in the
notebook analysis. The pipeline writes unadjusted p-values; the adjustment
must be applied and documented separately. This correction covers those
nine comparisons only, not the full set of exploratory model, regime, and
perturbation analyses.

The bootstrap weight-sensitivity analysis resamples the latest estimation
window and reports the mean one-way weight movement from the original
estimate. For the controlled one-standard-error mean perturbation, the
pipeline uses 30 repetitions. Seeds and successful replicate counts should
be recorded.

### 6.2 Performance metrics

Unless stated otherwise, annualization uses 252 trading days and the
risk-free rate is zero. Annualized return is the geometric growth rate of
daily returns; annualized volatility is the sample standard deviation
multiplied by \(\sqrt{252}\); the Sharpe ratio is annualized mean excess
return divided by annualized volatility. Sortino uses downside deviation
about zero. Maximum drawdown is calculated from the compounded wealth
series; Calmar is annualized return divided by the absolute maximum
drawdown. Historical 5% VaR and CVaR are reported as positive daily loss
numbers. Effective number of assets is
\[
N_{\rm eff}=\frac{1}{\sum_{i=1}^{N}w_i^2}.
\]
Turnover is reported separately from returns.

Walk-forward performance tables report both gross and net returns. The
bootstrap Sharpe comparisons and certainty-equivalent (CE) comparisons use
net returns. The S3 gap uses gross out-of-sample Sharpe and the common
sample-moment in-sample Sharpe described above. CE gain is computed as the
difference in annualized mean–variance utility,
\[
\Delta CE=252(\bar r_c-\bar r_b)
-\frac{\gamma}{2}\,252(s_c^2-s_b^2),
\]
with \(\gamma=5\), candidate \(c\), benchmark \(b\), and daily sample
variances \(s^2\). Statistical evidence and CE magnitude are interpreted
separately.

## 7. Reproducibility and interpretation

The raw download, processed data, data metadata, notebooks, scripts, and
library versions should be retained for each analysis snapshot. The current
metadata records Python, NumPy, pandas, SciPy, Matplotlib, seaborn, and
yfinance versions, as well as the download date and outlier flags. Random
procedures use fixed seeds where implemented; the exact seed and
replication count must accompany reported results.

Before publication, reconcile the five-ETF processed sample with the
15-ETF downloader configuration, align the processed-data filenames used
by the notebooks and pipeline, harmonize the missing-data procedure,
bootstrap replication counts, and cost-sensitivity grid, and decide how to
handle initial-trade costs. Until these discrepancies are resolved, the
notebooks describe the current five-ETF analysis, while the downloader and
pipeline do not constitute a matching end-to-end reproduction of it.

The project's literature positioning and proposed research gap are
discussed in `literature_review.md`. No novelty claim should be made until
the proposed systematic search and broader robustness checks are completed.
