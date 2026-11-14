# Research questions and evidence plan

## 1. Study scope and central question

The questions are bounded to the analysis currently implemented for five
exchange-traded funds: SPY, EFA, AGG, GLD, and VNQ. The processed sample
contains daily simple returns from 2012-01-04 through 2025-12-31 (3,519
observations). The comparison covers ten long-only portfolio-construction
rules; unconstrained global minimum variance is a theoretical reference,
not a core walk-forward strategy.

**Central research question.** In this five-asset historical sample, how do
the ten portfolio rules differ in the stability of their allocations, their
responses to changes in estimated inputs and market conditions, and their
net out-of-sample performance relative to equal weighting?

This is a comparative, exploratory study. Its results describe specified
models under one universe, period, and backtest design; they do not establish
universal superiority, causal effects, or predictive validity.

## 2. Research questions and evidence map

| ID | Research question | Primary evidence | Claim boundary |
|---|---|---|---|
| **RQ1 — Estimation sensitivity** | How much do portfolio allocations move when estimation inputs are resampled or perturbed, particularly under stationary-bootstrap resampling of the latest 504-observation window and one-standard-error perturbations to estimated means? | [Notebook 05](../notebooks/05_parameter_sensitivity.ipynb): bootstrap and mean-perturbation analyses; [Notebook 07](../notebooks/07_stress_testing.ipynb): controlled input perturbations; [Notebook 09](../notebooks/09_final_comparative_analysis.ipynb): S2 and S5 components. | Quantifies responses to the stated resampling and perturbation procedures. It is not a probability statement about all future inputs or market shocks. Replication counts vary by analysis and must be reported with each result. |
| **RQ2 — Covariance and temporal stability** | How do estimated covariance structure and portfolio weights vary over time, and how do covariance-estimation choices relate to weight stability? | [Notebook 04](../notebooks/04_covariance_stability.ipynb): covariance spectra, conditioning, eigenvector stability, and shrinkage; [Notebook 05](../notebooks/05_parameter_sensitivity.ipynb): rolling/expanding weights and estimator sensitivity; [Notebook 09](../notebooks/09_final_comparative_analysis.ipynb): S1. | Describes variation in the observed sample. Marchenko–Pastur bounds are a model-based noise benchmark, not proof that eigenvalues outside the band are economic signals. Consecutive target-weight changes do not include within-period drift. |
| **RQ3 — Regimes and return distributions** | How do dependence, covariance diagnostics, and portfolio performance differ across the market regimes examined, and what distributional or volatility-clustering diagnostics characterize the asset returns? | [Notebook 06](../notebooks/06_market_regime_analysis.ipynb): regime definitions, covariance comparisons, tail dependence, distribution tests, and ADF tests when the price file is available; [Notebook 08](../notebooks/08_out_of_sample_validation.ipynb): regime-conditional out-of-sample performance. | Regime summaries are conditional descriptions, not evidence that a regime caused a performance difference. Ex-post volatility labels use full-sample information and are not investable signals. Statistical test rejections alone do not demonstrate economic importance. |
| **RQ4 — Scenario and stress response** | How do allocations and estimated portfolio risk respond to the specific correlation, volatility, outlier, covariance-spectrum, and historical-return scenarios implemented in the study? | [Notebook 07](../notebooks/07_stress_testing.ipynb): controlled perturbations and historical scenario replay; [Notebook 06](../notebooks/06_market_regime_analysis.ipynb): high- versus low-volatility covariance comparison. | Measures sensitivity to selected scenarios only. The scenarios are neither probability-weighted forecasts nor a complete measure of resilience. Historical windows are usable only when the sample provides enough preceding data for estimation. |
| **RQ5 — Net out-of-sample performance** | How do the ten rules compare with equal weighting in the baseline walk-forward evaluation after modeled transaction costs, and how sensitive are results to cost rates, window lengths, rebalance intervals, and regime conditions? | [Notebook 08](../notebooks/08_out_of_sample_validation.ipynb): net performance, paired Sharpe comparisons, cost grid, design variants, and regime-conditional results. | Compares realized results for one historical path under the specified backtest. The proportional cost model omits market impact and capacity effects; the backtester also omits the cost of the initial transition from equal weights. |
| **RQ6 — Structural profile and performance association** | How do the models' component stability measures and composite ranks compare with net out-of-sample performance, and what cross-model association is observed? | [Notebook 09](../notebooks/09_final_comparative_analysis.ipynb): component table, composite ranking, leave-one-component-out analysis, Spearman association, and comparison figure. | The composite is exploratory and the association is based on ten related strategies. It is not a prediction test and cannot support causal or general claims that instability determines performance. |

These questions distinguish input and covariance diagnostics, controlled
stress tests, regime descriptions, and realized performance. Evidence from
one category is not treated as a substitute for another: for example,
non-normality does not by itself show that a portfolio rule is unstable or
economically inferior.

## 3. Operational measures used in the composite

The composite in Notebook 09 averages within-component ranks, with rank 1
assigned to the lowest value. Its five components are:

| Component | Operational definition | Interpretation and limitation |
|---|---|---|
| **S1 — Weight instability** | Mean \(\tfrac12\lVert w_{m,t}-w_{m,t-1}\rVert_1\) between consecutive target-weight vectors. | Measures target-weight movement, not turnover from drifted pre-trade holdings. |
| **S2 — Input sensitivity** | Mean \(\tfrac12\lVert w_m^{(b)}-\hat w_m\rVert_1\) across stationary-bootstrap estimates from the latest 504-observation window. | Conditional on one window, model specification, block length 10, and 100 attempted resamples in the composite analysis. Report successful fits. |
| **S3 — Ex-ante/OOS gap** | Common-moment in-sample Sharpe minus gross realized out-of-sample Sharpe. The in-sample Sharpe uses the estimation-window sample mean and sample covariance. | A descriptive gap, not each strategy's own forecast Sharpe or a direct calibration test. Negative values mean realized gross Sharpe exceeded this common-moment in-sample measure. |
| **S4 — Regime inconsistency** | Sample standard deviation of net out-of-sample Sharpe across ex-post volatility terciles, based on a 21-observation rolling volatility measure. | Depends on regime definitions and sample sizes; ex-post labels use full-sample thresholds. |
| **S5 — Perturbation fragility** | Mean \(\tfrac12\lVert w_m(\hat\mu+\delta\mu)-w_m(\hat\mu)\rVert_1\), where asset-mean estimates are perturbed using their estimated standard errors. | The composite uses 30 perturbations and mean shifts only. Covariance-focused models may be invariant to constant mean shifts; that does not imply robustness to covariance or return shocks. |

Lower values are treated as more favorable when ranking these components,
but that convention does not make the composite a validated scale of
portfolio quality. The arithmetic mean of ranks gives equal importance to
each component and discards magnitude information. Report the raw
components alongside the composite and include the leave-one-component-out
ranking analysis.

## 4. Inference and reporting rules

The primary inferential family comprises the nine non-equal-weight model
comparisons with equal weighting using net out-of-sample Sharpe ratios.
Notebook 08 uses 1,000 paired stationary-bootstrap resamples with mean block
length 10 and a Bonferroni threshold of \(0.05/9\approx0.0056\). The
recentered bootstrap p-values are indicative; the implementation is not the
full studentized Ledoit–Wolf test. This adjustment applies to those nine
comparisons, not to the full set of exploratory tests.

Report effect sizes and uncertainty together, including Sharpe differences
and confidence intervals, certainty-equivalent gains at \(\gamma=5\),
turnover, drawdown, tail-risk measures, model failures, and regime
observation counts where available. Statistical significance and economic
magnitude are distinct. Cost results apply only to the implemented
proportional turnover model.

Do not present post-analysis expectations as preregistered hypotheses. Do
not interpret the cross-model association in RQ6 as confirmatory or
predictive inference; report its sign, magnitude, and number of models, and
avoid causal language.

## 5. Reproducibility and interpretation

The notebook code writes tables and figures under `results/tables` and
`results/figures` when the relevant cells are run. Generated files should be
retained with the executed notebooks and the exact data snapshot used.

The current five-ETF sample is distinct from the downloader's planned
15-ETF, 2007–2026 design. The notebooks load
`data/processed/returns_simple_daily.csv`, whereas the pipeline expects
`data/processed/returns_daily.csv`. The pipeline and notebook settings also
differ in bootstrap replication counts and transaction-cost grids. Reconcile
these settings and the initial-trade-cost treatment before describing the
analysis as an end-to-end reproducible pipeline or combining results across
the two designs.
