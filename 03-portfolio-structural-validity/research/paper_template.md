# Beyond Optimization: Structural Stability and Out-of-Sample Performance of Portfolio Rules

**Author:** David Quayefio  
**Manuscript status:** empirical draft; replace all bracketed prompts with results from a documented, frozen run.  
**Scope:** five-ETF sample; do not describe these results as evidence from the separate planned 15-ETF design.

> **Drafting rule.** Do not copy numerical findings from an older draft or infer results from code. Report a statistic only after verifying it against the executed cell output and the exact data snapshot used. Treat the composite as exploratory and avoid universal, causal, or predictive claims.

## Abstract

**Purpose.** This study examines how ten portfolio-construction rules differ in allocation stability, sensitivity to estimated inputs and market conditions, and net out-of-sample performance relative to equal weighting.

**Design and data.** The analysis uses [confirm the frozen data artifact and hash] and daily simple returns for SPY, EFA, AGG, GLD, and VNQ from 2012-01-04 through 2025-12-31. The baseline walk-forward design uses a rolling 504-observation estimation window, rebalances every 21 observations, and applies a proportional cost of 10 basis points per dollar traded on each side.

**Findings.** [Report verified results for the six research questions, including effect sizes, uncertainty, sample sizes, and the multiple-testing adjustment. Do not select only the most favorable result.]

**Contribution and limitations.** [State a contribution only after completing the literature-gap search. Characterize the five-component score as exploratory, identify the survivor-selected five-ETF universe and single historical path, and summarize the principal inference and cost-model limitations.]

**Keywords:** portfolio optimization; estimation error; weight stability; covariance shrinkage; out-of-sample evaluation; transaction costs

## 1. Introduction

Mean–variance optimization identifies allocations that are optimal conditional on estimated inputs. In practice, expected returns and covariances must be estimated, and small input errors can create large changes in optimized weights. This motivates evaluating portfolio rules not only by realized returns, but also by the stability of their allocations and their sensitivity to estimation, market regimes, and modeling choices.

Distinguish throughout the paper among:

- **Mathematical optimality:** optimality for a stated objective and specified inputs.
- **Statistical reliability:** the extent to which conclusions persist under sampling uncertainty and model assumptions.
- **Structural stability:** the observed stability of weights and outcomes under the defined temporal, input, and scenario perturbations.
- **Economic usefulness:** realized net performance under the stated costs, constraints, and evaluation period.
- **Out-of-sample robustness:** persistence across the particular holdout path and design variants tested; it does not imply generalization to other universes or periods.

The study's central question is: **Within this five-asset historical sample, how do ten portfolio rules differ in allocation stability, response to changes in estimated inputs and market conditions, and net out-of-sample performance relative to equal weighting?**

Introduce the six detailed research questions in [research_questions.md](./research_questions.md). Treat the analyses as comparative and exploratory; do not present post-analysis expectations as preregistered hypotheses. State the paper's contribution only after the literature-gap search is documented and relevant sources are verified.

## 2. Literature review and research positioning

Synthesize, rather than annotate, the strands in [literature_review.md](./literature_review.md):

1. Mean–variance optimization and estimation error.
2. Equal weighting, constraints, and other simple benchmarks.
3. Covariance shrinkage, conditioning, and random-matrix diagnostics.
4. Robust, Bayesian, and risk-contribution approaches.
5. Time-varying volatility, regimes, and return-distribution assumptions.
6. Out-of-sample performance, Sharpe-ratio inference, and multiple testing.

Explain how these strands motivate separate measurement of portfolio weights, input sensitivity, risk, and realized performance. End with a **provisional** gap statement tied to a documented search. The literature review is currently narrative, not systematic; verify every citation against an authoritative source and complete the search and screening process before making a novelty claim.

**Provisional positioning prompt:** Does a transparent, multi-component profile of allocation stability and specified stress responses add descriptive information to conventional performance comparisons in this particular portfolio universe and evaluation design? Do not claim that the composite is novel or validated without broader literature review and independent validation.

## 3. Mathematical and portfolio framework

Use [mathematical_foundations.md](./mathematical_foundations.md) for derivations and the assumptions behind them. Include only the equations needed to define the paper's methods, and ensure the equations match the implemented code.

Let \(r_t\in\mathbb R^N\) be the vector of simple returns, with mean \(\mu\), covariance \(\Sigma\), and fully invested weights \(\mathbf 1^\top w=1\). For weights held over one interval,
\[
R_{p,t}=w^\top r_t,\qquad
\mathbb E[R_{p,t}]=w^\top\mu,\qquad
\operatorname{Var}(R_{p,t})=w^\top\Sigma w.
\]

The unconstrained global minimum-variance solution is a theoretical reference:
\[
w_{\mathrm{GMV}}=
\frac{\Sigma^{-1}\mathbf 1}
{\mathbf 1^\top\Sigma^{-1}\mathbf 1}.
\]
The core backtest uses long-only implementations, for which the optimizer solves a constrained problem and this closed form generally does not apply.

Define the other implemented rules and their project-specific choices: equal weight; target-return mean–variance; long-only maximum Sharpe; 25%-capped minimum volatility; Ledoit–Wolf minimum variance; robust mean–variance; equal risk contribution; Black–Litterman; and EWMA conditional-volatility minimum variance. Clarify that:

- Black–Litterman uses an equal-weight reference portfolio by default and sample-mean views; it is not a market-cap prior with independent investor views.
- Robust mean–variance uses \(\Omega=\hat\Sigma/T\), \(\kappa=1\), \(\gamma=5\), a Ledoit–Wolf covariance estimate, and sample means in the baseline implementation.
- EWMA uses a demeaned finite-window covariance with \(\lambda=0.94\); it is not a GARCH or dynamic-correlation model.
- Long-only constraints can regularize allocations and affect comparisons with unconstrained theoretical solutions.

For sample estimates, define
\[
\bar r=\frac1T\sum_{t=1}^T r_t,\qquad
S=\frac1{T-1}\sum_{t=1}^T(r_t-\bar r)(r_t-\bar r)^\top.
\]
If covariance shrinkage is discussed, state its target and estimator:
\[
\hat\Sigma_\alpha=(1-\alpha)S+\alpha F.
\]
Do not equate improved covariance estimation error with improved portfolio utility or net performance; evaluate these outcomes separately.

## 4. Research questions

Use these questions as the results-section structure. Do not add confirmatory hypotheses after inspecting the results.

1. **Estimation sensitivity:** How much do portfolio allocations move when estimation inputs are resampled or perturbed, including stationary-bootstrap resampling of the latest 504-observation window and one-standard-error perturbations to estimated means?
2. **Covariance and temporal stability:** How do estimated covariance structure and portfolio weights vary over time, and how do covariance-estimation choices relate to weight stability?
3. **Regimes and return distributions:** How do dependence, covariance diagnostics, and portfolio performance differ across the examined market regimes, and what distributional or volatility-clustering diagnostics characterize asset returns?
4. **Scenario and stress response:** How do allocations and estimated portfolio risk respond to the specific correlation, volatility, outlier, covariance-spectrum, and historical-return scenarios implemented in the study?
5. **Net out-of-sample performance:** How do the ten rules compare with equal weighting after modeled transaction costs, and how sensitive are results to cost rates, window lengths, rebalance intervals, and regime conditions?
6. **Structural profile and performance association:** How do component stability measures and composite ranks compare with net out-of-sample performance, and what cross-model association is observed?

## 5. Data and methodology

Use [methodology.md](./methodology.md) as the implementation reference. If
the final run differs from it, document the difference rather than silently
mixing design specifications.

### 5.1 Data, universe, and preprocessing

The current processed sample contains adjusted daily prices for SPY, EFA, AGG, GLD, and VNQ, obtained from Yahoo Finance via `yfinance` with `auto_adjust=True`. The aligned price sample has 3,520 observations from 2012-01-03 through 2025-12-31; simple returns have 3,519 observations from 2012-01-04 through 2025-12-31. Cite the cached source and processed files, the download timestamp, the exact analysis snapshot, and a cryptographic hash for the data used.

Report the asset roles, data-quality checks, and missingness/outlier handling. The current snapshot is described as having no missing aligned price observations, no rows dropped or values filled, and flagged outliers retained. The metadata flags observations exceeding an absolute five-standard-deviation threshold: SPY 12 days, EFA 12, AGG 10, GLD 6, and VNQ 12. Verify these counts against the data metadata for the run being reported.

The executed data-exploration notebook says not to forward-fill, while the current preprocessing function can forward-fill gaps of up to three observations. This did not alter the described sample because it has no missing prices, but harmonize and document the procedure before refreshing the data. Adjusted prices may be revised; use the cached file for reproduction. Discuss the survivor-selected universe, asynchronous closes across markets, ETF-specific structure, dividends/tracking differences, bond duration drift, and the sensitivity of results to asset selection.

### 5.2 Portfolio rules and implementation

Describe the ten core rules as implemented, including constraints, estimators, objective functions, risk-free-rate convention, and hyperparameters. All core strategies are long-only and fully invested in the walk-forward experiment unless explicitly stated otherwise. The unconstrained GMV solution is a theoretical reference, not an eleventh core strategy.

### 5.3 Walk-forward protocol and transaction costs

The baseline uses a rolling 504-observation estimation window and rebalances every 21 observations. At each rebalance, estimate weights only from information available before the holding interval; weights drift between rebalances. The baseline cost is 10 basis points per dollar traded on each side. With one-way turnover
\[
\operatorname{TO}_i=\frac12\lVert w_i^{\mathrm{target}}-w_i^{\mathrm{pretrade}}\rVert_1,
\]
the implementation deducts \(2c\,\operatorname{TO}_i\) from the first return after a rebalance, where \(c\) is the cost rate in decimal units.

The first rebalance is assigned zero turnover, so the cost of moving from initial equal-weight holdings to the first model target is omitted. Costs are proportional only; they do not capture market impact, capacity, taxes, borrowing, or stress-dependent bid–ask spreads. If an optimizer fails or returns invalid weights, the backtester retains prior drifted holdings and increments a failure counter. Report failures; a fallback is not a successful optimization.

Report the design variants exactly as run: 252 observations/21-day rebalance; 756 observations/63-day rebalance; and a 504-observation expanding design with 21-day rebalancing. The 756/63 variant jointly changes window and rebalance period. The notebook cost grid is 0, 5, 10, 25, and 50 basis points; do not attribute effects to only one changed parameter.

### 5.4 Structural-stability measures

Define the five components exactly as in [research_questions.md](./research_questions.md):

- **S1, weight instability:** mean one-way \(L_1\) movement between consecutive target-weight vectors, without within-period drift.
- **S2, input sensitivity:** mean one-way distance from the full-window weights to stationary-bootstrap weights from the latest 504-observation window. The composite analysis uses 100 attempted resamples and mean block length 10; report successful fits.
- **S3, ex-ante/OOS gap:** the common-moment in-sample Sharpe minus gross realized out-of-sample Sharpe. The in-sample moments are the estimation-window sample mean and covariance; this is not each strategy's own forecast Sharpe.
- **S4, regime inconsistency:** sample standard deviation of net out-of-sample Sharpe across ex-post volatility terciles based on a 21-observation rolling volatility measure.
- **S5, perturbation fragility:** mean one-way weight movement under 30 random one-standard-error perturbations to estimated means. Other stress tests do not enter S5.

The composite is the arithmetic mean of within-component ranks, with rank 1 assigned to the lowest value. It gives equal importance to components, discards magnitude information, and is not a validated scale of portfolio quality. Report the raw components and leave-one-component-out ranking analysis alongside it.

### 5.5 Covariance, regime, distribution, and stress analyses

Describe only analyses actually run. These include rolling covariance diagnostics; covariance spectra and Marchenko–Pastur reference bounds; shrinkage and eigenvector stability; ex-post and ex-ante volatility labels; bull–bear and stress labels; regime-conditional performance; Box's \(M\) and Frobenius comparisons; tail dependence; normality and ARCH-LM tests; and ADF tests when the required price file is available. Ex-post labels are descriptive, not investable signals. Regime performance is omitted below the implemented minimum observation count. Box's \(M\) is sensitive to non-normality; Marchenko–Pastur bounds rely on iid-noise assumptions. State the data timing, sample sizes, and thresholds for each label and test.

Notebook stress tests include controlled changes to correlation and volatility, common and idiosyncratic outliers, covariance-eigenvalue perturbations, and historical scenario replay. Report the exact shock level, number of replications, random seed, asset(s) affected, and baseline used for each reported result. Historical episodes are illustrative scenarios, not a statistical sample of crises; pre-sample windows cannot support an estimate using a full preceding 504-observation window.

### 5.6 Performance metrics and statistical inference

Unless stated otherwise, annualize with 252 trading days and use a zero risk-free rate. Report net and gross returns separately. Define annualized return, volatility, Sharpe, Sortino, maximum drawdown, Calmar, VaR/CVaR, turnover, effective number of assets, and certainty-equivalent gains as implemented. State that the zero risk-free rate can materially affect Sharpe ratios, particularly for bond-heavy portfolios.

The notebook's primary Sharpe comparisons use paired stationary-bootstrap resamples of daily strategy returns, mean block length 10, and 1,000 replications. Confidence intervals are percentile intervals; p-values use a recentered bootstrap difference distribution. This is a simplified, indicative procedure, not the full studentized Ledoit–Wolf test. The nine non-equal-weight comparisons with equal weighting use a Bonferroni threshold of \(0.05/9\approx0.0056\). This correction does not cover all exploratory analyses. Report unadjusted estimates and intervals as well as adjusted conclusions; discuss untested block-length sensitivity and dependence in overlapping rolling windows.

The bootstrap Sharpe and certainty-equivalent comparisons use net returns. S3 uses gross out-of-sample Sharpe. Distinguish statistical significance from economic magnitude and state that bootstrap uncertainty reflects only part of the uncertainty from evaluating one historical path.

## 6. Results

Populate this section only from verified outputs of the frozen run. Cite the relevant notebook tables and figures and the corresponding exported files under `results/tables` or `results/figures`. Do not retain numerical values from a prior draft unless they reproduce from the same data, code, and settings.

### 6.1 Sample and data diagnostics

[Report sample dates and counts, assets, price/return data checks, missing values, outlier flags, and any ADF results. Identify the data snapshot and hash.]

### 6.2 RQ1 — Estimation sensitivity

[Report bootstrap weight movement, mean-perturbation response, successful replicate counts, and other specified input-sensitivity results. Separate mean perturbations from covariance and return-shock scenarios.]

### 6.3 RQ2 — Covariance and temporal stability

[Report condition numbers, eigenvalue and eigenvector diagnostics, shrinkage results, estimator comparisons, and rolling/expanding weight stability. Describe overlapping-window dependence.]

### 6.4 RQ3 — Regimes and return distributions

[Report regime definitions and observation counts, conditional covariance/performance results, dependence and tail diagnostics, Mardia/ARCH-LM/ADF results where available. Interpret tests as diagnostics, not standalone economic conclusions.]

### 6.5 RQ4 — Scenario and stress response

[For each stress, report the shock, affected assets, repetitions, seed, comparison baseline, weight movement, and risk response. Identify historical scenarios excluded for insufficient prior data.]

### 6.6 RQ5 — Net out-of-sample performance

[Report the complete model performance table, model failures, net and gross results, turnover and drawdowns, wealth paths, cost sensitivity, design variants, Sharpe differences and intervals, adjusted and unadjusted inference, and certainty-equivalent gains.]

### 6.7 RQ6 — Structural profile and performance association

[Report all five raw components, the composite ranking, leave-one-component-out ranks, and the model-level association with net out-of-sample Sharpe. Give \(n=10\) and avoid predictive or causal interpretations.]

## 7. Discussion

Answer each research question directly, using only the mapped evidence. Separate findings from explanations and distinguish observed associations from causal mechanisms. Discuss whether results are consistent with the literature without treating a small-sample replication as a general test of a theory. State when the data do not distinguish strategies or when rankings depend on the evaluation design.

Interpret a model's risk-adjusted return together with costs, turnover, drawdown, estimation sensitivity, and uncertainty. A high in-sample objective is not evidence of robust performance; a low composite rank is not proof of economic superiority. Do not call any strategy universally best.

## 8. Limitations

Address each applicable item in [limitations.md](./limitations.md), including:

- **Data:** current ETFs create survivorship and selection concerns; Yahoo Finance data can be revised; asynchronous closes can bias daily cross-correlations; five assets and one common history cover few independent crises; fund-specific structure, tracking, dividends, and bond duration are not modeled.
- **Models:** Black–Litterman uses equal-weight prior weights and sample-mean views; robust mean–variance, caps, and risk-parity parameters are researcher choices; EWMA is not GARCH/DCC; long-only constraints act as implicit regularization; SLSQP and non-convex maximum-Sharpe optimization can fail, with fallbacks counted rather than treated as success.
- **Statistics:** regime labels are partly ex-post and uncertain; Box's \(M\) is sensitive to non-normality; Marchenko–Pastur assumptions may fail; bootstrap block-length sensitivity is untested; the rank composite loses magnitude and is based on ten models; multiple testing spans more than the adjusted nine comparisons; overlapping windows are dependent.
- **Economics:** proportional costs omit impact, spreads that widen in stress, taxes, borrowing, and capacity; the evaluation is one historical path; the zero risk-free rate affects Sharpe ratios.
- **Validation and process:** report whether statistical tests and optimizers were exercised on real data; disclose that `statsmodels`-based checks and a `cvxpy` cross-check require local validation if not run. Verify literature references before citation.

**Status check required before submission:** `limitations.md` currently states that no real-data results exist and all notebooks are unexecuted. The workspace also contains notebooks with execution outputs and files under `results/`. Verify the provenance and freshness of those outputs, then reconcile that statement with the actual run. Do not treat file presence alone as evidence that the tables correspond to the final code and data snapshot.

## 9. Conclusion

[Summarize only findings verified from the final run. State what the evidence supports for the specified universe, period, models, and costs; what remains uncertain; and which additional universes, periods, cost models, block-length choices, or parameter sweeps are needed. Do not claim universal superiority, prediction, causation, or a validated structural-validity scale.]

## References

Include only sources verified against authoritative bibliographic records. The review in [literature_review.md](./literature_review.md) is a narrative synthesis with a provisional research gap, not a systematic review. Complete and document the search and screening process before making novelty claims.

## Appendices

A. Derivations and model-specific objective functions.  
B. Complete result tables, figures, regime counts, and optimizer-failure logs.  
C. Data provenance, snapshot hash, metadata, package versions, and run instructions.  
D. Bootstrap settings, random seeds, successful replicate counts, and multiple-testing calculations.  
E. Additional design variants, sensitivity analyses, and robustness checks.
