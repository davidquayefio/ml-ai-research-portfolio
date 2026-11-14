# Limitations (known before any result exists)

## Data
1. **Survivorship.** Only ETFs alive today. Biases performance favourably; cannot be fixed with free data.
2. **Yahoo Finance.** Unofficial, revisable, personal-use licence. Adjusted prices may differ between downloads; use the cached CSV.
3. **Short common history.** Set by the youngest ETF. The window may contain few independent crises (2008-09, 2011, 2018, 2020, 2022 at best). Regime conclusions rest on very few episodes.
4. **Asynchronous closes.** US and international ETFs close at different times; daily cross-correlations are biased downward for non-US assets. Weekly returns reduce but do not remove this.
5. **ETF specifics.** Fund structure, dividends and tracking differences, and a bond ETF's duration drift, are not modelled.
6. **Universe choice.** Results may depend on the 15-asset selection. Test with other subsets.

## Models
7. **Black-Litterman prior** uses equal weights instead of market-cap weights, and its views are generated from sample means. This is a design choice, not a standard BL setup.
8. **Robust MV and risk-parity parameters** (kappa, gamma, caps) are researcher choices; sweeps are needed before generalising.
9. **Conditional volatility** is EWMA, not GARCH/DCC. Dynamic-correlation models are not implemented.
10. **Long-only constraint** acts as implicit shrinkage (Jagannathan and Ma, 2003), which partly masks estimation error in constrained models. Compare with unconstrained versions to see this.
11. **Solver.** SciPy SLSQP is local; max-Sharpe is non-convex in the ratio form and uses multi-start. Failures fall back to previous weights and are counted.

## Statistics
12. **Regime labels** are conventional and partly ex-post. Ex-post labels are descriptive only; ex-ante labels are noisy.
13. **Box's M** is highly sensitive to non-normality. A rejection is weak evidence on its own.
14. **Marchenko-Pastur** assumes iid observations; heavy tails and clustering move the noise edge.
15. **Sharpe-difference test** is a simplified bootstrap. Block length sensitivity is untested by default.
16. **Composite index** uses rank averaging over 10 models; ranks lose magnitudes and the sample is tiny. Always read components.
17. **Multiple testing** inflates false positives across models, tests, windows and costs. Corrections must be applied and reported.
18. **Overlapping windows** make rolling-window diagnostics serially dependent; do not treat consecutive windows as independent observations.

## Economic
19. **Costs** are proportional only. No impact, spreads that widen in stress, borrowing, tax or capacity.
20. **Out-of-sample is one path.** A single historical path is one draw from the data-generating process. Bootstrap uncertainty reflects only part of that.
21. **Risk-free rate** is zero by default; Sharpe ratios of bond-heavy portfolios are sensitive to this in high-rate periods. Use FRED T-bill data (a free source) as an extension.

## Process limitations of this repository
22. **No real-data results exist in this repository.** The code was tested only on simulated data. Every notebook is committed unexecuted. Nothing in `results/` is a finding until you generate it.
23. `statsmodels`-based pieces (ADF table, Markov-switching) were **not** exercised in the build environment (package not installed there); run them locally and fix anything that breaks.
24. `cvxpy` is listed in requirements but the shipped optimisers use SciPy. A cvxpy cross-check of the convex problems is a worthwhile validation step.
25. The literature review was written from the standard literature without access to full texts in the build session; verify every reference before citing.
