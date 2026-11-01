# Probability Simulation Laboratory

A computational laboratory for learning probability *by experiment*: eight modules, **42 experiments**, each following the same research workflow — mathematical foundation → research question → theoretical prediction → simulation → statistical comparison → interpretation and limitations.

Built as a foundation for quantitative research: probability, Monte Carlo, stochastic processes, Brownian motion and financial applications (option pricing, VaR/ES, GARCH).

> **A simulation is evidence, not proof.** Every number below is a Monte Carlo estimate with sampling error; every financial result comes from an *assumed model* with illustrative parameters. No market data is used and nothing here is investment advice.

## Modules

| # | Module | Experiments | Main ideas |
|---|---|---|---|
| 01 | Probability fundamentals | 6 | empirical frequency, conditional probability, independence, Bayes, birthday, Monty Hall |
| 02 | Random variables and distributions | 5 | 11 distributions, moments, MGF, inverse-transform, joint laws |
| 03 | Law of Large Numbers | 5 | weak/strong LLN, Chebyshev, rates, Cauchy failure, sample-size planning |
| 04 | Central Limit Theorem | 4 | convergence, Berry–Esseen/Edgeworth, heavy tails, t-interval coverage |
| 05 | Monte Carlo methods | 5 | π, integration vs dimension, antithetic, control variates, importance sampling |
| 06 | Stochastic processes | 5 | random walks, gambler's ruin, Markov chains, Poisson process |
| 07 | Brownian motion & GBM | 5 | Donsker, quadratic variation, Itô correction, Euler error, reflection principle |
| 08 | Quantitative finance | 7 | returns, fat tails, MC option pricing, Greeks, Asian options, VaR/ES, GARCH |

## Quick start

```bash
git clone <your-fork-url> probability-simulation-laboratory
cd probability-simulation-laboratory
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                      # 105 tests, ~15 s
jupyter lab                    # open any notebook under 0*/notebooks/
```

The notebooks ship **with outputs already executed**, so you can read them on GitHub without running anything. They find the repository root automatically, so run them from anywhere inside the repo.

## Repository structure

```
probability-simulation-laboratory/
├── README.md, LICENSE, requirements.txt
├── src/plab/                       shared library: seeding, statistics helpers, plotting, process simulators
├── 01_probability_fundamentals/    ┐ each module has
├── …                               │   notebooks/  executed notebook (12-part experiments)
├── 08_quantitative_finance/        ┘   src/        tested code   |   results/  figures and CSV tables
├── tests/                          pytest suite (theory-vs-simulation checks at fixed seeds)
├── research_notes/                 8 derivation notes + references
├── exercises/                      paper exercises (no solutions)
├── visualizations/                 one headline figure per module
├── experiments/                    your own extensions
└── .github/workflows/tests.yml     CI
```

## Methodology

* **Reproducibility.** One seeded `numpy.random.Generator` (PCG64) per experiment (`make_rng(20261031 + k)`); every table and figure regenerates identically.
* **Predict first.** Each experiment states its theoretical prediction *before* the code; results are then compared with it using standard errors, confidence intervals (Wilson for proportions) or tests (KS, exact binomial).
* **Honest comparison.** When a claim failed the check during development the text was corrected, not the data. Where chance deviations of ~2 standard errors occur they are reported as such (e.g. Experiment 6.1's martingale check).
* **Tested code.** `tests/` checks library functions against exact answers (e.g. closed-form Black–Scholes values, gambler's-ruin formulas, stationary distributions).

## Selected findings (details, tables and caveats in the notebooks)

* **Random walks.** Var $S_n/n$ ≈ 1 at every $n$; gambler's ruin from fortune 10 of 20: win probability 0.5 fair vs 0.310 at $p=0.48$ (closed form 0.3099, simulation 0.3104).
* **Markov chain.** The weather chain has $\pi=(21,13,12)/46$ and $|\lambda_2|=0.473$; $10^6$-step occupation frequencies are within $5\times10^{-4}$ of $\pi$.
* **Brownian motion.** Quadratic-variation variance slope −0.996 (theory −1); total-variation slope +0.500; Euler strong-error slope −0.49 and weak-error slope −1.00; discretely monitored maximum matches the $0.5826\sqrt{\Delta t}$ shifted-barrier theory in all five Wilson intervals.
* **Itô correction.** With $\mu=0.04,\sigma=0.3$, 63% of GBM paths end below their start after 400 years although $\mathbb E S_T$ grows by a factor $\approx9\times10^6$.
* **Option pricing.** Monte Carlo reproduces Black–Scholes (10.4506) at rate $n^{-1/2}$ (slope −0.48); variance ratios ≈ 0.50 (antithetic) and ≈ 0.14 (control); a geometric-Asian control variate gives a 36× smaller standard error; simulating with the real-world drift instead of $r$ gives ≈ 18.1, a bias that never shrinks.
* **Risk.** For $t_4$ returns a normal-model 99% VaR is ≈ 12% too low and ES ≈ 28% too low regardless of sample size; historical VaR from 250 days has ≈ 17% relative standard deviation.
* **GARCH(1,1).** Uncorrelated returns, strongly autocorrelated squared returns, and an unconditional normal VaR exceeded 1.46% of days with exceedances clustering 6.4× above the base rate; conditioning on $\sigma_t$ restores ≈ 1.0%.

## Limitations

* No real market data; financial parameters are illustrative, not calibrated.
* Black–Scholes/GBM assumes constant volatility, continuous paths and no frictions; GARCH here is simulated, not estimated.
* Pseudo-random simulation gives estimates with sampling error; 1-in-20 "misses" of 95% intervals are expected and are not bugs.
* The notebooks were executed with a small custom runner because the build environment had no Jupyter; the SymPy cell in Module 2 is included but was **not executed** there, and `statsmodels` is listed in `requirements.txt` for your own extensions but is not used by the tests. Re-run the notebooks in Jupyter to refresh the stored outputs.

## How to use this as a learning project

1. Read the research note, then **attempt the paper exercises** for the module.
2. Predict each experiment's outcome in writing; run it; reconcile differences.
3. Change one thing (a distribution, a parameter, the sample size) in `experiments/` and write the 12-part report.
4. Suggested order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08.

## License

MIT — see `LICENSE`.
