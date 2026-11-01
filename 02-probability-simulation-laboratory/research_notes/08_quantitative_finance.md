# Research note 8 — Financial probability

> All results come from assumed models. No market data is used; nothing here is a forecast or investment advice.

## Returns
$R_t=S_t/S_{t-1}-1$, $r_t=\ln(1+R_t)$. Log returns add over time; simple returns add across assets. Under GBM the log return is normal and the simple return is right-skewed with $\mathbb E[e^r-1]=e^{m+s^2/2}-1$.

## Risk-neutral pricing
$V_0=e^{-rT}\mathbb E^{\mathbb Q}[h(S_T)]$ with drift $r$ under $\mathbb Q$ (Black–Scholes market). Call: $C=S_0\Phi(d_1)-Ke^{-rT}\Phi(d_2)$. Put–call parity $C-P=S_0-Ke^{-rT}$. Using the real-world drift $\mu$ in the simulation prices the wrong measure; Experiment 8.3 shows the resulting bias (about 18.1 instead of 10.45 for $\mu=0.15$) does not shrink with $n$.

## Monte Carlo accuracy
Plain MC error $\propto n^{-1/2}$; antithetic variance ratio ≈ 0.50, control variate ≈ 0.14 for the European call; geometric-Asian control variate gives a ≈ 36× smaller SE for the arithmetic Asian call (payoff correlation 0.9996). Common random numbers cut delta-estimation spread ≈ 32×.

## Risk measures
$\mathrm{VaR}_\alpha=F_L^{-1}(\alpha)$, $\mathrm{ES}_\alpha=\mathbb E[L\mid L\ge\mathrm{VaR}_\alpha]$. For $t_4$ losses with sd 1.2% the normal model under-states 99% VaR by about 12% and ES by about 28% — a *model* bias that does not shrink with more data — while historical VaR from 250 days has relative sd ≈ 17% (estimation error that does shrink).

## GARCH(1,1)
$\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta\sigma_{t-1}^2$; stationary iff $\alpha+\beta<1$. Gaussian shocks still give excess kurtosis $6\alpha^2/(1-\beta^2-2\alpha\beta-3\alpha^2)$ and clustered extremes: with $(\alpha,\beta)=(0.10,0.88)$ an unconditional normal 99% VaR is exceeded 1.46% of the time and an exceedance is followed by another with probability 0.094 (6.4× the base rate).

## Limitations of the whole module
Constant volatility (Black–Scholes), no jumps, no transaction costs, parameters illustrative not estimated, i.i.d. assumption in the VaR study. These are the next things a real model must address (local/stochastic volatility, jump-diffusions, estimated GARCH).

## Link to the code
`08_quantitative_finance/src/finance.py`.
