# Research note 5 — Monte Carlo methods

## Estimator and error
To estimate $\theta=\mathbb Eh(X)$ use $\hat\theta_n=\frac1n\sum h(X_i)$. It is unbiased, consistent (SLLN) and $\sqrt n(\hat\theta_n-\theta)\Rightarrow N(0,\operatorname{Var}h(X))$. The **standard error** is $s/\sqrt n$ and the CI $\hat\theta\pm1.96\,SE$ has asymptotic coverage 95% — checked by simulation (Experiment 5.1). Because the rate is $n^{-1/2}$ **independent of dimension**, MC beats grid-based quadrature in high dimension (Experiment 5.2, trapezoid vs MC and the ball volume); but hit-or-miss estimators collapse when the target region has tiny probability (volume of the unit ball in high dimension).

## Variance reduction
* **Antithetic variates**: average $h(X)$ and $h(-X)$ (or $h(1-U)$). Variance factor $\tfrac12(1+\rho)$ where $\rho=\operatorname{Corr}(h(X),h(X'))$; helps for monotone $h$.
* **Control variates**: $Y=h(X)-\beta(g(X)-\mathbb Eg)$, optimal $\beta=\operatorname{Cov}(h,g)/\operatorname{Var}g$; variance factor $1-\rho^2$.
* **Importance sampling**: $\mathbb E_ph=\mathbb E_q[h\,p/q]$; for a rare event the proposal shifted into the tail gives orders of magnitude lower variance, but a poor proposal can have infinite variance. Experiment 5.5 estimates a normal tail probability with a shifted normal.
* Always compare by **efficiency** $=1/(\text{variance}\times\text{time})$, not variance alone.

## Cautions
Pseudo-random numbers are deterministic given the seed; results are reproducible but still *statistical estimates* with error bars. Report the SE every time.

## Link to the code
`05_monte_carlo_methods/src/monte_carlo.py`.
