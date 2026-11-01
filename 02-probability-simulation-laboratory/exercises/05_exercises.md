# Module 5 — Monte Carlo Methods: exercises

Work these **on paper first**, then check with the code in `05_monte_carlo_methods/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Problems

1. Show that $\hat\theta_n$ is unbiased and compute its variance. How many samples are needed to get the relative error of a probability $p=10^{-3}$ below $5\%$?
2. Derive the variance of the antithetic estimator and show that $-1\le\rho\le1$ implies variance ratio in $[0,2]$. When can antithetic variates *hurt*?
3. Prove $\operatorname{Var}Y_{\beta^*}=\sigma_g^2(1-\rho^2)$.
4. For the tail probability $P(Z>c)$, show the IS weight is $e^{-cx+c^2/2}$ and compute $\mathbb E_q[w^2\mathbf 1\{X>c\}]$ to show the IS variance is finite.
5. Why can the $O(n^{-1/2})$ rate of Monte Carlo not be improved by simply taking *more careful averaging*? What changes with quasi-Monte Carlo?
6. In Experiment 5.2 the hit probability for the $d$-ball decays. Compute the limit of $V_d$ as $d\to\infty$. Why is that surprising?
7. An option payoff is $\max(S_T-K,0)$. Which variance reduction ideas from this module apply directly? Which quantity serves as a control variate with known expectation under the risk-neutral measure?
