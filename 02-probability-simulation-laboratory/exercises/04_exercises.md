# Module 4 — Central Limit Theorem: exercises

Work these **on paper first**, then check with the code in `04_central_limit_theorem/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Problems

1. Verify that $Z_n$ has mean $0$ and variance $1$ for every $n$. Which assumption provides the variance statement?
2. Use the characteristic-function sketch to show the CLT for sums of i.i.d. Bernoulli($p$) — i.e. the de Moivre–Laplace theorem. How large must $np$ and $n(1-p)$ be for the normal approximation to the Binomial to be accurate? Test with simulation.
3. If daily returns have standard deviation $1\%$, what is the standard deviation of the 252-day average return? What is the probability (under the CLT) that the annual average daily return differs from its mean by more than $0.1\%$?
4. Show $\operatorname{skew}(\bar X_n)=\gamma/\sqrt n$ from $\kappa_3(\sum X_i)=n\kappa_3(X)$ (additivity of cumulants).
5. Why is the CLT *not* a statement about the distribution of the data itself?
6. Give an example of i.i.d. data with finite mean but infinite variance. What is the right scaling for $\bar X_n$?
7. Explain why a "95% confidence interval" for the mean of a skewed distribution can systematically undercover. How would you detect this by simulation?
