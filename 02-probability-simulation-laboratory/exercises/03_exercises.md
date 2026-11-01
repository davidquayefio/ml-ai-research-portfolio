# Module 3 — Law Of Large Numbers: exercises

Work these **on paper first**, then check with the code in `03_law_of_large_numbers/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Problems

1. Prove Chebyshev's inequality from Markov's inequality $P(Y\ge a)\le\mathbb E Y/a$ for $Y\ge0$.
2. Show $\operatorname{Var}\bar X_n=\sigma^2/n$ and state which assumption you used. What changes if $\operatorname{Corr}(X_i,X_j)=\rho>0$ for all $i\ne j$? (Hint: the variance no longer tends to $0$.)
3. For Bernoulli($p$), find the smallest $n$ guaranteed by Chebyshev to give $P(|\bar X_n-p|\ge0.05)\le0.05$ for *every* $p$, and compare with the CLT-based $n$ at $p=\tfrac12$.
4. Show with characteristic functions that the mean of $n$ i.i.d. Cauchy variables is Cauchy. Why does that contradict no law?
5. Give an example (a sequence $Y_n$) that converges in probability but not almost surely. (Hint: "typewriter" indicator sequence.)
6. For Pareto with tail index $\alpha\in(1,2)$ explain why the mean exists, the variance does not, and the sampling error decays more slowly than $n^{-1/2}$.
7. A trading strategy's daily returns have mean $0.05\%$ and sd $1\%$. How many days of i.i.d. data are needed before the standard error of the mean is a quarter of the mean? Comment on what this implies about estimating expected returns.
