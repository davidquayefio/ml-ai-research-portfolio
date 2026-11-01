# Module 2 — Probability Distributions: exercises

Work these **on paper first**, then check with the code in `02_probability_distributions/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Problems

1. Derive $\mathbb E X$ and $\operatorname{Var}X$ for Binomial($n,p$) using indicators, and again from the MGF.
2. Derive the MGF of Gamma($k,\theta$) and use it to show that the sum of $m$ i.i.d. Exponential($\lambda$) is Gamma($m,1/\lambda$).
3. Show that $\operatorname{Var}X=M''(0)-M'(0)^2=\frac{d^2}{dt^2}\ln M(t)\big|_{t=0}$ (the cumulant generating function).
4. For $X\sim t_\nu$ explain why $\mathbb E|X|^k<\infty$ iff $k<\nu$. Which sample moments in Experiment 2.1 are unreliable for $\nu=3$?
5. Prove the probability integral transform for strictly increasing continuous $F$.
6. Give an example of $X,Y$ with $\operatorname{Cov}(X,Y)=0$, $Y=g(X)$ and $g$ non-constant different from $x^2$.
7. Show $\operatorname{Var}(\sum w_iX_i)=w^\top\Sigma w$ and explain why that equals $\|L^\top w\|^2$ when $\Sigma=LL^\top$.
8. Why is $f_X(x)>1$ possible but $P(X\in A)\le1$ always?
