# Research note 3 — Law of Large Numbers

## Statements
Let $X_i$ be i.i.d. with $\mathbb E|X_1|<\infty$, mean $\mu$, and $\bar X_n=\frac1n\sum X_i$.
* **Weak LLN**: $\bar X_n\to\mu$ in probability.
* **Strong LLN**: $\bar X_n\to\mu$ almost surely (Kolmogorov).
* Mean-square: if $\operatorname{Var}X_1=\sigma^2<\infty$, $\mathbb E(\bar X_n-\mu)^2=\sigma^2/n\to0$.

## Proof of the WLLN under finite variance (Chebyshev)
$P(|\bar X_n-\mu|\ge\varepsilon)\le\dfrac{\operatorname{Var}\bar X_n}{\varepsilon^2}=\dfrac{\sigma^2}{n\varepsilon^2}\to0.$
This bound is *conservative*; the CLT (note 4) gives the much sharper $\approx2\bar\Phi(\varepsilon\sqrt n/\sigma)$. Experiment 3.2 compares them.

## Rates and failure
* Typical error $\sim\sigma/\sqrt n$: a log–log plot of $|\bar X_n-\mu|$ against $n$ has slope $-1/2$ (Experiment 3.3). Ten times the accuracy costs a hundred times the samples.
* **Cauchy counter-example**: $\mathbb E|X|=\infty$ and $\bar X_n$ is again standard Cauchy for every $n$; it never settles down (visible in Experiment 3.1). The LLN hypotheses are not decoration.
* Almost-sure convergence is a statement about each *individual path*; the convergence *in probability* shown by many-path envelopes is weaker.

## Practical use
Sample-size planning: to estimate a mean to within $\pm\delta$ with 95% confidence, $n\approx(1.96\,\sigma/\delta)^2$ (Experiment 3.5). $\sigma$ is itself unknown and must be estimated or bounded.

## Link to the code
`03_law_of_large_numbers/src/lln.py` (`running_mean`, `chebyshev_bound`, `clt_approximation`, `loglog_slope`, `SAMPLERS` including Cauchy).
