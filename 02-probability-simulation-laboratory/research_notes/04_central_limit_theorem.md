# Research note 4 — Central Limit Theorem

## Statement
For i.i.d. $X_i$ with mean $\mu$ and finite variance $\sigma^2>0$:
$$\sqrt n\,\frac{\bar X_n-\mu}{\sigma}\xrightarrow{d}N(0,1).$$

## Proof sketch (characteristic functions)
Let $Y_i=(X_i-\mu)/\sigma$, $\varphi(t)=\mathbb Ee^{itY}=1-t^2/2+o(t^2)$. Then $\mathbb Ee^{it\sum Y_i/\sqrt n}=\varphi(t/\sqrt n)^n=(1-t^2/2n+o(1/n))^n\to e^{-t^2/2}$, and Lévy's continuity theorem gives the result.

## Quantitative refinements
* **Berry–Esseen**: $\sup_x|F_n(x)-\Phi(x)|\le C\rho/(\sigma^3\sqrt n)$ with $\rho=\mathbb E|X-\mu|^3$ and an absolute constant $C<1$.
* **Edgeworth**: $F_n(x)\approx\Phi(x)-\dfrac{\gamma}{6\sqrt n}(x^2-1)\varphi(x)$, $\gamma$ = skewness — the leading error is *skewness divided by $\sqrt n$*. Hence skewed laws need larger $n$ (Experiment 4.2).
* **Lattice effect**: for discrete laws the exact cdf has jumps of size $\sim1/\sqrt n$, so the Kolmogorov distance cannot fall faster than that (Experiments 4.1–4.2 and Module 7.1, where $D_n\sqrt n\to1/\sqrt{2\pi}$).

## When the CLT fails
* **Infinite variance**: normalised sums converge to *stable* laws with scaling $n^{1/\alpha}$, not $\sqrt n$ (Experiment 4.3: IQR slopes).
* **Strong dependence** or non-identical heavy tails need other versions (Lindeberg–Feller, mixing CLTs).
* The *t-interval* $\bar X\pm t_{n-1}s/\sqrt n$ is exact only for normal data; its actual coverage under skewness is measured in Experiment 4.4.

## Link to the code
`04_central_limit_theorem/src/clt.py`.
