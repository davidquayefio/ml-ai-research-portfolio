# Research note 7 — Brownian motion and GBM

## Definition and basic properties
Standard BM: $W_0=0$, independent increments, $W_t-W_s\sim N(0,t-s)$, continuous paths. Then $\operatorname{Cov}(W_s,W_t)=\min(s,t)$ and $W_{ct}\overset d=\sqrt cW_t$; $W_t$, $W_t^2-t$ and $e^{\lambda W_t-\lambda^2t/2}$ are martingales.

## Donsker
The rescaled walk $S_{\lfloor nt\rfloor}/\sqrt n\Rightarrow W$ in $C[0,1]$ (Billingsley). Fixed-time version = CLT; the exact Kolmogorov distance of the $\pm1$ walk at $t=1$ satisfies $D_n\sqrt n\to1/\sqrt{2\pi}$ (Experiment 7.1).

## Quadratic variation
$\sum(\Delta W)^2\to T$ with mean $T$ and variance $2T^2/n$; $\mathbb E\sum|\Delta W|=\sqrt{2nT/\pi}\to\infty$. Hence paths are of infinite variation and nowhere differentiable, and $(dW)^2=dt$ in Itô calculus.

## Itô's formula and GBM
For $f\in C^{1,2}$: $df=f_tdt+f_xdW+\tfrac12f_{xx}dt$. With $f=\ln S$ and $dS=\mu Sdt+\sigma SdW$: $d\ln S=(\mu-\tfrac12\sigma^2)dt+\sigma dW$, so $S_t=S_0e^{(\mu-\sigma^2/2)t+\sigma W_t}$. $\mathbb ES_t=S_0e^{\mu t}$ but the typical path grows at $\mu-\sigma^2/2$ — which can be negative while the mean explodes (Experiment 7.4: with $\mu=0.04,\sigma=0.3$, $T=400$ about 63% of paths end below $S_0$).

## Numerics
Euler–Maruyama: strong order $1/2$, weak order $1$; measured slopes $-0.49$ and $-1.00$ (Experiment 7.4). Milstein improves strong order to $1$. Path-dependent functionals sampled on a grid carry a bias of order $\sqrt{\Delta t}$: for the maximum, shifting the barrier by $0.5826\sqrt{\Delta t}$ (Broadie–Glasserman–Kou) explains the gap in Experiment 7.5.

## Link to the code
`07_brownian_motion/src/brownian.py`, `src/plab/processes.py`.
