# Module 7 — Brownian Motion: exercises

Work these **on paper first**, then check with the code in `07_brownian_motion/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Problems

1. Prove $\operatorname{Cov}(W_s,W_t)=\min(s,t)$ from independent increments. Show that $W_t^2-t$ and $\exp(\lambda W_t-\lambda^2t/2)$ are martingales.
2. Derive $\mathbb E|\Delta W|=\sqrt{2\Delta t/\pi}$ and $\operatorname{Var}(\Delta W^2)=2\Delta t^2$; deduce the mean and variance of the quadratic variation on an $n$-step grid.
3. Apply Itô's formula to $f(S)=\log S$ for $dS=\mu Sdt+\sigma SdW$ and obtain the GBM solution. Where does $-\tfrac12\sigma^2$ come from?
4. Prove the lognormal moment formulas using $\mathbb E e^{\sigma Z}=e^{\sigma^2/2}$. Compute the *mode* of $S_T$.
5. Show that $\mathbb ES^{\text{Euler}}_T=S_0(1+\mu T/n)^n$ and expand $(1+\mu T/n)^n-e^{\mu T}$ to first order in $1/n$ (weak error constant).
6. Derive the reflection principle via the strong Markov property at the hitting time $\tau_a$; deduce the density of $\tau_a$.
7. For $\mu=0.04,\ \sigma=0.3$ find the horizon $T$ beyond which more than $60\%$ of GBM paths end below $S_0$.
