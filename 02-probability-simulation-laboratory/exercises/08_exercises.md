# Module 8 — Quantitative Finance: exercises

Work these **on paper first**, then check with the code in `08_quantitative_finance/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Problems

1. Show that if $r\sim\mathcal N(m,s^2)$ then $\mathbb E[e^r-1]=e^{m+s^2/2}-1$ and $\operatorname{Skew}(e^r)=(e^{s^2}+2)\sqrt{e^{s^2}-1}$.
2. Derive the Black–Scholes call price as $e^{-rT}\mathbb E^{\mathbb Q}(S_T-K)^+$ using $S_T=S_0e^{(r-\sigma^2/2)T+\sigma\sqrt TZ}$ and $\mathbb E[e^{aZ}\mathbf 1_{Z>b}]=e^{a^2/2}\bar\Phi(b-a)$.
3. Prove put–call parity by a no-arbitrage argument and by $\mathbb E^{\mathbb Q}[S_T-K]$.
4. For two antithetic samples with correlation $\rho$ between $f(Z)$ and $f(-Z)$ show $\operatorname{Var}\big(\tfrac12[f(Z)+f(-Z)]\big)=\tfrac12(1+\rho)\operatorname{Var}f(Z)$. When is antithetic sampling harmful?
5. Compute $99\%$ VaR and ES of $\mathcal N(0,1)$ losses by hand: $2.326$ and $\varphi(2.326)/0.01=2.665$.
6. Give an example (two assets) where VaR is not sub-additive.
7. Derive $\mathbb Er_t^2=\omega/(1-\alpha-\beta)$ for GARCH(1,1), and explain why $\operatorname{corr}(r_t,r_{t-k})=0$ while $\operatorname{corr}(r_t^2,r_{t-k}^2)>0$.
