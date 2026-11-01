# Research note 6 — Random walks, Markov chains, Poisson process

## Simple random walk
$S_n=\sum X_i$, $P(X_i=\pm1)=p,q$. $\mathbb ES_n=n(p-q)$, $\operatorname{Var}S_n=4npq$, $S_n=2\,\text{Bin}(n,p)-n$. Symmetric case is a **martingale**; so is $S_n^2-n$. In $\mathbb Z^2$, $\mathbb E|S_n|^2=n$ and $\mathbb E|S_n|\approx\sqrt{\pi n}/2$ (Rayleigh limit).

## Gambler's ruin
Fair: $P_i(\text{hit }N)=i/N$ (optional stopping on $S_n$), $\mathbb E_iT=i(N-i)$ (optional stopping on $S_n^2-n$). Biased, $r=q/p\ne1$: $P_i=\dfrac{1-r^i}{1-r^N}$ because $r^{S_n}$ is a martingale. The same numbers come from the **fundamental matrix** $F=(I-Q)^{-1}$ of the absorbing chain: $F\mathbf1$ = expected steps, $FR$ = absorption probabilities. Experiment 6.3 shows the three methods agree and that a 2% per-round disadvantage cuts a 50% chance to about 31% for $N=20,i=10$.

## Finite Markov chains
$P$ stochastic; $\mu_n=\mu_0P^n$. Irreducible + aperiodic ⇒ unique stationary $\pi$ with $\pi P=\pi$ and $\mu_n\to\pi$ geometrically at rate $|\lambda_2|$. **Ergodic theorem**: time averages converge to $\pi$. **Kac**: mean return time to $i$ is $1/\pi_i$. For the 3-state weather chain of Experiment 6.4, $\pi=(21,13,12)/46$ and $|\lambda_2|=0.4732$.

## Poisson process
Inter-arrival times i.i.d. Exp($\lambda$) ⇔ $N(t)\sim$ Poisson($\lambda t$) with independent stationary increments. Memorylessness $P(T>s+t\mid T>s)=P(T>t)$ is what makes it Markov.

## Caveats
All are *idealised* models: constant bet sizes, time-homogeneous transition probabilities, constant intensity. Real event data are over-dispersed and clustered.

## Link to the code
`06_stochastic_processes/src/stochastic.py`.
