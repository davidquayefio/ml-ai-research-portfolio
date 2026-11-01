# Module 6 — Stochastic Processes: exercises

Work these **on paper first**, then check with the code in `06_stochastic_processes/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Problems

1. Show $\operatorname{Var}S_n=4npq$ for the biased walk and prove that $M_n=(q/p)^{S_n}$ is a martingale. Use optional stopping on $M_n$ to derive the gambler's-ruin probability $\frac{1-r^i}{1-r^N}$.
2. For the symmetric walk show $S_n^2-n$ is a martingale and derive $\mathbb E_i[T]=i(N-i)$.
3. For the weather chain compute $P^2$ by hand and verify $\pi P=\pi$ for $\pi=(21,13,12)/46$.
4. Show that a Markov chain with $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ is irreducible but periodic. Does $\mu_n$ converge? Does the time average converge?
5. Using the memoryless property of the exponential distribution, show that the exponential-gap construction gives independent Poisson increments (sketch).
6. Why is the *simple random walk* a martingale but not a good model for a stock *price* (what goes wrong with positivity and scale)? How do the exponential of a walk and Module 7's geometric Brownian motion fix this?
7. Classify the states of the chain with $P=\begin{pmatrix}1&0&0\\0.5&0&0.5\\0&0&1\end{pmatrix}$, find the absorption probabilities and expected time to absorption from state 2 by hand and by `absorbing_chain_analysis`.
