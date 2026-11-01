# Research note 2 — Random variables and distributions

## Definitions
A **random variable** is a measurable map $X:\Omega\to\mathbb R$. Its cdf $F(x)=P(X\le x)$ is non-decreasing, right-continuous, with limits $0$ and $1$. Discrete: pmf; continuous: pdf $f=F'$. **Moments** $\mathbb EX^k$, mean, variance, skewness, kurtosis. **MGF** $M(t)=\mathbb Ee^{tX}$ (when finite near $0$): $M^{(k)}(0)=\mathbb EX^k$.

## The eleven distributions used
Bernoulli, binomial, geometric, Poisson, uniform, exponential, normal, gamma, beta, Student-$t$, chi-square — all via frozen `scipy.stats` objects; theoretical moments are from scipy and cross-checked against closed forms in the tests.

## Theorems used
* **Inverse-transform sampling**: if $U\sim U(0,1)$ then $F^{-1}(U)\sim F$. *Proof:* $P(F^{-1}(U)\le x)=P(U\le F(x))=F(x)$.
* **Probability integral transform**: if $X$ has continuous cdf $F$ then $F(X)\sim U(0,1)$.
* **Exponential** inverse: $-\ln(1-U)/\lambda$.
* **Correlated normals**: if $Z\sim N(0,I)$ and $\Sigma=LL^\top$ (Cholesky) then $LZ\sim N(0,\Sigma)$.
* **DKW inequality**: $P(\sup_x|F_n-F|>\varepsilon)\le2e^{-2n\varepsilon^2}$ — the empirical cdf converges uniformly at rate $n^{-1/2}$.

## Key cautions
* **Uncorrelated $\ne$ independent.** For $X\sim N(0,1)$, $\operatorname{Cov}(X,X^2)=\mathbb EX^3=0$ yet $X^2$ is a function of $X$ (Experiment 2.5).
* Heavy tails: for Student-$t_\nu$ the moments of order $\ge\nu$ do not exist; sample moments of such data are unstable.
* The MGF may not exist (Student-$t$, lognormal); characteristic functions always do.

## Link to the code
`02_probability_distributions/src/distributions.py`. Tests compare sample moments with theory at 5 standard errors, so they should not fail by chance.
