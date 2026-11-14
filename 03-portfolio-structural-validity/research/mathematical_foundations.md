# Mathematical foundations

## 1. Notation and portfolio returns

There are \(N\) assets and \(T\) observations. At time \(t\), let
\[
r_t=(r_{1t},\ldots,r_{Nt})^\top\in\mathbb R^N
\]
be the vector of simple returns. Let \(\mu=\mathbb E[r_t]\) and
\(\Sigma=\operatorname{Cov}(r_t)\). A fully invested portfolio has weights
\(w\in\mathbb R^N\) satisfying \(\mathbf 1^\top w=1\). Unless stated
otherwise, returns and moments are measured at the same periodic frequency.

For weights held over one return interval,
\[
R_{p,t}=w^\top r_t,\qquad
\mathbb E[R_{p,t}]=w^\top\mu,\qquad
\operatorname{Var}(R_{p,t})=w^\top\Sigma w.
\]
The last equality follows from
\(\operatorname{Var}(w^\top r_t)=w^\top\operatorname{Cov}(r_t)w\).
If \(\Sigma\) is positive definite, every nonzero portfolio has positive
variance; if it is only positive semidefinite, zero-variance directions may
exist.

For simple returns, portfolio aggregation is exact for weights established at
the start of the interval. Across multiple periods, compounded portfolio
returns are not generally the weighted sum of compounded asset returns.

## 2. Mean–variance portfolios

### 2.1 Global minimum variance

The unconstrained global minimum-variance (GMV) problem is
\[
\begin{aligned}
\min_w\quad & \frac12 w^\top\Sigma w\\
\text{subject to}\quad & \mathbf 1^\top w=1.
\end{aligned}
\]
Its Lagrangian and first-order condition are
\[
\mathcal L(w,\lambda)=\frac12w^\top\Sigma w
-\lambda(\mathbf 1^\top w-1),\qquad
\nabla_w\mathcal L=\Sigma w-\lambda\mathbf 1=0.
\]
Thus \(w=\lambda\Sigma^{-1}\mathbf 1\); imposing the budget constraint gives
\[
\boxed{
w_{\rm GMV}=
\frac{\Sigma^{-1}\mathbf 1}{\mathbf 1^\top\Sigma^{-1}\mathbf 1}
},\qquad
\boxed{
\sigma^2_{\rm GMV}=
\frac{1}{\mathbf 1^\top\Sigma^{-1}\mathbf 1}
}.
\]
Positive definiteness makes the objective strictly convex, so this is the
unique global minimum. Short positions are permitted. With long-only
constraints, the problem becomes a convex quadratic programme:
\[
\min_w w^\top\Sigma w
\quad\text{subject to}\quad
\mathbf 1^\top w=1,\quad w_i\ge 0.
\]
Its solution generally has no single closed form; the active constraints
determine which assets receive zero weight.

### 2.2 Efficient frontier

For a target expected return \(m\), impose \(w^\top\mu=m\) as a second
equality constraint. Define
\[
A=\mathbf 1^\top\Sigma^{-1}\mu,\qquad
B=\mu^\top\Sigma^{-1}\mu,\qquad
C=\mathbf 1^\top\Sigma^{-1}\mathbf 1,\qquad
D=BC-A^2.
\]
When \(\mu\) is not proportional to \(\mathbf 1\), \(D>0\). The minimum
variance portfolio with expected return \(m\) and its variance are
\[
\boxed{
w(m)=\frac{\Sigma^{-1}\big[(B-Am)\mathbf 1+(Cm-A)\mu\big]}{D}
},\qquad
\boxed{
\sigma^2(m)=\frac{Cm^2-2Am+B}{D}
}.
\]
The variance is minimized at \(m=A/C\), the expected return of the GMV
portfolio. Since \(w(m)\) is affine in \(m\), every portfolio on this
unconstrained frontier is an affine combination of any two distinct frontier
portfolios.

### 2.3 Tangency and maximum Sharpe

Let \(r_f\) be the risk-free return per period and
\(\tilde\mu=\mu-r_f\mathbf 1\) the vector of expected excess returns. The
Sharpe ratio of \(w\) is
\[
S(w)=\frac{w^\top\tilde\mu}{\sqrt{w^\top\Sigma w}}.
\]
With unrestricted borrowing and short selling, a tangency direction is
\[
w_T\ \propto\ \Sigma^{-1}\tilde\mu.
\]
If \(\mathbf 1^\top\Sigma^{-1}\tilde\mu\ne0\), it can be normalized to unit
investment by dividing by that sum. Long-only maximum-Sharpe portfolios impose
\(w_i\ge0\), so this closed form need not apply. If all feasible portfolios
have nonpositive expected excess return, the ratio optimization does not
identify a meaningful positive-Sharpe tangency portfolio.

Multiplying all excess means by a positive constant changes the attainable
Sharpe ratios by that constant, but not the unrestricted tangency direction.
This invariance does not protect the direction from errors that change the
relative asset means.

### 2.4 Risk contributions and equal risk contribution

For portfolio volatility
\(\sigma_p=\sqrt{w^\top\Sigma w}>0\), Euler's identity gives
\[
\sigma_p=\sum_{i=1}^{N}RC_i,\qquad
RC_i=\frac{w_i(\Sigma w)_i}{\sigma_p}.
\]
Equal risk contribution (ERC) chooses weights such that
\[
RC_i=\frac{\sigma_p}{N}\quad(i=1,\ldots,N).
\]
Equivalently, because the denominator is common to all assets,
\(w_i(\Sigma w)_i\) is equal across \(i\). For uncorrelated assets with
volatilities \(\sigma_i\), the long-only ERC weights are
\[
w_i=\frac{\sigma_i^{-1}}{\sum_j\sigma_j^{-1}}.
\]
In the general correlated case, ERC is a nonlinear system and is solved
numerically.

## 3. Mean uncertainty and Bayesian views

### 3.1 Black–Litterman

Let \(w_{\rm mkt}\) be reference portfolio weights and \(\delta>0\) the
risk-aversion parameter. Reverse optimization sets the prior mean to
\[
\pi=\delta\Sigma w_{\rm mkt}.
\]
The prior and views are modeled as
\[
\mu\sim\mathcal N(\pi,\tau\Sigma),\qquad
Q=P\mu+\varepsilon,\qquad
\varepsilon\sim\mathcal N(0,\Omega),
\]
where \(P\) encodes the views, \(Q\) their stated values, and \(\Omega\) their
error covariance. Gaussian conditioning yields
\[
\boxed{
\mu_{\rm BL}=
\left[(\tau\Sigma)^{-1}+P^\top\Omega^{-1}P\right]^{-1}
\left[(\tau\Sigma)^{-1}\pi+P^\top\Omega^{-1}Q\right]
}.
\]
With no views, \(\mu_{\rm BL}=\pi\). Unconstrained mean–variance optimization
with objective
\[
\max_w\quad w^\top\mu_{\rm BL}
-\frac{\delta}{2}w^\top\Sigma w,\qquad \mathbf 1^\top w=1
\]
then returns \(w_{\rm mkt}\). In this project the reference weights default
to equal weights rather than market-cap weights. The implemented model also
uses sample-mean views by default; this is a project-specific design choice.

### 3.2 Ellipsoidal robust mean–variance

Suppose the unknown mean belongs to the uncertainty set
\[
\mathcal U=
\left\{m:(m-\hat\mu)^\top\Omega^{-1}(m-\hat\mu)\le\kappa^2\right\},
\]
where \(\hat\mu\) is the estimated mean and \(\Omega\) describes its
uncertainty. For any portfolio \(w\),
\[
\min_{m\in\mathcal U}w^\top m
=w^\top\hat\mu-\kappa\sqrt{w^\top\Omega w}.
\]
This follows by writing \(m=\hat\mu+\Omega^{1/2}u\), with
\(\lVert u\rVert_2\le\kappa\), and minimizing the resulting inner product.
The robust mean–variance objective is therefore
\[
\max_{w\in\mathcal W}\quad
w^\top\hat\mu-\kappa\sqrt{w^\top\Omega w}
-\frac{\gamma}{2}w^\top\hat\Sigma w.
\]
Here \(\mathcal W\) is the feasible weight set, \(\gamma>0\) is risk aversion,
and \(\hat\Sigma\) is the estimated return covariance. The uncertainty penalty
is portfolio-specific; it is large when the portfolio loads on uncertain
directions of the mean.

## 4. Estimating moments

Given observations \(r_1,\ldots,r_T\), the sample mean and unbiased sample
covariance are
\[
\bar r=\frac1T\sum_{t=1}^{T}r_t,\qquad
S=\frac1{T-1}\sum_{t=1}^{T}(r_t-\bar r)(r_t-\bar r)^\top.
\]
Under independent observations with common covariance \(\Sigma\),
\(\mathbb E[S]=\Sigma\), and
\(\operatorname{Cov}(\bar r)=\Sigma/T\). Serial dependence changes the latter:
the covariance of the sample mean depends on autocovariances at nonzero lags.

If returns are iid multivariate normal,
\[
(T-1)S\sim\mathcal W_N(\Sigma,T-1),
\]
where \(\mathcal W_N\) denotes the \(N\)-dimensional Wishart distribution.
Consequently, \(S\) is singular when \(T\le N\), and for \(T>N+2\),
\[
\mathbb E[S^{-1}]
=\frac{T-1}{T-N-2}\Sigma^{-1}.
\]
Thus even an unbiased covariance estimator does not produce an unbiased
inverse-covariance input.

For a normally distributed sample mean, the standard error of asset \(i\)'s
estimated mean is
\[
\operatorname{SE}(\bar r_i)=\frac{\sigma_i}{\sqrt T}.
\]
Mean estimates can therefore be noisy relative to the small differences in
expected returns that often determine an optimizer's weights.

### 4.1 Covariance shrinkage

Linear shrinkage combines a sample covariance \(S\) with a structured target
\(F\):
\[
\hat\Sigma_\alpha=(1-\alpha)S+\alpha F,\qquad 0\le\alpha\le1.
\]
The target may reduce estimation variance while introducing bias. The
mean-squared Frobenius error is
\[
\mathbb E\lVert\hat\Sigma_\alpha-\Sigma\rVert_F^2.
\]
Choosing \(\alpha\) balances the variance of \(S\) against the discrepancy
between \(F\) and \(\Sigma\). The Ledoit–Wolf estimator estimates a shrinkage
intensity for a scaled-identity target.

### 4.2 Random-matrix benchmark

For an iid \(T\times N\) data matrix with zero mean and unit variance, as
\(N,T\to\infty\) with \(q=N/T\in(0,1)\), the eigenvalues of the sample
covariance converge to a distribution supported on
\[
\left[(1-\sqrt q)^2,\ (1+\sqrt q)^2\right].
\]
This Marchenko–Pastur interval is a noise benchmark under its assumptions, not
a universal cutoff for financial data. Dependence, volatility clustering,
heavy tails, and finite-sample effects can alter the spectrum.

## 5. Time-varying covariance and estimation windows

### 5.1 EWMA covariance

An exponentially weighted covariance assigns more weight to recent
observations. For demeaned returns \(x_t=r_t-\bar r\), one finite-sample form
is
\[
\hat\Sigma_{\rm EWMA}
=\sum_{t=1}^{T}\omega_t x_tx_t^\top,\qquad
\omega_t=\frac{(1-\lambda)\lambda^{T-t}}
{1-\lambda^T},\qquad 0<\lambda<1.
\]
The weights sum to one. A larger \(\lambda\) gives older observations more
persistence. The implementation uses this normalized finite-window form with
\(\lambda=0.94\); it centers observations by the window sample mean.

In the univariate zero-mean recursion,
\[
\sigma_t^2=\lambda\sigma_{t-1}^2+(1-\lambda)r_{t-1}^2,
\]
EWMA is the integrated GARCH(1,1) boundary case: the intercept is zero and
the shock and lagged-variance coefficients sum to one.

### 5.2 ARCH and GARCH

An ARCH(\(q\)) model and a GARCH(1,1) model are
\[
r_t=\sigma_t z_t,\qquad
\sigma_t^2=\omega+\sum_{i=1}^{q}\alpha_i r_{t-i}^2,
\]
\[
\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta\sigma_{t-1}^2.
\]
For GARCH(1,1), if
\(\omega>0\), \(\alpha,\beta\ge0\), and \(\alpha+\beta<1\), the unconditional
variance is
\[
\mathbb E[\sigma_t^2]=\frac{\omega}{1-\alpha-\beta}.
\]
The condition \(\alpha+\beta<1\) ensures finite unconditional variance under
the standard model assumptions. Persistent conditional variance can produce
volatility clustering even when the innovations \(z_t\) are independent.

### 5.3 Rolling and expanding windows

A rolling covariance estimate based on \(W\) observations is
\[
S_t^{(W)}=\frac1{W-1}\sum_{j=t-W+1}^{t}
(r_j-\bar r_t)(r_j-\bar r_t)^\top.
\]
An expanding estimate replaces the fixed left endpoint with the beginning of
the available sample. A shorter rolling window adapts more quickly to change
but uses fewer observations; an expanding window uses more observations but
retains older data when the underlying moments change.

## 6. Numerical conditioning and optimizer sensitivity

For a positive-definite covariance matrix,
\[
\operatorname{cond}_2(\Sigma)
=\frac{\lambda_{\max}(\Sigma)}{\lambda_{\min}(\Sigma)}.
\]
A large condition number means some directions have much lower variance than
others. Inversion magnifies errors in those directions.

For the unconstrained GMV portfolio, a first-order perturbation of \(\Sigma\)
gives
\[
d w_{\rm GMV}
=-(I-w_{\rm GMV}\mathbf 1^\top)\,
\Sigma^{-1}(d\Sigma)\,w_{\rm GMV}.
\]
This differential follows by differentiating
\(\Sigma^{-1}\mathbf 1/
(\mathbf 1^\top\Sigma^{-1}\mathbf 1)\). It makes explicit how covariance
perturbations can move weights, especially when \(\Sigma\) is poorly
conditioned. A large weight change need not imply a similarly large change in
predicted variance: the objective can be relatively flat along low-variance
directions.

Long-only minimum variance is a convex quadratic programme because
\(\Sigma\succeq0\) and its constraints are affine. A target-return constraint
preserves convexity. By contrast, the Sharpe ratio is fractional; with
positive excess means and long-only weights, the substitution
\[
y=\frac{w}{w^\top\tilde\mu}
\]
gives the equivalent formulation
\[
\min_{y\ge0}\ y^\top\Sigma y
\quad\text{subject to}\quad
y^\top\tilde\mu=1,
\qquad w=\frac{y}{\mathbf 1^\top y}.
\]

## 7. Structural-stability quantities

These definitions specify the quantities compared in the project; lower
values indicate less instability or fragility.

For portfolio weights immediately before and after a rebalance,
\[
\operatorname{TO}_t
=\frac12\sum_{i=1}^{N}|w_{i,t}^{+}-w_{i,t}^{-}|.
\]
The factor \(1/2\) makes turnover the fraction of portfolio value traded on
one side. When weights drift between rebalances, \(w_{i,t}^{-}\) denotes the
pre-trade drifted weights, not simply the previous target weights.

For an estimated weight vector \(\hat w\) and bootstrap estimates
\(\hat w^{(1)},\ldots,\hat w^{(B)}\), bootstrap weight sensitivity can be
summarized as
\[
\frac1B\sum_{b=1}^{B}
\frac12\sum_{i=1}^{N}|\hat w_i^{(b)}-\hat w_i|.
\]
The bootstrap scheme should preserve the serial dependence relevant to the
data. The stationary bootstrap resamples blocks with geometrically distributed
lengths; its mean block length is a tuning parameter.

For an out-of-sample period, the Sharpe gap is
\[
\Delta S=S_{\rm ex\ ante}-S_{\rm realized,OOS}.
\]
For regimes \(k=1,\ldots,K\), regime inconsistency is summarized by
\[
\operatorname{SD}(S_1,\ldots,S_K).
\]
Under a controlled input perturbation \(\delta\theta\), weight fragility can
be measured by
\[
\frac12\lVert w(\theta+\delta\theta)-w(\theta)\rVert_1.
\]
These are descriptive statistics, not proofs that a model is valid or
invalid. Their interpretation depends on the estimation window, rebalance
schedule, bootstrap design, regime definition, and perturbation scale.

## 8. Exercises

1. Derive the efficient-frontier weights from the two equality constraints.
2. For two assets, derive the GMV weight as a function of volatilities and
   correlation. Find when the weight on the more volatile asset is negative.
3. Show that multiplying \(\tilde\mu\) by a positive constant leaves the
   unrestricted tangency direction unchanged.
4. Explain why \(S\) is singular for \(T\le N\), and why the plug-in GMV
   formula then fails.
5. Derive the worst-case mean over the ellipsoid in Section 3.2.
6. Derive the GARCH(1,1) unconditional variance from its recursion.
7. Verify algebraically that the Black–Litterman no-view solution followed
   by unconstrained mean–variance optimization returns \(w_{\rm mkt}\).

## References

The results above are standard in portfolio theory, multivariate statistics,
and financial econometrics. Relevant works include Merton (1972), Ledoit and
Wolf (2004), Marchenko and Pastur (1967), and Politis and Romano (1994).
Verify bibliographic details and the applicability of each result before
using these references in a paper.
