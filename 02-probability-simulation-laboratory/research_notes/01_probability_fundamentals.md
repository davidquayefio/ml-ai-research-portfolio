# Research note 1 — Probability fundamentals

## Definitions
A **probability space** $(\Omega,\mathcal F,P)$: sample space, σ-algebra of events, and a measure with the **Kolmogorov axioms** $P(A)\ge0$, $P(\Omega)=1$, countable additivity. Consequences: $P(A^c)=1-P(A)$, $P(A\cup B)=P(A)+P(B)-P(A\cap B)$, monotonicity.

**Conditional probability** $P(A\mid B)=P(A\cap B)/P(B)$ for $P(B)>0$. **Independence**: $P(A\cap B)=P(A)P(B)$. Events $A_1..A_n$ are *mutually* independent if the product rule holds for every sub-collection; *pairwise* independence is strictly weaker (Experiment 1.4 tests independence and constructs the standard example with two fair coins and the event "the coins agree").

## Theorems used
* **Law of total probability**: $P(A)=\sum_iP(A\mid B_i)P(B_i)$ for a partition $\{B_i\}$.
* **Bayes' theorem**: $P(B_j\mid A)=\dfrac{P(A\mid B_j)P(B_j)}{\sum_iP(A\mid B_i)P(B_i)}$.
* **Binomial law**: number of heads in $n$ independent Bernoulli($p$) trials is Bin$(n,p)$.

## Derivations worth doing by hand
1. *Base-rate fallacy.* Prevalence $\pi$, sensitivity $s$, specificity $c$: $P(D\mid +)=\dfrac{s\pi}{s\pi+(1-c)(1-\pi)}$. For small $\pi$ the posterior is approximately $s\pi/(1-c)\cdot1/(1-\pi)$ — dominated by the false-positive rate.
2. *Birthday problem.* $P(\text{no match among }n)=\prod_{k=0}^{n-1}(1-k/365)$; the smallest $n$ with match probability $\ge\tfrac12$ is $23$ (under the uniform-birthday assumption).
3. *Monty Hall.* With the host *always* opening a goat door that is not the contestant's, switching wins with probability $2/3$. The answer depends on the host's protocol — the simulation makes the protocol explicit.

## Assumptions and caveats
Fair coins, uniform birthdays, a host who knows where the car is: each is a modelling assumption. An empirical frequency converges to the probability only in the sense of the Law of Large Numbers (note 3); at finite $n$ it fluctuates by about $\sqrt{p(1-p)/n}$.

## Link to the code
`01_probability_fundamentals/src/fundamentals.py`; exact enumeration uses `fractions.Fraction` so that exact answers are exact rationals, not floats.
