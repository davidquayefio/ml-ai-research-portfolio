# Module 1 — Probability Fundamentals: exercises

Work these **on paper first**, then check with the code in `01_probability_fundamentals/src/`. Solutions are intentionally not provided; discuss them with a study partner or compare with the textbooks in `research_notes/references.md`.

## Exercises (attempt on paper first — no solutions are given here)

1. For three coin tosses write $\Omega$, list the event "exactly two heads", and compute its probability. Why is "number of heads" *not* a valid sample space with equally likely outcomes?
2. Prove from the axioms: (a) $P(\emptyset)=0$; (b) $A\subseteq B\Rightarrow P(A)\le P(B)$; (c) $P(A\cup B)=P(A)+P(B)-P(A\cap B)$.
3. A fair die is rolled 60 times and 4 sixes appear. What are the expected count, the standard deviation of the count, and the z-score? Does this contradict $P(\text{six})=1/6$?
4. Show that if $A$ and $B$ are independent then so are $A$ and $B^c$.
5. Give your own example (not in this notebook) of events that are pairwise independent but not mutually independent.
6. In Experiment 1.5 compute the posterior if the test is repeated and is positive twice (assume conditional independence given disease status). Then check by simulation.
7. A stock is "up", "down" or "flat" tomorrow. Someone assigns probability $1/3$ to each. Which axioms hold? What assumption is doing the real work? How would you *estimate* these probabilities from data, and what could go wrong?
8. (Challenge) Show that the probability that $n$ people have no shared birthday is at most $e^{-n(n-1)/(2\cdot365)}$, using $1-x\le e^{-x}$.
