"""Module 1: probability fundamentals.

Exact (enumeration) and simulated probabilities for coins, dice, conditional
probability, independence, Bayes' theorem and combinatorial problems.

Mathematical dictionary -> code
-------------------------------
* sample space  Omega                -> a Python list of outcomes (tuples)
* event A (subset of Omega)          -> a boolean function ``event(outcome)``
* P(A) = |A| / |Omega|               -> :func:`exact_probability`
* relative frequency n_A / n         -> mean of a boolean/0-1 array
* P(A | B) = P(A and B) / P(B)       -> :func:`exact_conditional_probability`
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import comb
from typing import Callable, Iterable, Sequence

import numpy as np

Outcome = tuple


# ----------------------------------------------------------------------------
# Exact probability by enumeration
# ----------------------------------------------------------------------------
def sample_space_dice(n_dice: int = 2, sides: int = 6) -> list[Outcome]:
    """Sample space for ``n_dice`` distinguishable dice: ``sides**n_dice`` ordered tuples."""
    return list(product(range(1, sides + 1), repeat=n_dice))


def exact_probability(event: Callable[[Outcome], bool], omega: Sequence[Outcome]) -> Fraction:
    """Classical probability ``|A| / |Omega|`` (valid only if outcomes are equally likely)."""
    favourable = sum(1 for w in omega if event(w))
    return Fraction(favourable, len(omega))


def exact_conditional_probability(
    event_a: Callable[[Outcome], bool],
    event_b: Callable[[Outcome], bool],
    omega: Sequence[Outcome],
) -> Fraction:
    """``P(A | B) = P(A and B) / P(B)``; raises if ``P(B) = 0``."""
    p_b = exact_probability(event_b, omega)
    if p_b == 0:
        raise ValueError("P(B) = 0: conditional probability is undefined")
    p_ab = exact_probability(lambda w: event_a(w) and event_b(w), omega)
    return p_ab / p_b


# ----------------------------------------------------------------------------
# Bernoulli trials and relative frequencies
# ----------------------------------------------------------------------------
def simulate_bernoulli(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` independent Bernoulli(p) trials as an int8 array of 0/1.

    ``U ~ Uniform[0,1)`` satisfies ``P(U < p) = p``, so ``1{U < p}`` is Bernoulli(p).
    """
    return (rng.random(n) < p).astype(np.int8)


def running_frequency(indicator: np.ndarray) -> np.ndarray:
    """Relative frequency after each trial: ``f_k = (x_1 + ... + x_k) / k``."""
    x = np.asarray(indicator, dtype=float)
    return np.cumsum(x) / np.arange(1, x.size + 1)


def frequency_envelope(n_grid: np.ndarray, p: float, z: float = 3.0) -> np.ndarray:
    """Half-width ``z * sqrt(p (1-p) / n)``: where ``f_n`` should lie ~99.7% of the time (z=3)."""
    n_grid = np.asarray(n_grid, dtype=float)
    return z * np.sqrt(p * (1.0 - p) / n_grid)


def prob_k_heads(n: int, k: int, p: float = 0.5) -> float:
    """Binomial probability ``C(n,k) p^k (1-p)^(n-k)`` (combinatorial probability)."""
    return comb(n, k) * p**k * (1.0 - p) ** (n - k)


# ----------------------------------------------------------------------------
# Dice, conditional probability, independence
# ----------------------------------------------------------------------------
def roll_dice(n_rolls: int, n_dice: int, rng: np.random.Generator, sides: int = 6) -> np.ndarray:
    """``(n_rolls, n_dice)`` array of fair-die outcomes in ``{1, ..., sides}``."""
    return rng.integers(1, sides + 1, size=(n_rolls, n_dice))


def empirical_pmf(values: np.ndarray, support: Iterable[int]) -> np.ndarray:
    """Relative frequency of each value in ``support``."""
    values = np.asarray(values)
    return np.array([(values == s).mean() for s in support])


def estimate_conditional_probability(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Estimate ``P(A | B)`` as the frequency of A among trials where B occurred.

    Returns ``(estimate, n_b)`` where ``n_b`` is the effective sample size.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n_b = int(b.sum())
    if n_b == 0:
        raise ValueError("event B never occurred")
    return float(a[b].mean()), n_b


# ----------------------------------------------------------------------------
# Bayes' theorem
# ----------------------------------------------------------------------------
def bayes_posterior(prior: float, sensitivity: float, specificity: float) -> float:
    """``P(D | +) = P(+ | D) P(D) / P(+)`` with the law of total probability in the denominator.

    ``sensitivity = P(+ | D)``; ``specificity = P(- | not D)``.
    """
    p_pos = sensitivity * prior + (1.0 - specificity) * (1.0 - prior)
    return sensitivity * prior / p_pos


def simulate_screening(
    n: int, prevalence: float, sensitivity: float, specificity: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate disease status and test result for ``n`` people. Returns ``(disease, positive)``."""
    disease = rng.random(n) < prevalence
    u = rng.random(n)
    positive = np.where(disease, u < sensitivity, u < 1.0 - specificity)
    return disease, positive


# ----------------------------------------------------------------------------
# Combinatorial probability: birthday problem and Monty Hall
# ----------------------------------------------------------------------------
def birthday_exact(n: int, days: int = 365) -> float:
    """P(at least two of ``n`` people share a birthday) = ``1 - prod_{i<n} (days - i)/days``."""
    if n > days:
        return 1.0
    p_all_distinct = 1.0
    for i in range(n):
        p_all_distinct *= (days - i) / days
    return 1.0 - p_all_distinct


def birthday_simulated(n: int, trials: int, rng: np.random.Generator, days: int = 365) -> float:
    """Monte Carlo estimate of the birthday-collision probability."""
    b = rng.integers(0, days, size=(trials, n))
    b.sort(axis=1)
    return float(np.mean((np.diff(b, axis=1) == 0).any(axis=1)))


def monty_hall(n: int, switch: bool, rng: np.random.Generator) -> float:
    """Win frequency in the Monty Hall game, simulating the host's behaviour explicitly.

    The host opens a door that is neither the contestant's pick nor the prize
    (choosing uniformly if two doors qualify).
    """
    pick = rng.integers(0, 3, size=n)
    prize = rng.integers(0, 3, size=n)
    r = rng.random((n, 3))
    rows = np.arange(n)
    r[rows, pick] = -1.0
    r[rows, prize] = -1.0
    host = np.argmax(r, axis=1)  # a valid goat door not chosen by the contestant
    final = 3 - pick - host if switch else pick
    return float(np.mean(final == prize))
