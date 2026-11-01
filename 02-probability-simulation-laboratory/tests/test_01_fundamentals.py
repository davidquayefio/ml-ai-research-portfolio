from fractions import Fraction

import numpy as np
import pytest

import fundamentals as f
from plab import make_rng


def test_two_dice_sum_seven_is_one_sixth():
    omega = f.sample_space_dice(2)
    assert len(omega) == 36
    assert f.exact_probability(lambda w: sum(w) == 7, omega) == Fraction(1, 6)


def test_complement_rule_exact():
    omega = f.sample_space_dice(2)
    a = lambda w: sum(w) >= 10
    p = f.exact_probability(a, omega)
    assert p + f.exact_probability(lambda w: not a(w), omega) == 1


def test_inclusion_exclusion_exact():
    omega = f.sample_space_dice(2)
    a = lambda w: w[0] % 2 == 0
    b = lambda w: sum(w) > 8
    lhs = f.exact_probability(lambda w: a(w) or b(w), omega)
    rhs = f.exact_probability(a, omega) + f.exact_probability(b, omega) - f.exact_probability(lambda w: a(w) and b(w), omega)
    assert lhs == rhs


def test_conditional_probability_exact():
    omega = f.sample_space_dice(2)
    # P(sum=8 | first die even): even first die in {2,4,6} -> partners 6,4,2 => 3/18
    p = f.exact_conditional_probability(lambda w: sum(w) == 8, lambda w: w[0] % 2 == 0, omega)
    assert p == Fraction(1, 6)
    with pytest.raises(ValueError):
        f.exact_conditional_probability(lambda w: True, lambda w: False, omega)


def test_independence_sum7_and_even_first_die():
    omega = f.sample_space_dice(2)
    a = lambda w: w[0] % 2 == 0
    b = lambda w: sum(w) == 7
    pa, pb = f.exact_probability(a, omega), f.exact_probability(b, omega)
    assert f.exact_probability(lambda w: a(w) and b(w), omega) == pa * pb


def test_running_frequency_ends_at_mean():
    x = f.simulate_bernoulli(1000, 0.4, make_rng(1))
    assert f.running_frequency(x)[-1] == pytest.approx(x.mean())


def test_empirical_close_to_theory_within_envelope():
    n, p = 100_000, 0.5
    x = f.simulate_bernoulli(n, p, make_rng(2))
    assert abs(x.mean() - p) < 4 * np.sqrt(p * (1 - p) / n)


def test_prob_k_heads():
    assert f.prob_k_heads(10, 5) == pytest.approx(252 / 1024)


def test_bayes_posterior_value():
    # prevalence 1%, sensitivity 95%, specificity 90%
    assert f.bayes_posterior(0.01, 0.95, 0.90) == pytest.approx(0.0095 / (0.0095 + 0.099))


def test_screening_simulation_matches_bayes():
    d, pos = f.simulate_screening(2_000_000, 0.01, 0.95, 0.90, make_rng(3))
    est, n_b = f.estimate_conditional_probability(d, pos)
    se = np.sqrt(est * (1 - est) / n_b)
    assert abs(est - f.bayes_posterior(0.01, 0.95, 0.90)) < 4 * se


def test_birthday_exact_known_value():
    assert f.birthday_exact(23) == pytest.approx(0.5072972, abs=1e-6)
    assert f.birthday_exact(366) == 1.0


def test_birthday_simulation():
    p = f.birthday_exact(23)
    est = f.birthday_simulated(23, 100_000, make_rng(4))
    assert abs(est - p) < 4 * np.sqrt(p * (1 - p) / 100_000)


def test_monty_hall():
    n = 200_000
    stay = f.monty_hall(n, switch=False, rng=make_rng(5))
    swap = f.monty_hall(n, switch=True, rng=make_rng(6))
    se = np.sqrt((2 / 9) / n)
    assert abs(stay - 1 / 3) < 4 * se
    assert abs(swap - 2 / 3) < 4 * se
