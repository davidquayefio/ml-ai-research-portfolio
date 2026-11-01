import numpy as np
import pytest

import stochastic as sp
from plab import make_rng


def test_random_walk_mean_zero_variance_n():
    paths = sp.simple_random_walk(100, 100_000, make_rng(61))
    s100 = paths[:, -1]
    assert abs(s100.mean()) < 5 * 10 / np.sqrt(100_000)
    assert s100.var() == pytest.approx(100, rel=0.03)
    assert np.all(paths[:, 0] == 0)
    assert np.all(np.abs(np.diff(paths, axis=1)) == 1)


def test_walk_parity():
    paths = sp.simple_random_walk(51, 1000, make_rng(62))
    assert np.all(paths[:, -1] % 2 == 1)  # S_n has the parity of n


def test_2d_walk_mean_squared_distance_is_n():
    x, y = sp.lattice_walk_2d(200, 50_000, make_rng(63))
    r2 = x[:, -1] ** 2 + y[:, -1] ** 2
    assert r2.mean() == pytest.approx(200, rel=0.03)


def test_gamblers_ruin_closed_forms_agree_with_matrix_method():
    for p in (0.5, 0.48, 0.55):
        N = 20
        P = sp.gamblers_ruin_matrix(N, p)
        transient, steps, absorb = sp.absorbing_chain_analysis(P, [0, N])
        for idx, i in enumerate(transient):
            assert absorb[idx, 1] == pytest.approx(sp.gamblers_ruin_win_probability(i, N, p))
            assert steps[idx] == pytest.approx(sp.gamblers_ruin_expected_duration(i, N, p))


def test_gamblers_ruin_fair_known_values():
    assert sp.gamblers_ruin_win_probability(5, 20, 0.5) == 0.25
    assert sp.gamblers_ruin_expected_duration(5, 20, 0.5) == 75


def test_gamblers_ruin_simulation_matches_theory():
    n = 40_000
    won, dur = sp.gamblers_ruin_simulate(10, 20, 0.48, n, make_rng(64))
    p_theory = sp.gamblers_ruin_win_probability(10, 20, 0.48)
    assert abs(won.mean() - p_theory) < 4 * np.sqrt(p_theory * (1 - p_theory) / n)
    t_theory = sp.gamblers_ruin_expected_duration(10, 20, 0.48)
    assert abs(dur.mean() - t_theory) < 4 * dur.std(ddof=1) / np.sqrt(n)


P_WEATHER = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])


def test_stationary_distribution_is_fixed_point():
    pi = sp.stationary_distribution(P_WEATHER)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi > 0)
    assert np.allclose(pi @ P_WEATHER, pi)


def test_markov_chain_long_run_frequencies_match_stationary():
    pi = sp.stationary_distribution(P_WEATHER)
    path = sp.simulate_chain(P_WEATHER, 0, 300_000, make_rng(65))
    emp = sp.empirical_distribution(path, 3)
    assert np.max(np.abs(emp - pi)) < 0.01


def test_n_step_distribution_converges_to_stationary():
    pi = sp.stationary_distribution(P_WEATHER)
    assert sp.total_variation(sp.n_step_distribution(P_WEATHER, 2, 60), pi) < 1e-10


def test_kac_mean_return_time_is_inverse_stationary_probability():
    pi = sp.stationary_distribution(P_WEATHER)
    path = sp.simulate_chain(P_WEATHER, 0, 300_000, make_rng(66))
    gaps = sp.return_times(path, 1)
    assert gaps.mean() == pytest.approx(1 / pi[1], rel=0.03)


def test_chain_only_uses_possible_transitions():
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    path = sp.simulate_chain(P, 0, 20, make_rng(67))
    assert list(path[:4]) == [0, 1, 0, 1]


def test_poisson_process_counts_and_interarrivals():
    rng = make_rng(68)
    rate, T = 3.0, 10.0
    counts = np.array([sp.poisson_process_arrivals(rate, T, rng).size for _ in range(4000)])
    assert counts.mean() == pytest.approx(rate * T, rel=0.02)
    assert counts.var() == pytest.approx(rate * T, rel=0.08)   # Poisson: mean == variance
    arr = sp.poisson_process_arrivals(rate, 20_000.0, rng)
    assert np.diff(arr).mean() == pytest.approx(1 / rate, rel=0.02)
