import numpy as np
import pytest
from scipy import stats

import monte_carlo as mc
from plab import make_rng


def test_pi_estimate_within_five_standard_errors():
    r = mc.estimate_pi(400_000, make_rng(41))
    assert abs(r.estimate - np.pi) < 5 * r.std_error


def test_pi_standard_error_matches_theory():
    r = mc.estimate_pi(400_000, make_rng(42))
    p = np.pi / 4
    assert r.std_error == pytest.approx(4 * np.sqrt(p * (1 - p) / 400_000), rel=0.02)


def test_real_root_probability_is_one_twelfth():
    r = mc.estimate_real_root_probability(2_000_000, make_rng(43))
    assert abs(r.estimate - 1 / 12) < 5 * r.std_error


def test_plain_integral_of_exp():
    r = mc.mc_integral(np.exp, 0, 1, 400_000, make_rng(44))
    assert abs(r.estimate - (np.e - 1)) < 5 * r.std_error


def test_antithetic_reduces_variance_for_monotone_integrand():
    n = 100_000
    plain = mc.mc_integral(np.exp, 0, 1, n, make_rng(45))
    anti = mc.antithetic_integral(np.exp, 0, 1, n, make_rng(46))
    assert abs(anti.estimate - (np.e - 1)) < 5 * anti.std_error
    # theory: variance ratio 1 + rho = about 0.0323 (30x lower)
    assert anti.std_error ** 2 < plain.std_error ** 2 / 10


def test_control_variate_reduces_variance_and_beta_is_near_theory():
    n = 100_000
    plain = mc.mc_integral(np.exp, 0, 1, n, make_rng(47))
    cv, beta = mc.control_variate_integral(np.exp, 0, 1, n, make_rng(48))
    assert abs(cv.estimate - (np.e - 1)) < 5 * cv.std_error
    assert cv.std_error ** 2 < plain.std_error ** 2 / 20
    assert beta == pytest.approx((1 - 0.5 * (np.e - 1)) * 12, abs=0.05)  # Cov(e^U,U)/Var(U)


def test_importance_sampling_beats_naive_for_rare_event():
    c, n = 4.0, 200_000
    truth = stats.norm.sf(c)
    imp = mc.importance_sampling_normal_tail(c, n, make_rng(49))
    naive = mc.naive_normal_tail(c, n, make_rng(50))
    assert abs(imp.estimate - truth) < 5 * imp.std_error
    assert naive.std_error > 0                      # naive run saw at least one hit
    assert imp.std_error < naive.std_error / 10     # roughly 20-40x lower in practice


def test_importance_sampling_relative_error_is_small():
    c = 4.0
    imp = mc.importance_sampling_normal_tail(c, 100_000, make_rng(51))
    assert imp.std_error / stats.norm.sf(c) < 0.02


def test_confidence_interval_coverage_close_to_nominal():
    cov = mc.ci_coverage(lambda g: mc.estimate_pi(2000, g), np.pi, 1500, make_rng(52))
    assert abs(cov - 0.95) < 0.025


def test_ball_volume():
    est =mc.ball_volume_hit_or_miss(3, 500_000, make_rng(53))
    assert abs(est.estimate - mc.ball_volume_exact(3)) < 5 * est.std_error
    assert mc.ball_volume_exact(2) == pytest.approx(np.pi)
    assert mc.ball_volume_exact(3) == pytest.approx(4 * np.pi / 3)
