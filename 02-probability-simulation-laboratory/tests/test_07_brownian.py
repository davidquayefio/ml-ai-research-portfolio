import numpy as np
import pytest
from scipy import stats

import brownian as bm
from plab import make_rng


def test_brownian_variance_and_covariance():
    t, W = bm.brownian_paths(100, 1.0, 100_000, make_rng(71))
    i, j = 30, 80   # s = 0.3, t = 0.8
    assert W[:, j].var() == pytest.approx(t[j], rel=0.02)
    assert np.mean(W[:, i] * W[:, j]) == pytest.approx(min(t[i], t[j]), abs=0.01)


def test_increments_normal_and_independent():
    dW = bm.brownian_increments(2, 1.0, 100_000, make_rng(72))
    assert stats.kstest(dW[:, 0], stats.norm(0, np.sqrt(0.5)).cdf).pvalue > 0.001
    assert abs(np.corrcoef(dW[:, 0], dW[:, 1])[0, 1]) < 0.02


def test_quadratic_variation_converges_to_T_with_variance_2T2_over_n():
    T, n = 2.0, 1024
    dW = bm.brownian_increments(n, T, 40_000, make_rng(73))
    qv = bm.quadratic_variation(dW)
    assert qv.mean() == pytest.approx(T, rel=0.01)
    assert qv.var() == pytest.approx(2 * T**2 / n, rel=0.05)


def test_total_variation_grows_like_sqrt_n():
    T = 1.0
    out = []
    for n in (256, 4096):
        dW = bm.brownian_increments(n, T, 5000, make_rng(74))
        out.append(bm.total_variation(dW).mean())
        assert out[-1] == pytest.approx(np.sqrt(2 * n * T / np.pi), rel=0.02)
    assert out[1] > 3.5 * out[0]


def test_coarsening_preserves_path_endpoint_and_is_consistent():
    dW = bm.brownian_increments(64, 1.0, 10, make_rng(75))
    c = bm.coarsen_increments(dW, 8)
    assert c.shape == (10, 8)
    assert np.allclose(c.sum(axis=1), dW.sum(axis=1))
    with pytest.raises(ValueError):
        bm.coarsen_increments(dW, 7)


def test_donsker_terminal_value_is_close_to_standard_normal():
    z = bm.scaled_walk_terminal(10_000, 50_000, make_rng(76))
    assert abs(z.mean()) < 0.02 and z.var() == pytest.approx(1.0, rel=0.03)


def test_reflection_principle_probability():
    a, T = 1.0, 1.0
    theory = bm.reflection_max_probability(a, T)
    assert theory == pytest.approx(0.3173105, abs=1e-6)
    est = bm.discrete_max_exceedance(a, T, 2000, 20_000, make_rng(77))
    assert est < theory                                   # discrete monitoring biases the maximum low
    assert abs(est - theory) < 0.03


def test_gbm_terminal_moments():
    S0, mu, sigma, T = 100.0, 0.08, 0.3, 1.5
    z = make_rng(78).standard_normal(800_000)
    s = bm.gbm_terminal(S0, mu, sigma, T, z)
    th = bm.gbm_theoretical_moments(S0, mu, sigma, T)
    assert s.mean() == pytest.approx(th["mean"], rel=0.01)
    assert np.median(s) == pytest.approx(th["median"], rel=0.01)
    assert th["mean"] > th["median"]                      # lognormal skew


def test_log_gbm_is_normal_with_itos_drift():
    S0, mu, sigma, T = 50.0, 0.1, 0.4, 2.0
    z = make_rng(79).standard_normal(200_000)
    logret = np.log(bm.gbm_terminal(S0, mu, sigma, T, z) / S0)
    assert logret.mean() == pytest.approx((mu - 0.5 * sigma**2) * T, abs=0.01)
    assert logret.std() == pytest.approx(sigma * np.sqrt(T), rel=0.01)


def test_euler_strong_order_one_half():
    steps = [4, 8, 16, 32, 64, 128]
    err = bm.euler_strong_error(100.0, 0.05, 0.4, 1.0, steps, 20_000, make_rng(80))
    slope = np.polyfit(np.log(steps), np.log(err), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.12)
