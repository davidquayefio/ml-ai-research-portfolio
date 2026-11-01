import numpy as np
import pytest
from scipy import stats

import clt
from plab import make_rng


def _setup(dist):
    return clt.from_frozen(dist)


def test_ks_distance_shrinks_with_n_for_exponential():
    draw, mu, sigma = _setup(stats.expon())
    rng = make_rng(31)
    d1 = clt.ks_distance_to_normal(clt.standardized_means(draw, mu, sigma, 1, 20_000, rng))
    d100 = clt.ks_distance_to_normal(clt.standardized_means(draw, mu, sigma, 100, 20_000, rng))
    assert d100 < d1 / 4


def test_standardized_means_have_mean_zero_variance_one():
    draw, mu, sigma = _setup(stats.uniform())
    z = clt.standardized_means(draw, mu, sigma, 50, 40_000, make_rng(32))
    assert abs(z.mean()) < 0.03 and abs(z.var() - 1) < 0.03


def test_skewness_of_sample_mean_is_gamma_over_sqrt_n():
    draw, mu, sigma = _setup(stats.expon())   # skewness 2
    n = 25
    z = clt.standardized_means(draw, mu, sigma, n, 200_000, make_rng(33))
    assert stats.skew(z) == pytest.approx(2 / np.sqrt(n), abs=0.04)


def test_sample_means_chunking_consistent_shape():
    draw, _, _ = _setup(stats.norm())
    m = clt.sample_means(draw, 1000, 50, make_rng(34), max_block=10_000)
    assert m.shape == (50,)


def test_t_interval_is_exact_for_normal_data():
    draw, mu, _ = _setup(stats.norm(3, 2))
    cov = clt.student_t_coverage(draw, mu, 10, 20_000, make_rng(35))
    assert abs(cov - 0.95) < 0.01


def test_t_interval_undercovers_for_skewed_data_small_n():
    draw, mu, _ = _setup(stats.lognorm(s=1.0))
    cov = clt.student_t_coverage(draw, mu, 10, 20_000, make_rng(36))
    assert cov < 0.92


def test_cauchy_means_do_not_normalise():
    draw = lambda rng, size: rng.standard_cauchy(size)
    m = clt.sample_means(draw, 500, 6000, make_rng(37))
    assert clt.iqr(m) == pytest.approx(2.0, abs=0.2)
