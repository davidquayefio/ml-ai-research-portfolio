import numpy as np
import pytest
from scipy import stats

import distributions as d
from plab import make_rng


def test_eleven_distributions():
    assert len(d.default_distributions()) == 11


@pytest.mark.parametrize("name", list(d.default_distributions()))
def test_simulated_mean_and_variance_match_theory(name):
    dist = d.default_distributions()[name]
    n = 400_000
    x = d.draw_samples(dist, n, make_rng(11))
    th = d.theoretical_moments(dist)
    se_mean = np.sqrt(th["var"] / n)
    assert abs(x.mean() - th["mean"]) < 5 * se_mean
    # sample variance: relative tolerance (heavy tails make its SE larger, so be generous)
    assert abs(x.var(ddof=1) / th["var"] - 1) < 0.05


def test_discrete_samples_have_small_tv_distance():
    dist = d.default_distributions()["Poisson(4)"]
    x = d.draw_samples(dist, 200_000, make_rng(12))
    assert d.tv_distance_discrete(x, dist) < 0.01


def test_ks_accepts_correct_distribution_rejects_wrong():
    dist = d.default_distributions()["Normal(1, 2^2)"]
    x = d.draw_samples(dist, 5000, make_rng(13))
    assert d.ks_test(x, dist)[1] > 0.001
    assert d.ks_test(x, stats.norm(0, 1))[1] < 1e-6


def test_mgf_finite_difference_recovers_moments():
    for name, (dist, mgf) in d.mgf_catalogue().items():
        mean, var = d.mgf_moments_finite_difference(mgf)
        assert mean == pytest.approx(dist.mean(), rel=1e-4, abs=1e-5), name
        assert var == pytest.approx(dist.var(), rel=1e-3, abs=1e-4), name


def test_empirical_mgf_matches_closed_form():
    dist, mgf = d.mgf_catalogue()["Normal(1, 2^2)"]
    t = np.array([-0.5, 0.0, 0.25])
    x = d.draw_samples(dist, 400_000, make_rng(14))
    assert np.allclose(d.empirical_mgf(x, t), mgf(t), rtol=0.02)


def test_inverse_transform_exponential_ks():
    x = d.inverse_transform_exponential(20_000, 1.5, make_rng(15))
    assert d.ks_test(x, stats.expon(scale=1 / 1.5))[1] > 0.001


def test_inverse_transform_geometric_pmf():
    x = d.inverse_transform_geometric(200_000, 0.2, make_rng(16))
    assert x.min() >= 1
    assert d.tv_distance_discrete(x, stats.geom(0.2)) < 0.01


def test_cholesky_sampling_has_requested_covariance():
    cov = np.array([[1.0, 0.7], [0.7, 2.0]])
    y = d.correlated_normals(300_000, cov, make_rng(17))
    assert np.allclose(np.cov(y.T), cov, atol=0.02)


def test_uncorrelated_but_dependent():
    x, y = d.uncorrelated_but_dependent(500_000, make_rng(18))
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.01          # Cov(X, X^2) = E[X^3] = 0
    assert np.corrcoef(np.abs(x), y)[0, 1] > 0.8         # but strongly dependent
