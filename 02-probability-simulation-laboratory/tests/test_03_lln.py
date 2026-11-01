import numpy as np
import pytest

import lln
from plab import make_rng


def test_running_mean_last_value():
    x = np.array([[1.0, 2.0, 3.0, 6.0]])
    assert lln.running_mean(x)[0, -1] == pytest.approx(3.0)


def test_chebyshev_bound_dominates_empirical_probability():
    s = lln.SAMPLERS["Bernoulli(0.3)"]
    n_grid = [50, 200, 800]
    means = lln.sample_means_at(s, n_grid, reps=4000, rng=make_rng(21))
    for j, n in enumerate(n_grid):
        emp = lln.exceedance_probability(means[:, j], s.mean, 0.05)
        assert emp <= lln.chebyshev_bound(s.var, n, 0.05) + 1e-9


def test_clt_approximation_close_to_empirical():
    s = lln.SAMPLERS["Uniform(0,1)"]
    means = lln.sample_means_at(s, [400], reps=20_000, rng=make_rng(22))[:, 0]
    emp = lln.exceedance_probability(means, s.mean, 0.01)
    approx = lln.clt_approximation(s.var, 400, 0.01)
    assert abs(emp - approx) < 0.02


def test_finite_variance_error_decays_like_inverse_sqrt_n():
    s = lln.SAMPLERS["Exponential(1)"]
    n_grid = [10, 100, 1000, 5000]
    means = lln.sample_means_at(s, n_grid, reps=1500, rng=make_rng(23))
    err = np.median(np.abs(means - s.mean), axis=0)
    assert lln.loglog_slope(n_grid, err) == pytest.approx(-0.5, abs=0.08)


def test_cauchy_sample_mean_does_not_concentrate():
    s = lln.SAMPLERS["Cauchy"]
    means = lln.sample_means_at(s, [1, 100, 3000], reps=3000, rng=make_rng(24))
    iqrs = [np.subtract(*np.percentile(means[:, j], [75, 25])) for j in range(3)]
    # Xbar_n is again standard Cauchy for every n: IQR stays near 2
    assert all(1.7 < v < 2.3 for v in iqrs)
