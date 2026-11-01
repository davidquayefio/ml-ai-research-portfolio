"""Tests for the shared library."""

import numpy as np

from plab import ecdf, make_rng, mean_ci, spawn_rngs, standard_error, wilson_interval
from plab.processes import brownian_paths, gbm_euler_paths, gbm_exact_paths, brownian_increments


def test_rng_reproducible():
    assert np.array_equal(make_rng(1).random(5), make_rng(1).random(5))
    assert not np.array_equal(make_rng(1).random(5), make_rng(2).random(5))


def test_spawned_streams_differ():
    a, b = spawn_rngs(7, 2)
    assert not np.array_equal(a.random(5), b.random(5))


def test_standard_error_and_ci():
    x = make_rng(0).normal(5, 2, 10_000)
    assert abs(standard_error(x) - 2 / 100) < 0.003
    m, lo, hi = mean_ci(x)
    assert lo < m < hi and lo < 5 < hi


def test_wilson_interval_contains_p():
    lo, hi = wilson_interval(500, 1000)
    assert lo < 0.5 < hi
    lo, hi = wilson_interval(0, 20)  # boundary case still sensible
    assert 0 <= lo <= hi < 0.2


def test_ecdf_properties():
    xs, ys = ecdf([3, 1, 2])
    assert list(xs) == [1, 2, 3] and ys[-1] == 1.0


def test_brownian_shapes_and_start():
    t, W = brownian_paths(100, 2.0, 7, make_rng(0))
    assert t.shape == (101,) and W.shape == (7, 101)
    assert np.all(W[:, 0] == 0) and t[-1] == 2.0


def test_gbm_exact_positive_and_start():
    t, S = gbm_exact_paths(100, 0.05, 0.2, 1.0, 50, 10, make_rng(0))
    assert np.all(S > 0) and np.allclose(S[:, 0], 100)


def test_euler_matches_exact_for_zero_vol():
    dW = brownian_increments(1000, 1.0, 3, make_rng(0))
    S = gbm_euler_paths(100, 0.05, 0.0, 1 / 1000, dW)
    assert abs(S[0, -1] - 100 * np.exp(0.05)) < 0.1
