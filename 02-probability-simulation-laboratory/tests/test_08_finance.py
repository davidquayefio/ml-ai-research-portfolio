import numpy as np
import pytest
from scipy import stats

import finance as fin
from plab import make_rng

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0


def test_black_scholes_reference_values():
    assert fin.bs_call(S0, K, R, SIG, T) == pytest.approx(10.4506, abs=1e-3)
    assert fin.bs_put(S0, K, R, SIG, T) == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity_exact():
    c, p = fin.bs_call(S0, 95, R, SIG, 0.75), fin.bs_put(S0, 95, R, SIG, 0.75)
    assert c - p == pytest.approx(S0 - 95 * np.exp(-R * 0.75))


def test_implied_vol_round_trip():
    price = fin.bs_call(S0, 110, R, 0.33, 0.5)
    assert fin.implied_vol_call(price, S0, 110, R, 0.5) == pytest.approx(0.33, abs=1e-6)


def test_vega_matches_finite_difference():
    h = 1e-5
    fd = (fin.bs_call(S0, K, R, SIG + h, T) - fin.bs_call(S0, K, R, SIG - h, T)) / (2 * h)
    assert fin.bs_vega(S0, K, R, SIG, T) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
def test_mc_call_within_four_standard_errors_of_black_scholes(method):
    est = fin.mc_european_call(S0, K, R, SIG, T, 400_000, make_rng(81), method)
    assert abs(est.price - fin.bs_call(S0, K, R, SIG, T)) < 4 * est.std_error


def test_variance_reduction_ordering():
    plain = fin.mc_european_call(S0, K, R, SIG, T, 200_000, make_rng(82), "plain")
    anti = fin.mc_european_call(S0, K, R, SIG, T, 200_000, make_rng(83), "antithetic")
    ctrl = fin.mc_european_call(S0, K, R, SIG, T, 200_000, make_rng(84), "control")
    assert anti.std_error < plain.std_error
    assert ctrl.std_error < plain.std_error


def test_mc_put_call_parity():
    c, p = fin.mc_european_put_call_pair(S0, K, R, SIG, T, 400_000, make_rng(85))
    # parity holds *sample by sample* up to the sample mean of S_T, so error is tiny
    assert c.price - p.price == pytest.approx(S0 - K * np.exp(-R * T), abs=0.2)


def test_price_increases_with_volatility():
    prices = [fin.bs_call(S0, K, R, s, T) for s in (0.1, 0.2, 0.3, 0.5)]
    assert all(np.diff(prices) > 0)


def test_geometric_asian_formula_matches_monte_carlo():
    m, n = 12, 400_000
    z = make_rng(86).standard_normal((n, m))
    dt = T / m
    log_s = np.log(S0) + np.cumsum((R - 0.5 * SIG**2) * dt + SIG * np.sqrt(dt) * z, axis=1)
    y = np.exp(-R * T) * np.maximum(np.exp(log_s.mean(axis=1)) - K, 0)
    se = y.std(ddof=1) / np.sqrt(n)
    assert abs(y.mean() - fin.geometric_asian_call(S0, K, R, SIG, T, m)) < 4 * se


def test_asian_control_variate_reduces_variance_dramatically():
    plain = fin.mc_asian_call(S0, K, R, SIG, T, 12, 100_000, make_rng(87), control=False)
    ctrl = fin.mc_asian_call(S0, K, R, SIG, T, 12, 100_000, make_rng(88), control=True)
    assert abs(plain.price - ctrl.price) < 4 * plain.std_error
    assert ctrl.std_error < plain.std_error / 5


def test_returns_relationships():
    prices = np.array([100.0, 150.0, 75.0])
    R_ = fin.simple_returns(prices)
    r = fin.log_returns(prices)
    assert np.allclose(R_, [0.5, -0.5])
    assert np.allclose(r, np.log1p(R_))
    assert r.sum() == pytest.approx(np.log(prices[-1] / prices[0]))       # log returns add up
    assert (1 + R_).prod() - 1 == pytest.approx(prices[-1] / prices[0] - 1)  # simple returns compound
    assert R_.sum() != pytest.approx(prices[-1] / prices[0] - 1)


def test_normal_var_es_known_values():
    var, es = fin.normal_var_es(0.0, 1.0, 0.99)
    assert var == pytest.approx(2.3263, abs=1e-3)
    assert es == pytest.approx(2.6652, abs=1e-3)


def test_historical_var_es_close_to_normal_theory():
    losses = make_rng(89).standard_normal(2_000_000)
    var, es = fin.historical_var_es(losses, 0.99)
    tv, te = fin.normal_var_es(0, 1, 0.99)
    assert var == pytest.approx(tv, abs=0.02) and es == pytest.approx(te, abs=0.03)
    assert es > var


def test_student_t_var_es_matches_simulation():
    nu, sigma = 5.0, 1.0
    s = sigma * np.sqrt((nu - 2) / nu)
    losses = s * make_rng(90).standard_t(nu, 3_000_000)
    var, es = fin.historical_var_es(losses, 0.99)
    tv, te = fin.student_t_var_es(0.0, sigma, nu, 0.99)
    assert var == pytest.approx(tv, rel=0.02) and es == pytest.approx(te, rel=0.03)


def test_es_exceeds_var_and_fat_tails_raise_es_relative_to_normal():
    nv, ne = fin.normal_var_es(0, 1, 0.99)
    tv, te = fin.student_t_var_es(0, 1, 4.0, 0.99)
    assert te > tv and ne > nv
    assert te / tv > ne / nv      # heavier tails -> ES/VaR ratio is larger


def test_tail_probabilities():
    assert fin.normal_tail_probability(4) == pytest.approx(6.334e-5, rel=1e-3)
    assert fin.student_t_tail_probability(4, 4.0) > 20 * fin.normal_tail_probability(4)


def test_garch_has_fat_tails_and_volatility_clustering():
    r = fin.garch11(300_000, 1e-6, 0.10, 0.88, make_rng(91))
    assert stats.kurtosis(r) > 1.5
    assert abs(fin.acf(r, 5)).max() < 0.05           # returns are nearly uncorrelated ...
    assert fin.acf(r ** 2, 5)[0] > 0.1               # ... but squared returns are not


def test_normal_var_underestimates_garch_risk():
    r = fin.garch11(400_000, 1e-6, 0.10, 0.88, make_rng(92))
    losses = -r
    var, _ = fin.normal_var_es(losses.mean(), losses.std(), 0.99)
    assert fin.exceedance_rate(losses, var) > 0.011
