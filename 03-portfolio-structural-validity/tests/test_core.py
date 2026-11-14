"""Unit tests. All data here is SIMULATED (src.data_processing.simulate_returns)
and is used only to verify that the code is mathematically correct."""
import numpy as np
import pandas as pd
import pytest

from src import covariance_estimators as ce
from src import portfolio_models as pm
from src import evaluation as ev
from src import stability_tests as st
from src import stress_testing as sx
from src import regime_detection as rd
from src.backtesting import walk_forward, make_models, in_sample_vs_out_of_sample
from src.data_processing import simulate_returns, compute_returns, clean_prices


@pytest.fixture(scope="module")
def R():
    return simulate_returns(n_obs=1200, n_assets=5, seed=1)


def test_weights_sum_to_one_and_nonneg(R):
    X = R.to_numpy()
    for name in pm.CORE_TEN:
        w = pm.MODELS[name](X)
        assert w.shape == (5,)
        assert abs(w.sum() - 1) < 1e-6, name
        assert w.min() >= -1e-8, name


def test_gmv_closed_form_matches_numeric_when_interior():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 4)) * 0.01
    A += 0.01 * rng.normal(size=(400, 1))  # positive correlation, interior solution likely
    S = ce.sample_covariance(A)
    w_cf = pm.gmv_unconstrained(A)
    # closed form is the global minimum among all sum-to-one portfolios
    for _ in range(200):
        d = rng.normal(size=4)
        d -= d.mean()
        w2 = w_cf + 0.01 * d
        assert w2 @ S @ w2 >= w_cf @ S @ w_cf - 1e-14


def test_gmv_long_only_not_worse_than_equal_weight(R):
    X = R.to_numpy()
    S = ce.sample_covariance(X)
    w = pm.gmv_long_only(X)
    ew = np.full(5, 0.2)
    assert w @ S @ w <= ew @ S @ ew + 1e-12


def test_risk_parity_equalises_contributions(R):
    X = R.to_numpy()
    S = ce.sample_covariance(X)
    w = pm.risk_parity(X)
    rc = pm.risk_contributions(w, S)
    assert np.max(np.abs(rc - 0.2)) < 5e-3


def test_min_vol_cap_respected(R):
    w = pm.min_volatility_capped(R.to_numpy(), cap=0.3)
    assert w.max() <= 0.3 + 1e-6


def test_black_litterman_no_views_recovers_prior():
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    mu = pm.black_litterman_mu(S, w, delta=2.5)
    np.testing.assert_allclose(mu, 2.5 * S @ w)


def test_black_litterman_view_moves_mu_toward_view():
    S = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.5, 0.5])
    P = np.array([[1.0, 0.0]])
    prior = pm.black_litterman_mu(S, w)
    post = pm.black_litterman_mu(S, w, P=P, Q=np.array([prior[0] + 0.05]))
    assert post[0] > prior[0]


def test_shrinkage_improves_conditioning():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 30)) * 0.01   # N close to T -> ill conditioned
    assert ce.condition_number(ce.ledoit_wolf_covariance(X)) < ce.condition_number(ce.sample_covariance(X))


def test_nearest_psd():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    B = ce.nearest_psd(A)
    assert np.linalg.eigvalsh(B).min() > 0


def test_marchenko_pastur_edges():
    lo, hi = st.marchenko_pastur_edges(T=1000, N=100)
    assert 0 < lo < 1 < hi
    # pure noise: largest sample-corr eigenvalue ~ upper edge
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 100))
    ev_max = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)).max()
    assert abs(ev_max - hi) < 0.15


def test_metrics_basic():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005] * 50)
    assert ev.max_drawdown(r) <= 0
    assert ev.historical_cvar(r, 0.05) >= ev.historical_var(r, 0.05) - 1e-12
    assert ev.effective_n(np.full(4, 0.25)) == pytest.approx(4.0)


def test_stationary_bootstrap_indices_valid():
    rng = np.random.default_rng(0)
    ix = ev.stationary_bootstrap_indices(500, 10, rng)
    assert ix.min() >= 0 and ix.max() < 500 and len(ix) == 500


def test_bootstrap_sharpe_difference_identical_series_not_significant(R):
    r = R.iloc[:, 0]
    out = ev.bootstrap_sharpe_difference(r, r, n_boot=200)
    assert out["diff"] == 0 and out["p_value"] >= 0.99


def test_backtest_no_lookahead_and_costs(R):
    models = make_models({"ew": pm.equal_weight, "gmv": pm.gmv_long_only})
    res0 = walk_forward(R, models, window=250, rebalance_every=21, cost_bps=0)
    res1 = walk_forward(R, models, window=250, rebalance_every=21, cost_bps=50)
    assert res0.net_returns.index[0] == R.index[250]
    assert (res1.net_returns.sum() <= res0.net_returns.sum() + 1e-12).all()
    assert (res0.net_returns == res0.gross_returns).all().all()


def test_backtest_uses_only_past_data():
    """Corrupting FUTURE data must not change earlier realised weights."""
    R = simulate_returns(n_obs=700, n_assets=4, seed=5)
    models = make_models({"gmv": pm.gmv_long_only})
    a = walk_forward(R, models, window=250, rebalance_every=21)
    R2 = R.copy()
    R2.iloc[600:] *= 5.0
    b = walk_forward(R2, models, window=250, rebalance_every=21)
    cut = R.index[600]
    wa, wb = a.weights["gmv"], b.weights["gmv"]
    pd.testing.assert_frame_equal(wa[wa.index < cut], wb[wb.index < cut])


def test_regimes_and_labels(R):
    m = rd.market_proxy(R)
    lab = rd.volatility_regimes_ex_ante(m)
    assert set(lab.dropna().unique()) <= {"low_vol", "mid_vol", "high_vol"}
    bb = rd.bull_bear_regimes(m)
    assert set(bb.unique()) <= {"bull", "bear"}


def test_ex_ante_labels_ignore_future(R):
    m = rd.market_proxy(R)
    a = rd.volatility_regimes_ex_ante(m)
    m2 = m.copy()
    m2.iloc[-100:] *= 10
    b = rd.volatility_regimes_ex_ante(m2)
    pd.testing.assert_series_equal(a.iloc[:-150], b.iloc[:-150])


def test_distribution_tests_detect_fat_tails(R):
    tab = st.univariate_normality_table(R)
    assert (tab["excess_kurtosis"] > 0).all()
    m = st.mardia_test(R)
    assert m["kurt_p"] < 0.05


def test_arch_lm_detects_clustering():
    rng = np.random.default_rng(0)
    n = 3000
    e = np.zeros(n)
    h = np.full(n, 1e-4)
    for t in range(1, n):
        h[t] = 1e-5 + 0.2 * e[t - 1] ** 2 + 0.7 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.normal()
    assert st.arch_lm_test(pd.Series(e))["p_value"] < 0.01
    assert st.arch_lm_test(pd.Series(rng.normal(size=n)))["p_value"] > 0.001


def test_box_m_detects_covariance_break():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 3))
    b = rng.normal(size=(500, 3)) * 2
    assert st.box_m_test(a, b)["p_value"] < 1e-6
    assert st.box_m_test(a, rng.normal(size=(500, 3)))["p_value"] > 0.001


def test_perturbation_response_grows_with_level(R):
    X = R.to_numpy()
    df = sx.weight_response(
        pm.MODELS["max_sharpe"], X,
        lambda x, lev, rng: sx.perturb_mean(x, lev, rng),
        levels=[0.1, 3.0], n_rep=15,
    )
    g = df.groupby("level")["turnover_equiv"].mean()
    assert g[3.0] > g[0.1]


def test_equal_weight_is_insensitive_to_perturbation(R):
    X = R.to_numpy()
    df = sx.weight_response(
        pm.MODELS["equal_weight"], X,
        lambda x, lev, rng: sx.perturb_mean(x, lev, rng), levels=[3.0], n_rep=5,
    )
    assert df["turnover_equiv"].max() < 1e-12


def test_structural_validity_table_runs(R):
    names = ["equal_weight", "gmv_long_only", "max_sharpe"]
    models = make_models({n: pm.MODELS[n].fn for n in names})
    res = walk_forward(R, models, window=250, rebalance_every=21)
    wsum = st.weight_stability_summary(res.weights)
    sens = pd.DataFrame(
        {n: st.weight_sensitivity_to_sampling(pm.MODELS[n], R.to_numpy()[:250], n_boot=10)
         for n in names}
    ).T[["mean_turnover_equiv"]].astype(float)
    gap = in_sample_vs_out_of_sample(res, ev.sharpe_ratio)
    lab = rd.volatility_regimes_ex_post(rd.market_proxy(R))
    reg = ev.regime_conditional_performance(res.net_returns, lab, min_obs=20)
    tab = st.structural_validity_table(wsum, sens, gap, reg)
    assert "composite_rank_score" in tab.columns and len(tab) == 3
    assert st.composite_robustness_to_dropping(tab).shape[0] == 3


def test_clean_and_returns():
    idx = pd.bdate_range("2020-01-01", periods=10)
    p = pd.DataFrame({"A": np.arange(1, 11, dtype=float), "B": np.arange(1, 11, dtype=float)}, index=idx)
    p.iloc[3, 0] = np.nan
    c = clean_prices(p, max_ffill_days=1)
    assert len(c) == 10  # single gap forward-filled, not invented from nothing
    r = compute_returns(c)
    assert len(r) == 9


def test_black_litterman_default_differs_from_equal_weight_and_no_views_equals_it(R):
    X = R.to_numpy()
    w_views = pm.black_litterman(X)
    w_none = pm.black_litterman(X, data_driven_views=False)
    np.testing.assert_allclose(w_none, np.full(5, 0.2), atol=1e-4)  # reverse optimisation recovers prior
    assert np.abs(w_views - 0.2).sum() > 1e-3
