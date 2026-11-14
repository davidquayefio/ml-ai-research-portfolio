"""Run the full empirical pipeline headlessly and write tables and figures.

    python scripts/run_pipeline.py                 # real data (after download_data.py)
    python scripts/run_pipeline.py --simulate DIR  # SIMULATED smoke test -> DIR only

Real-data outputs go to results/tables and results/figures.
The --simulate mode refuses to write inside results/, so simulated numbers
can never be mistaken for findings.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_processing as dp  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src import portfolio_models as pm  # noqa: E402
from src import regime_detection as rd  # noqa: E402
from src import stability_tests as st  # noqa: E402
from src import stress_testing as sx  # noqa: E402
from src import covariance_estimators as ce  # noqa: E402
from src.backtesting import in_sample_vs_out_of_sample, make_models, walk_forward  # noqa: E402

warnings.filterwarnings("ignore")


def run(returns: pd.DataFrame, out: Path, window=504, rebalance=21, cost_bps=10.0, n_boot=500):
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    X = returns.to_numpy()

    # Test 5: distributions
    st.univariate_normality_table(returns).to_csv(tab / "t5_univariate_normality.csv")
    pd.Series(st.mardia_test(returns)).to_csv(tab / "t5_mardia.csv")
    pd.DataFrame({c: st.arch_lm_test(returns[c]) for c in returns.columns}).T.to_csv(tab / "t5_arch_lm.csv")

    # Test 2: covariance structure
    diag = st.rolling_covariance_diagnostics(returns, 252, 21)
    diag.to_csv(tab / "t2_rolling_cov_diagnostics.csv")
    st.eigenvector_stability(returns, 252, 21).to_csv(tab / "t2_eigenvector_stability.csv")
    fig1, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    diag["condition_number"].plot(ax=ax[0], logy=True, title="Condition number (252d windows)")
    diag["pc1_share"].plot(ax=ax[1], title="Variance share of PC1")
    diag["mean_corr"].plot(ax=ax[2], title="Mean pairwise correlation")
    fig1.tight_layout(); fig1.savefig(fig / "t2_covariance_diagnostics.png", dpi=150); plt.close(fig1)

    # Walk-forward backtest (Tests 3 and 6)
    spec = {n: pm.MODELS[n].fn for n in pm.CORE_TEN}
    models = make_models(spec)
    res = walk_forward(returns, models, window=window, rebalance_every=rebalance, cost_bps=cost_bps)
    res_exp = walk_forward(returns, models, window=window, rebalance_every=rebalance,
                           expanding=True, cost_bps=cost_bps)
    res.net_returns.to_csv(tab / "t6_net_returns_rolling.csv")
    perf = ev.performance_table(res.net_returns, res.mean_turnover(), res.mean_effective_n())
    perf.to_csv(tab / "t6_performance_rolling_net.csv")
    ev.performance_table(res.gross_returns).to_csv(tab / "t6_performance_rolling_gross.csv")
    ev.performance_table(res_exp.net_returns, res_exp.mean_turnover(), res_exp.mean_effective_n()
                         ).to_csv(tab / "t6_performance_expanding_net.csv")
    gap = in_sample_vs_out_of_sample(res, ev.sharpe_ratio)
    gap.to_csv(tab / "t6_in_vs_out_of_sample.csv")
    wsum = st.weight_stability_summary(res.weights)
    wsum.to_csv(tab / "t3_weight_stability.csv")

    # Cost sensitivity
    costs = {}
    for c in (0, 10, 25, 50):
        r = walk_forward(returns, models, window=window, rebalance_every=rebalance, cost_bps=c)
        costs[c] = {m: ev.sharpe_ratio(r.net_returns[m]) for m in r.net_returns.columns}
    pd.DataFrame(costs).to_csv(tab / "t6_sharpe_vs_cost_bps.csv")

    # Bootstrap comparisons against equal weight
    rows = {}
    for m in res.net_returns.columns:
        if m == "equal_weight":
            continue
        d = ev.bootstrap_sharpe_difference(res.net_returns[m], res.net_returns["equal_weight"], n_boot=n_boot)
        d["CE_gain_gamma5"] = ev.economic_significance(res.net_returns[m], res.net_returns["equal_weight"])
        rows[m] = d
    pd.DataFrame(rows).T.to_csv(tab / "t6_sharpe_diff_vs_equal_weight.csv")

    # Test 4: regimes
    mkt = rd.market_proxy(returns)
    reg_vol = rd.volatility_regimes_ex_post(mkt)
    reg_bb = rd.bull_bear_regimes(mkt)
    reg_st = rd.stress_labels(returns.index)
    regs = {"vol": reg_vol, "bullbear": reg_bb, "stress": reg_st}
    all_reg = []
    for k, lab in regs.items():
        t = ev.regime_conditional_performance(res.net_returns, lab)
        t.insert(0, "scheme", k)
        all_reg.append(t)
    reg_tab = pd.concat(all_reg)
    reg_tab.to_csv(tab / "t4_regime_performance.csv", index=False)

    # Test 1: sensitivity on the latest estimation window
    last = X[-window:]
    sens = {m: st.weight_sensitivity_to_sampling(pm.MODELS[m], last, n_boot=100)
            for m in pm.CORE_TEN}
    sens_df = pd.DataFrame(sens).T[["mean_L1_distance", "mean_turnover_equiv"]].astype(float)
    sens_df.to_csv(tab / "t1_sampling_sensitivity.csv")

    # Test 7: perturbations
    pert_mean = sx.compare_models_under_perturbation(
        {m: pm.MODELS[m] for m in pm.CORE_TEN}, last,
        lambda x, lev, rng: sx.perturb_mean(x, lev, rng), level=1.0, n_rep=30)
    pert_corr = sx.compare_models_under_perturbation(
        {m: pm.MODELS[m] for m in pm.CORE_TEN}, last,
        lambda x, lev, rng: sx.shift_correlation(x, lev), level=0.5, n_rep=1)
    pert_mean.to_csv(tab / "t7_perturb_mean_1SE.csv")
    pert_corr.to_csv(tab / "t7_perturb_correlation_shift.csv")

    # Composite
    comp = st.structural_validity_table(
        wsum, sens_df, gap, reg_tab[reg_tab.scheme == "vol"], pert_mean)
    comp.to_csv(tab / "composite_structural_validity.csv")
    st.composite_robustness_to_dropping(comp).to_csv(tab / "composite_drop_one_component.csv")

    # Figures
    fig2, ax = plt.subplots(figsize=(9, 5))
    (1 + res.net_returns).cumprod().plot(ax=ax, logy=True, title="Out-of-sample wealth (net of costs)")
    fig2.tight_layout(); fig2.savefig(fig / "t6_cumulative_wealth.png", dpi=150); plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(comp["S1_weight_instability"], perf.loc[comp.index, "sharpe"])
    for m in comp.index:
        ax.annotate(m, (comp.loc[m, "S1_weight_instability"], perf.loc[m, "sharpe"]), fontsize=7)
    ax.set_xlabel("Mean turnover between rebalances (weight instability)")
    ax.set_ylabel("Out-of-sample net Sharpe")
    ax.set_title("Performance vs structural instability")
    fig3.tight_layout(); fig3.savefig(fig / "composite_sharpe_vs_instability.png", dpi=150); plt.close(fig3)

    return {"performance": perf, "composite": comp, "gap": gap}


def main():
    if "--simulate" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--simulate") + 1]).resolve()
        if str(out).startswith(str((ROOT / "results").resolve())):
            sys.exit("Refusing to write simulated output inside results/.")
        returns = dp.simulate_returns(n_obs=1600, n_assets=8, seed=42)
        print("SIMULATED DATA -- smoke test only, not findings.")
        res = run(returns, out, window=400, n_boot=50)
    else:
        path = ROOT / "data/processed/returns_daily.csv"
        if not path.exists():
            sys.exit("Missing data. Run: python scripts/download_data.py")
        returns = pd.read_csv(path, index_col=0, parse_dates=True)
        res = run(returns, ROOT / "results")
    print(res["performance"].round(3).to_string())
    print(res["composite"].round(3).to_string())


if __name__ == "__main__":
    main()
