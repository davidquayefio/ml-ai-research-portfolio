"""Structural validity tests (Tests 1, 2, 3, 5 and the composite index).

Research purpose: these functions turn the informal question "is the model
stable?" into numbers that can be compared across models and tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import covariance_estimators as ce


# ----------------------------------------------------------------------------
# Weight diagnostics
# ----------------------------------------------------------------------------
def herfindahl(w: np.ndarray) -> float:
    return float(np.sum(np.asarray(w) ** 2))


def weight_turnover_series(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover between consecutive target weight vectors (no drift)."""
    return 0.5 * weights.diff().abs().sum(axis=1).dropna()


def weight_dispersion(weights: pd.DataFrame) -> pd.Series:
    """Time-series standard deviation of each asset's weight."""
    return weights.std(ddof=1)


def weight_stability_summary(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for m, W in weights.items():
        t = weight_turnover_series(W)
        rows[m] = {
            "mean_turnover": t.mean(),
            "p95_turnover": t.quantile(0.95),
            "mean_weight_std": weight_dispersion(W).mean(),
            "mean_HHI": W.apply(lambda r: herfindahl(r.values), axis=1).mean(),
            "mean_effective_N": W.apply(lambda r: 1 / herfindahl(r.values), axis=1).mean(),
            "max_single_weight": W.max().max(),
            "min_single_weight": W.min().min(),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# Test 1: parameter-estimation sensitivity
# ----------------------------------------------------------------------------
def weight_sensitivity_to_sampling(
    model, returns: np.ndarray, n_boot: int = 200, block: int = 10, seed: int = 0
) -> dict:
    """Re-estimate weights on block-bootstrapped samples of the SAME window.

    Reports the cross-sample dispersion of weights and the mean L1 distance
    to the full-sample weights: a data-driven estimate of how much the
    optimal portfolio moves because of sampling noise alone.
    """
    from .evaluation import stationary_bootstrap_indices

    X = np.asarray(returns, dtype=float)
    T = X.shape[0]
    rng = np.random.default_rng(seed)
    w0 = model(X)
    W = []
    for _ in range(n_boot):
        ix = stationary_bootstrap_indices(T, block, rng)
        try:
            W.append(model(X[ix]))
        except Exception:
            continue
    W = np.array(W)
    return {
        "w_full": w0,
        "w_boot_mean": W.mean(axis=0),
        "w_boot_std": W.std(axis=0, ddof=1),
        "mean_L1_distance": float(np.abs(W - w0).sum(axis=1).mean()),
        "mean_turnover_equiv": float(0.5 * np.abs(W - w0).sum(axis=1).mean()),
        "n_success": int(len(W)),
    }


# ----------------------------------------------------------------------------
# Test 2: covariance structure
# ----------------------------------------------------------------------------
def eigen_summary(cov: np.ndarray) -> dict:
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return {
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "condition_number": float(vals[0] / max(vals[-1], 1e-300)),
        "pct_variance_pc1": float(vals[0] / vals.sum()),
        "is_psd": bool(vals[-1] >= -1e-12),
    }


def marchenko_pastur_edges(T: int, N: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Theoretical eigenvalue support of a pure-noise correlation matrix
    when N, T -> inf with N/T = q fixed: sigma2 * (1 ± sqrt(q))^2.
    Eigenvalues of the sample correlation matrix above the upper edge are
    candidates for genuine signal; the rest are statistically
    indistinguishable from noise."""
    q = N / T
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def rolling_covariance_diagnostics(
    returns: pd.DataFrame, window: int = 252, step: int = 21, estimator=ce.sample_covariance
) -> pd.DataFrame:
    """Condition number, top eigenvalue share, mean pairwise correlation over time."""
    rows, idx = [], []
    for i in range(window, len(returns) + 1, step):
        X = returns.iloc[i - window : i].to_numpy()
        S = estimator(X)
        es = eigen_summary(S)
        C = ce.to_correlation(S)
        n = C.shape[0]
        rows.append(
            {
                "condition_number": es["condition_number"],
                "pc1_share": es["pct_variance_pc1"],
                "mean_corr": (C.sum() - n) / (n * (n - 1)),
                "min_eig": es["eigenvalues"][-1],
            }
        )
        idx.append(returns.index[i - 1])
    return pd.DataFrame(rows, index=idx)


def eigenvector_stability(
    returns: pd.DataFrame, window: int = 252, step: int = 21, k: int = 3
) -> pd.DataFrame:
    """|cos angle| between the top-k eigenvectors of consecutive windows.
    Values near 1 = stable principal directions; near 0 = rotating structure."""
    prev, rows, idx = None, [], []
    for i in range(window, len(returns) + 1, step):
        S = ce.sample_covariance(returns.iloc[i - window : i].to_numpy())
        vals, vecs = np.linalg.eigh(S)
        V = vecs[:, np.argsort(vals)[::-1][:k]]
        if prev is not None:
            rows.append({f"pc{j+1}": abs(float(prev[:, j] @ V[:, j])) for j in range(k)})
            idx.append(returns.index[i - 1])
        prev = V
    return pd.DataFrame(rows, index=idx)


def frobenius_change(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """Relative Frobenius-norm change ||A-B||_F / ||A||_F."""
    return float(np.linalg.norm(cov_a - cov_b) / np.linalg.norm(cov_a))


def box_m_test(X1: np.ndarray, X2: np.ndarray) -> dict:
    """Box's M test of equal covariance matrices across two sub-samples.

    Asymptotic chi-square approximation. CAUTION: very sensitive to
    non-normality (heavy tails inflate rejections), so it is reported as
    evidence of instability, not as a decisive test.
    """
    X1, X2 = np.asarray(X1), np.asarray(X2)
    n1, n2 = len(X1), len(X2)
    p = X1.shape[1]
    S1, S2 = np.cov(X1, rowvar=False), np.cov(X2, rowvar=False)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    ld = lambda M: np.linalg.slogdet(M)[1]
    M = (n1 + n2 - 2) * ld(Sp) - (n1 - 1) * ld(S1) - (n2 - 1) * ld(S2)
    c = (
        (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (2 - 1))
        * (1 / (n1 - 1) + 1 / (n2 - 1) - 1 / (n1 + n2 - 2))
    )
    stat = M * (1 - c)
    dof = p * (p + 1) / 2
    return {"statistic": float(stat), "dof": float(dof), "p_value": float(stats.chi2.sf(stat, dof))}


# ----------------------------------------------------------------------------
# Test 5: distributional assumptions
# ----------------------------------------------------------------------------
def univariate_normality_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in returns.columns:
        x = returns[c].dropna().to_numpy()
        jb = stats.jarque_bera(x)
        rows[c] = {
            "mean": x.mean(), "std": x.std(ddof=1),
            "skew": stats.skew(x), "excess_kurtosis": stats.kurtosis(x),
            "JB_stat": jb.statistic, "JB_pvalue": jb.pvalue,
            "min": x.min(), "max": x.max(),
            "n_beyond_4sd": int((np.abs((x - x.mean()) / x.std(ddof=1)) > 4).sum()),
        }
    return pd.DataFrame(rows).T


def mardia_test(returns: pd.DataFrame) -> dict:
    """Mardia's multivariate skewness and kurtosis tests of normality."""
    X = returns.dropna().to_numpy()
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n
    D = Xc @ np.linalg.solve(S, Xc.T)
    b1 = (D**3).sum() / n**2
    b2 = (np.diag(D) ** 2).sum() / n
    skew_stat = n * b1 / 6
    skew_df = p * (p + 1) * (p + 2) / 6
    kurt_z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return {
        "mardia_skewness": float(b1),
        "skew_stat": float(skew_stat),
        "skew_p": float(stats.chi2.sf(skew_stat, skew_df)),
        "mardia_kurtosis": float(b2),
        "kurt_expected_under_normal": float(p * (p + 2)),
        "kurt_z": float(kurt_z),
        "kurt_p": float(2 * stats.norm.sf(abs(kurt_z))),
    }


def tail_dependence_empirical(x: pd.Series, y: pd.Series, q: float = 0.05) -> dict:
    """Empirical lower- and upper-tail co-exceedance probabilities.

    lambda_L(q) = P(Y <= Q_y(q) | X <= Q_x(q)). For a Gaussian copula with
    correlation rho < 1 this tends to 0 as q -> 0; large values signal
    dependence that Gaussian models understate.
    """
    df = pd.concat([x, y], axis=1).dropna()
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    lo = ((a <= a.quantile(q)) & (b <= b.quantile(q))).sum() / max((a <= a.quantile(q)).sum(), 1)
    hi = ((a >= a.quantile(1 - q)) & (b >= b.quantile(1 - q))).sum() / max((a >= a.quantile(1 - q)).sum(), 1)
    return {"lower_tail": float(lo), "upper_tail": float(hi)}


def adf_table(returns: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Augmented Dickey-Fuller tests (requires statsmodels).

    Expect: price levels fail to reject a unit root; returns reject it.
    Stationarity of returns in the ADF sense does not imply stable
    covariances, which is tested separately.
    """
    from statsmodels.tsa.stattools import adfuller

    rows = {}
    for c in returns.columns:
        r = adfuller(returns[c].dropna(), autolag="AIC")
        row = {"ADF_returns_stat": r[0], "ADF_returns_p": r[1]}
        if prices is not None and c in prices:
            p = adfuller(np.log(prices[c].dropna()), autolag="AIC")
            row.update({"ADF_logprice_stat": p[0], "ADF_logprice_p": p[1]})
        rows[c] = row
    return pd.DataFrame(rows).T


def arch_lm_test(x: pd.Series, lags: int = 5) -> dict:
    """Engle's ARCH-LM test for volatility clustering (OLS, no extra deps).

    Regress x_t^2 on its own lags; LM = T * R^2 ~ chi2(lags) under no ARCH.
    """
    e = (x.dropna() - x.dropna().mean()).to_numpy() ** 2
    T = len(e)
    Y = e[lags:]
    X = np.column_stack([np.ones(T - lags)] + [e[lags - j - 1 : T - j - 1] for j in range(lags)])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    r2 = 1 - resid.var() / Y.var()
    lm = (T - lags) * r2
    return {"LM": float(lm), "p_value": float(stats.chi2.sf(lm, lags)), "lags": lags}


# ----------------------------------------------------------------------------
# Composite structural-validity index
# ----------------------------------------------------------------------------
def structural_validity_table(
    weight_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    oos_gap: pd.DataFrame,
    regime_sharpe: pd.DataFrame,
    perturbation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble raw component metrics per model plus a rank-based composite.

    Components (all oriented so LOWER = more structurally valid):
      S1 weight instability   = mean one-way turnover between rebalances
      S2 input sensitivity    = mean turnover-equivalent under bootstrap
      S3 overfitting gap      = in-sample Sharpe - OOS Sharpe
      S4 regime inconsistency = cross-regime std of Sharpe
      S5 perturbation fragility (optional) = mean weight shift under perturbation

    The composite is the mean of within-component ranks (1 = best). It is
    deliberately weight-free so no arbitrary weighting choice drives the
    result; the raw components are always reported alongside it, and the
    composite's sensitivity to dropping a component should be checked.
    """
    comp = pd.DataFrame(
        {
            "S1_weight_instability": weight_summary["mean_turnover"],
            "S2_input_sensitivity": sensitivity["mean_turnover_equiv"],
            "S3_overfitting_gap": oos_gap["gap"],
            "S4_regime_inconsistency": regime_sharpe.groupby("model")["sharpe"].std(ddof=1),
        }
    )
    if perturbation is not None:
        comp["S5_perturbation_fragility"] = perturbation["mean_turnover_equiv"]
    comp = comp.dropna(how="all")
    ranks = comp.rank(ascending=True, method="average")
    comp["composite_rank_score"] = ranks.mean(axis=1)
    return comp.sort_values("composite_rank_score")


def composite_robustness_to_dropping(comp: pd.DataFrame) -> pd.DataFrame:
    """Recompute the composite leaving out each component in turn."""
    cols = [c for c in comp.columns if c.startswith("S")]
    out = {}
    for drop in cols:
        keep = [c for c in cols if c != drop]
        out[f"without_{drop}"] = comp[keep].rank().mean(axis=1)
    return pd.DataFrame(out)
