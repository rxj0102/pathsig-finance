"""
Statistical evaluation of cross-sectional equity return predictions.

Functions
---------
out_of_sample_r_squared    — Campbell & Thompson (2008) OOS R²
diebold_mariano_test       — Diebold & Mariano (1995) equal-accuracy test
newey_west_se              — HAC standard error for a time series mean
fama_macbeth_regression    — Fama & MacBeth (1973) two-pass regression
portfolio_sort_analysis    — Quantile portfolio long-short returns & t-stats

All standard errors use Newey-West (HAC) corrections to account for
serial correlation in financial panel data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OOS R²
# ---------------------------------------------------------------------------


def out_of_sample_r_squared(
    predicted: np.ndarray,
    realized: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
) -> float:
    """Compute the out-of-sample R² of Campbell & Thompson (2008).

    OOS R² = 1 - SS_model / SS_benchmark

    where:
        SS_model     = sum_t (r_t - r_hat_t)^2
        SS_benchmark = sum_t (r_t - r_bar_t)^2
        r_bar_t      = expanding historical mean up to t (if benchmark is None)

    In cross-sectional equity prediction, OOS R² is typically very small
    (0.1% – 2%) but economically meaningful: Goyal & Welch (2008) show
    that even 0.5% OOS R² translates to economically large portfolio returns.

    Parameters
    ----------
    predicted:
        Model predicted returns, shape (N,).
    realized:
        Realised returns, shape (N,).
    benchmark:
        Benchmark predictions (e.g. historical mean).  If None, uses the
        expanding mean of ``realized`` as the naive benchmark.

    Returns
    -------
    float
        OOS R².  Negative values indicate the model is worse than the
        historical-mean benchmark.
    """
    predicted = np.asarray(predicted, dtype=float)
    realized = np.asarray(realized, dtype=float)

    mask = np.isfinite(predicted) & np.isfinite(realized)
    predicted, realized = predicted[mask], realized[mask]

    if benchmark is None:
        # Expanding mean of realised returns (naive benchmark)
        benchmark = np.array([realized[:i].mean() if i > 0 else 0.0
                               for i in range(len(realized))])
    else:
        benchmark = np.asarray(benchmark, dtype=float)[mask]

    ss_model = np.sum((realized - predicted) ** 2)
    ss_bench = np.sum((realized - benchmark) ** 2)

    if ss_bench == 0.0:
        return np.nan

    return float(1.0 - ss_model / ss_bench)


# ---------------------------------------------------------------------------
# Newey-West HAC standard errors
# ---------------------------------------------------------------------------


def newey_west_se(x: np.ndarray, lags: Optional[int] = None) -> float:
    """Compute the Newey-West HAC standard error for the sample mean of x.

    The HAC variance estimator is:

        V_HAC = γ_0 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * γ_j

    where γ_j = (1/T) sum_{t=j+1}^{T} (x_t - x̄)(x_{t-j} - x̄) and
    the default lag truncation is L = floor(4 * (T/100)^{2/9}).

    The standard error of the mean is sqrt(V_HAC / T).

    Parameters
    ----------
    x:
        Time series of shape (T,).
    lags:
        HAC lag truncation L.  Default follows the Andrews (1991) rule:
        floor(4 * (T/100)^{2/9}).

    Returns
    -------
    float
        Newey-West HAC standard error of mean(x).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 2:
        return np.nan

    if lags is None:
        lags = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))

    xc = x - x.mean()  # centred
    gamma0 = np.dot(xc, xc) / T

    V_hac = gamma0
    for j in range(1, lags + 1):
        gamma_j = np.dot(xc[j:], xc[:-j]) / T
        weight = 1.0 - j / (lags + 1.0)  # Bartlett kernel
        V_hac += 2.0 * weight * gamma_j

    V_hac = max(V_hac, 0.0)  # numerical stability
    return float(np.sqrt(V_hac / T))


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
) -> dict:
    """Diebold-Mariano (1995) test for equal predictive accuracy.

    Tests H₀: E[d_t] = 0  where  d_t = L(e₁_t) - L(e₂_t),
    i.e., the two models have equal expected loss.

    Uses Newey-West HAC standard errors with bandwidth h-1 (appropriate
    for h-step-ahead forecasts to account for MA(h-1) error structure).

    Parameters
    ----------
    errors_1, errors_2:
        Forecast errors for model 1 and model 2.  Positive values mean
        the model over-predicted the realised value.
    h:
        Forecast horizon in periods.  Used to set the HAC bandwidth.
    loss:
        Loss function: ``'squared'`` (MSE-based) or ``'absolute'`` (MAE-based).

    Returns
    -------
    dict with keys:
        ``'statistic'``: DM test statistic.
        ``'p_value'``: two-sided p-value (asymptotically standard normal).
        ``'mean_loss_diff'``: sample mean of d_t (negative = model 1 better).
        ``'nw_se'``: Newey-West SE of the mean loss differential.
        ``'preferred_model'``: 1 or 2 depending on which has lower expected loss.
    """
    e1 = np.asarray(errors_1, dtype=float)
    e2 = np.asarray(errors_2, dtype=float)

    if loss == "squared":
        L1 = e1 ** 2
        L2 = e2 ** 2
    elif loss == "absolute":
        L1 = np.abs(e1)
        L2 = np.abs(e2)
    else:
        raise ValueError(f"Unknown loss '{loss}'. Choose 'squared' or 'absolute'.")

    d = L1 - L2
    mask = np.isfinite(d)
    d = d[mask]

    if len(d) < 5:
        return {"statistic": np.nan, "p_value": np.nan, "mean_loss_diff": np.nan,
                "nw_se": np.nan, "preferred_model": None}

    mean_d = d.mean()
    nw_se = newey_west_se(d, lags=max(0, h - 1))

    if nw_se < 1e-15:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = mean_d / nw_se
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_loss_diff": float(mean_d),
        "nw_se": float(nw_se),
        "preferred_model": 1 if mean_d < 0 else 2,
    }


# ---------------------------------------------------------------------------
# Fama-MacBeth regression
# ---------------------------------------------------------------------------


def fama_macbeth_regression(
    panel: pd.DataFrame,
    feature_cols: list[str],
    return_col: str = "ret_forward",
    date_col: str = "date",
    nw_lags: Optional[int] = None,
) -> dict:
    """Fama-MacBeth (1973) two-pass cross-sectional regression.

    Pass 1 — Cross-sectional regression for each date t:
        r_{i,t} = α_t + Σ_k β_{k,t} * f_{k,i,t} + ε_{i,t}

    Pass 2 — Time-series average of the slope coefficients:
        β_k = (1/T) Σ_t β_{k,t}
        with Newey-West t-statistics

    Parameters
    ----------
    panel:
        Long-format panel DataFrame.
    feature_cols:
        Predictor column names.
    return_col:
        Target variable column name.
    date_col:
        Date column name.
    nw_lags:
        HAC lag truncation (default: Andrews rule).

    Returns
    -------
    dict with keys:
        ``'coefficients'``: dict mapping feature name → mean coefficient.
        ``'t_statistics'``: dict mapping feature name → NW t-stat.
        ``'p_values'``: dict mapping feature name → two-sided p-value.
        ``'n_dates'``: number of cross-sectional regressions run.
        ``'avg_n_stocks'``: average cross-sectional sample size.
        ``'intercept'``: time-series mean of the cross-sectional intercept.
        ``'intercept_t_stat'``: Newey-West t-stat for the intercept.
    """
    from sklearn.linear_model import LinearRegression

    dates = sorted(panel[date_col].unique())
    betas = []  # list of dicts: {feature: coef, 'intercept': coef}
    n_stocks_list = []

    for t in dates:
        grp = panel[panel[date_col] == t].dropna(subset=feature_cols + [return_col])
        if len(grp) < len(feature_cols) + 2:
            continue
        X = grp[feature_cols].values
        y = grp[return_col].values
        model = LinearRegression().fit(X, y)
        beta_t = dict(zip(feature_cols, model.coef_))
        beta_t["intercept"] = model.intercept_
        betas.append(beta_t)
        n_stocks_list.append(len(grp))

    if not betas:
        raise ValueError("No dates had sufficient observations for Fama-MacBeth regression.")

    beta_df = pd.DataFrame(betas)
    cols = list(feature_cols) + ["intercept"]
    result: dict = {
        "coefficients": {},
        "t_statistics": {},
        "p_values": {},
        "n_dates": len(betas),
        "avg_n_stocks": float(np.mean(n_stocks_list)),
    }

    for col in cols:
        series = beta_df[col].dropna().values
        mean_b = float(series.mean())
        se = newey_west_se(series, lags=nw_lags)
        if se < 1e-15:
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = float(mean_b / se)
            p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

        if col == "intercept":
            result["intercept"] = mean_b
            result["intercept_t_stat"] = t_stat
            result["intercept_p_value"] = p_val
        else:
            result["coefficients"][col] = mean_b
            result["t_statistics"][col] = t_stat
            result["p_values"][col] = p_val

    return result


# ---------------------------------------------------------------------------
# Portfolio sort analysis
# ---------------------------------------------------------------------------


def portfolio_sort_analysis(
    features: pd.DataFrame,
    returns: pd.Series,
    n_quantiles: int = 5,
    date_col: str = "date",
    signal_col: str = "signal",
    value_weights: Optional[pd.Series] = None,
    nw_lags: Optional[int] = None,
) -> dict:
    """Standard quantile portfolio sort test.

    At each date t:
        1. Sort stocks into ``n_quantiles`` groups by signal strength.
        2. Compute equal-weighted (and optionally value-weighted) portfolio
           return for each quantile.

    Then compute the long-short (top-minus-bottom quantile) portfolio and
    report its mean return with Newey-West t-statistic.

    Parameters
    ----------
    features:
        DataFrame with columns [date_col, signal_col] (and optionally a
        stock-id column).  The signal_col is the sorting variable.
    returns:
        Series of realised returns aligned with ``features`` by index.
    n_quantiles:
        Number of quantile portfolios.
    date_col:
        Date column name in ``features``.
    signal_col:
        Column in ``features`` to sort on.
    value_weights:
        Optional Series of market-cap weights aligned with ``features``.
        If provided, value-weighted returns are also computed.
    nw_lags:
        HAC lag truncation for Newey-West t-statistics.

    Returns
    -------
    dict with keys:
        ``'quantile_returns'``: pd.DataFrame, index=date, cols=Q1..QN.
        ``'long_short_ew'``: pd.Series of long-short equal-weighted returns.
        ``'mean_ls_ew'``: mean of long-short series.
        ``'t_stat_ew'``: Newey-West t-stat for mean long-short return.
        ``'p_value_ew'``: two-sided p-value.
        ``'sharpe_ew'``: annualised Sharpe ratio of long-short portfolio.
        ``'long_short_vw'``: value-weighted long-short (if weights given).
    """
    df = features[[date_col, signal_col]].copy()
    df["return"] = returns.values
    if value_weights is not None:
        df["weight"] = value_weights.values

    dates = sorted(df[date_col].unique())
    quantile_rets: dict[str, list] = {f"Q{q+1}": [] for q in range(n_quantiles)}
    date_index = []

    for t in dates:
        grp = df[df[date_col] == t].dropna(subset=[signal_col, "return"])
        if len(grp) < n_quantiles:
            continue
        grp = grp.copy()
        grp["quantile"] = pd.qcut(
            grp[signal_col], q=n_quantiles, labels=False, duplicates="drop"
        )
        for q in range(n_quantiles):
            bucket = grp[grp["quantile"] == q]
            quantile_rets[f"Q{q+1}"].append(bucket["return"].mean())

        date_index.append(t)

    q_df = pd.DataFrame(quantile_rets, index=date_index)
    ls_ew = q_df[f"Q{n_quantiles}"] - q_df["Q1"]

    mean_ls = float(ls_ew.mean())
    se = newey_west_se(ls_ew.dropna().values, lags=nw_lags)
    t_stat = mean_ls / se if se > 1e-15 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

    ls_std = float(ls_ew.std())
    sharpe = (mean_ls / ls_std * np.sqrt(252)) if ls_std > 1e-12 else np.nan

    result: dict = {
        "quantile_returns": q_df,
        "long_short_ew": ls_ew,
        "mean_ls_ew": mean_ls,
        "t_stat_ew": float(t_stat),
        "p_value_ew": float(p_val),
        "sharpe_ew": float(sharpe),
        "n_dates": len(date_index),
    }

    if value_weights is not None:
        ls_vw_list = []
        for t in date_index:
            grp = df[df[date_col] == t].dropna(subset=[signal_col, "return", "weight"])
            if len(grp) < n_quantiles:
                ls_vw_list.append(np.nan)
                continue
            grp = grp.copy()
            grp["quantile"] = pd.qcut(
                grp[signal_col], q=n_quantiles, labels=False, duplicates="drop"
            )
            top = grp[grp["quantile"] == n_quantiles - 1]
            bot = grp[grp["quantile"] == 0]

            def _vw_ret(g: pd.DataFrame) -> float:
                w = g["weight"].values
                r = g["return"].values
                tot_w = w.sum()
                return float(np.dot(w, r) / tot_w) if tot_w > 0 else np.nan

            ls_vw_list.append(_vw_ret(top) - _vw_ret(bot))

        ls_vw = pd.Series(ls_vw_list, index=date_index)
        result["long_short_vw"] = ls_vw
        mean_vw = float(ls_vw.mean())
        se_vw = newey_west_se(ls_vw.dropna().values, lags=nw_lags)
        result["mean_ls_vw"] = mean_vw
        result["t_stat_vw"] = float(mean_vw / se_vw) if se_vw > 1e-15 else 0.0

    return result


# ---------------------------------------------------------------------------
# Utility: summarise prediction errors
# ---------------------------------------------------------------------------


def prediction_summary(
    predictions: pd.DataFrame,
    date_col: str = "date",
    predicted_col: str = "predicted",
    realized_col: str = "realized",
) -> dict:
    """Compute a concise set of prediction quality metrics.

    Parameters
    ----------
    predictions:
        DataFrame with columns [date_col, predicted_col, realized_col].

    Returns
    -------
    dict with keys:
        ``'oos_r2'``, ``'mae'``, ``'rmse'``, ``'corr_pearson'``,
        ``'corr_spearman'``, ``'hit_rate'``.
    """
    pred = predictions[predicted_col].values
    real = predictions[realized_col].values
    mask = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[mask], real[mask]

    errors = pred - real
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    corr_p = float(np.corrcoef(pred, real)[0, 1]) if len(pred) > 1 else np.nan
    corr_s = float(stats.spearmanr(pred, real).correlation) if len(pred) > 1 else np.nan
    hit = float(np.mean(np.sign(pred) == np.sign(real)))

    return {
        "oos_r2": out_of_sample_r_squared(pred, real),
        "mae": mae,
        "rmse": rmse,
        "corr_pearson": corr_p,
        "corr_spearman": corr_s,
        "hit_rate": hit,
        "n_obs": int(mask.sum()),
    }
