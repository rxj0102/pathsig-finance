"""Tests for pathsig/evaluation.py."""

import numpy as np
import pytest

from pathsig.evaluation import (
    diebold_mariano_test,
    fama_macbeth_regression,
    newey_west_se,
    out_of_sample_r_squared,
    portfolio_sort_analysis,
    prediction_summary,
)


# ---------------------------------------------------------------------------
# out_of_sample_r_squared
# ---------------------------------------------------------------------------


def test_oos_r2_perfect_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r2 = out_of_sample_r_squared(y, y)
    assert r2 == pytest.approx(1.0)


def test_oos_r2_mean_predictor_is_zero():
    """Predicting the mean as benchmark gives OOS R² = 0."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100)
    # Build expanding mean as both prediction and benchmark
    bench = np.array([y[:i].mean() if i > 0 else 0.0 for i in range(len(y))])
    r2 = out_of_sample_r_squared(bench, y, benchmark=bench)
    assert r2 == pytest.approx(0.0, abs=1e-10)


def test_oos_r2_negative_for_bad_predictor():
    """Constant zero predictor should give negative OOS R² for nonzero mean."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal(100) + 0.5  # positive mean
    predicted = np.zeros_like(y)
    r2 = out_of_sample_r_squared(predicted, y)
    assert r2 < 0.0


def test_oos_r2_ignores_nan():
    y = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    p = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    r2 = out_of_sample_r_squared(p, y)
    assert np.isfinite(r2)


# ---------------------------------------------------------------------------
# newey_west_se
# ---------------------------------------------------------------------------


def test_nw_se_iid_approx_ols():
    """For i.i.d. data, NW SE ≈ OLS SE (both equal σ/√T)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    ols_se = x.std() / np.sqrt(len(x))
    nw_se = newey_west_se(x)
    # Should be in the same ballpark (within 50%)
    assert abs(nw_se - ols_se) / ols_se < 0.5


def test_nw_se_autocorrelated_larger():
    """For strongly autocorrelated data, NW SE ≥ OLS SE."""
    rng = np.random.default_rng(3)
    n = 500
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.standard_normal()
    ols_se = x.std() / np.sqrt(n)
    nw_se = newey_west_se(x)
    assert nw_se > ols_se


def test_nw_se_scalar():
    assert np.isfinite(newey_west_se(np.array([1.0, 2.0, 3.0])))


def test_nw_se_short_series():
    # Series of length 1 → NaN
    result = newey_west_se(np.array([1.0]))
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def test_dm_equal_forecasts_high_pvalue():
    """Two identical forecast error series → p-value near 1 (fail to reject H₀)."""
    rng = np.random.default_rng(4)
    e = rng.standard_normal(200)
    result = diebold_mariano_test(e, e, loss="squared")
    assert result["p_value"] > 0.05


def test_dm_different_forecasts_detects():
    """When model 1 has systematically larger errors, DM should prefer model 2."""
    rng = np.random.default_rng(5)
    e1 = rng.standard_normal(500) * 2.0  # worse model
    e2 = rng.standard_normal(500) * 1.0  # better model
    result = diebold_mariano_test(e1, e2, loss="squared")
    assert result["preferred_model"] == 2
    assert result["p_value"] < 0.05


def test_dm_absolute_loss():
    rng = np.random.default_rng(6)
    e1 = rng.standard_normal(300)
    e2 = rng.standard_normal(300)
    result = diebold_mariano_test(e1, e2, loss="absolute")
    assert np.isfinite(result["statistic"])


def test_dm_unknown_loss():
    with pytest.raises(ValueError, match="Unknown loss"):
        diebold_mariano_test(np.array([1.0]), np.array([1.0]), loss="kl_divergence")


# ---------------------------------------------------------------------------
# Fama-MacBeth regression
# ---------------------------------------------------------------------------


def _make_fm_panel(n_dates=100, n_stocks=50, seed=0):
    rng = np.random.default_rng(seed)
    import pandas as pd

    dates = pd.date_range("2010-01-01", periods=n_dates, freq="B")
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    beta_true = np.array([0.5, -0.3])

    records = []
    for t in range(n_dates):
        X = rng.standard_normal((n_stocks, 2))
        y = X @ beta_true + rng.standard_normal(n_stocks) * 0.1
        for i in range(n_stocks):
            records.append({
                "date": dates[t], "ticker": tickers[i],
                "f1": X[i, 0], "f2": X[i, 1], "ret_forward": y[i],
            })
    return pd.DataFrame(records)


def test_fm_recovers_true_coefficients():
    panel = _make_fm_panel()
    result = fama_macbeth_regression(panel, ["f1", "f2"], return_col="ret_forward")

    assert abs(result["coefficients"]["f1"] - 0.5) < 0.05
    assert abs(result["coefficients"]["f2"] - (-0.3)) < 0.05


def test_fm_result_keys():
    panel = _make_fm_panel(n_dates=30, n_stocks=20)
    result = fama_macbeth_regression(panel, ["f1", "f2"])
    assert "coefficients" in result
    assert "t_statistics" in result
    assert "p_values" in result
    assert "n_dates" in result
    assert "avg_n_stocks" in result


# ---------------------------------------------------------------------------
# Portfolio sort analysis
# ---------------------------------------------------------------------------


def _make_sort_panel(n_dates=100, n_stocks=50, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_dates, freq="B")
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    records = []
    for t in range(n_dates):
        signal = rng.standard_normal(n_stocks)
        ret = 0.5 * signal + rng.standard_normal(n_stocks) * 0.5  # positive relation
        for i in range(n_stocks):
            records.append({
                "date": dates[t], "ticker": tickers[i],
                "signal": signal[i], "return": ret[i],
            })
    return pd.DataFrame(records)


def test_portfolio_sort_positive_ls():
    """With a positive signal-return relation, long-short return should be positive."""
    panel = _make_sort_panel(n_dates=200)
    result = portfolio_sort_analysis(
        features=panel[["date", "signal"]],
        returns=panel["return"],
        n_quantiles=5,
    )
    assert result["mean_ls_ew"] > 0
    assert "t_stat_ew" in result
    assert "sharpe_ew" in result


def test_portfolio_sort_returns_dataframe():
    panel = _make_sort_panel()
    result = portfolio_sort_analysis(
        features=panel[["date", "signal"]],
        returns=panel["return"],
    )
    import pandas as pd
    assert isinstance(result["quantile_returns"], pd.DataFrame)
    assert "Q1" in result["quantile_returns"].columns
    assert "Q5" in result["quantile_returns"].columns


# ---------------------------------------------------------------------------
# prediction_summary
# ---------------------------------------------------------------------------


def test_prediction_summary_perfect():
    import pandas as pd

    n = 100
    y = np.random.default_rng(0).standard_normal(n)
    df = pd.DataFrame({"predicted": y, "realized": y})
    summary = prediction_summary(df)
    assert summary["corr_pearson"] == pytest.approx(1.0)
    assert summary["hit_rate"] == pytest.approx(1.0)


def test_prediction_summary_keys():
    import pandas as pd

    n = 50
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"predicted": rng.standard_normal(n),
                        "realized": rng.standard_normal(n)})
    summary = prediction_summary(df)
    for key in ["oos_r2", "mae", "rmse", "corr_pearson", "corr_spearman", "hit_rate"]:
        assert key in summary
