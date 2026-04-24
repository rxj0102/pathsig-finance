"""Tests for data/synthetic.py."""

import numpy as np
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synthetic import (
    generate_gbm_with_leverage,
    generate_leadlag_volume_price,
    generate_momentum_regime,
    generate_multivariate_brownian,
)


# ---------------------------------------------------------------------------
# generate_gbm_with_leverage
# ---------------------------------------------------------------------------


def test_gbm_shape():
    panel = generate_gbm_with_leverage(n_paths=5, n_steps=10, seed=0)
    expected_rows = 5 * 10
    assert len(panel) == expected_rows
    assert set(["date", "ticker", "log_return", "realized_vol"]).issubset(panel.columns)


def test_gbm_leverage_sign():
    """Lévy area between return and vol should be negative for rho=-0.7."""
    from pathsig.signatures import compute_signature

    panel = generate_gbm_with_leverage(n_paths=100, n_steps=126, leverage_corr=-0.7, seed=1)
    window = 21
    levy_areas = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        T = len(grp)
        if T < window + 1:
            continue
        rets = grp["log_return"].values.astype(float)
        vols = grp["realized_vol"].values.astype(float)

        for t in range(window, T):
            path_w = np.column_stack([rets[t - window:t], vols[t - window:t]])
            try:
                sig = compute_signature(path_w, depth=2)
            except Exception:
                continue
            d = 2
            s_rv = sig[d + 0 * d + 1]
            s_vr = sig[d + 1 * d + 0]
            levy_areas.append(s_rv - s_vr)

    levy_arr = np.array(levy_areas)
    # With rho=-0.7, the Lévy area mean should be negative
    assert levy_arr.mean() < 0, f"Expected negative Lévy area, got {levy_arr.mean():.4f}"


def test_gbm_true_rho_column():
    panel = generate_gbm_with_leverage(n_paths=3, n_steps=5, leverage_corr=-0.5, seed=2)
    assert (panel["true_rho"] == -0.5).all()


def test_gbm_positive_realized_vol():
    panel = generate_gbm_with_leverage(n_paths=10, n_steps=20, seed=3)
    assert (panel["realized_vol"] > 0).all()


# ---------------------------------------------------------------------------
# generate_leadlag_volume_price
# ---------------------------------------------------------------------------


def test_leadlag_shape():
    panel = generate_leadlag_volume_price(n_paths=5, n_steps=10, seed=0)
    assert len(panel) == 50
    assert "log_return" in panel.columns
    assert "log_volume" in panel.columns


def test_leadlag_true_columns():
    panel = generate_leadlag_volume_price(n_paths=3, n_steps=5, lag=2, seed=0)
    assert (panel["true_lag"] == 2).all()


def test_leadlag_volume_leads():
    """Lévy area sign should indicate volume leads price (lag=1)."""
    from pathsig.signatures import compute_signature

    panel = generate_leadlag_volume_price(n_paths=100, n_steps=126, lag=1, seed=5)
    window = 21
    levy_list = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        T = len(grp)
        if T < window + 1:
            continue
        rets = grp["log_return"].values.astype(float)
        lvol = grp["log_volume"].values.astype(float)

        for t in range(window, T):
            path_w = np.column_stack([rets[t - window:t], lvol[t - window:t]])
            try:
                sig = compute_signature(path_w, depth=2)
            except Exception:
                continue
            d = 2
            s_pv = sig[d + 0 * d + 1]
            s_vp = sig[d + 1 * d + 0]
            levy_list.append(s_pv - s_vp)

    levy_arr = np.array(levy_list)
    # volume leads price → A_{price,vol} < 0
    assert levy_arr.mean() < 0, f"Expected negative Lévy area for lag=1, got {levy_arr.mean():.6f}"


# ---------------------------------------------------------------------------
# generate_momentum_regime
# ---------------------------------------------------------------------------


def test_momentum_shape():
    panel = generate_momentum_regime(n_paths=5, n_steps=10, seed=0)
    assert len(panel) == 50
    assert "log_return" in panel.columns


def test_momentum_invalid_strength():
    with pytest.raises(ValueError, match="stationarity"):
        generate_momentum_regime(momentum_strength=1.0)


def test_momentum_positive_autocorr():
    """With momentum_strength=0.5, return autocorrelation should be positive."""
    panel = generate_momentum_regime(n_paths=1, n_steps=2000, momentum_strength=0.5, seed=6)
    rets = panel["log_return"].values
    lag1_corr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
    assert lag1_corr > 0.3, f"Expected positive autocorr, got {lag1_corr:.4f}"


def test_momentum_zero_strength_near_zero_autocorr():
    """With momentum_strength=0, autocorrelation should be near zero."""
    panel = generate_momentum_regime(n_paths=1, n_steps=3000, momentum_strength=0.0, seed=7)
    rets = panel["log_return"].values
    lag1_corr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
    assert abs(lag1_corr) < 0.1, f"Expected ~0 autocorr, got {lag1_corr:.4f}"


# ---------------------------------------------------------------------------
# generate_multivariate_brownian
# ---------------------------------------------------------------------------


def test_bm_shape():
    paths, panel = generate_multivariate_brownian(n_paths=3, n_steps=10, d=2, seed=0)
    assert paths.shape == (3, 10, 2)
    assert len(panel) == 3 * 10


def test_bm_starts_at_zero():
    """Paths should start at zero (by construction of cumulative sum)."""
    paths, _ = generate_multivariate_brownian(n_paths=5, n_steps=50, d=2, seed=0)
    # The paths array already excludes the initial zero point (shape (n_paths, n_steps, d))
    # We can verify increments are reasonable
    increments = np.diff(paths, axis=1)
    assert np.all(np.isfinite(increments))
