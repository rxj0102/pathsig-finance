"""
Synthetic validation experiment.

For each synthetic generator (leverage, lead-lag, momentum):
1. Generate panel data with known ground-truth parameters.
2. Extract signature features and run the interpretability analysis.
3. Verify that signature terms recover the known parameters.
4. Report: estimated vs true parameter, confidence intervals, p-values.

Usage
-----
    python experiments/run_synthetic_validation.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synthetic import (
    generate_gbm_with_leverage,
    generate_leadlag_volume_price,
    generate_momentum_regime,
)
from pathsig.augmentations import lead_lag_augmentation, time_augmentation
from pathsig.evaluation import newey_west_se
from pathsig.features import SignatureFeatureExtractor
from pathsig.interpretability import (
    interpret_level1,
    interpret_level2_cross,
    interpret_price_volume_leadlag,
    full_interpretability_report,
)
from pathsig.signatures import compute_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leverage effect validation
# ---------------------------------------------------------------------------


def validate_leverage(cfg: dict) -> dict:
    """Verify that the Lévy area recovers the leverage correlation."""
    syn_cfg = cfg["data"]["synthetic"]
    rho_true = syn_cfg["leverage_corr"]

    logger.info("=== Leverage Effect Validation (ρ_true=%.2f) ===", rho_true)
    panel = generate_gbm_with_leverage(
        n_paths=syn_cfg["n_paths"],
        n_steps=syn_cfg["n_steps"],
        leverage_corr=rho_true,
        seed=syn_cfg["seed"],
    )

    sig_cfg = cfg["signature"]
    window = sig_cfg["window"]
    depth = sig_cfg["depth"]
    channels = ["log_return", "realized_vol", "log_volume"]

    levy_areas = []
    direct_corrs = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < window + 1:
            continue

        vals = grp[channels].values.astype(float)
        T = len(vals)

        for t in range(window, T):
            path_window = vals[t - window : t]
            aug = lead_lag_augmentation(time_augmentation(path_window))
            try:
                sig = compute_signature(aug, depth=2)
            except Exception:
                continue

            d_aug = aug.shape[1]
            # Return channel is 1 after time prepend, lag-return is 1+d_aug//2
            # For simplicity, use the raw window's direct cross-term
            ret_w = path_window[:, 0]
            vol_w = path_window[:, 1]
            delta_vol = np.diff(vol_w)
            if np.std(delta_vol) > 1e-12 and np.std(ret_w[1:]) > 1e-12:
                direct_corrs.append(np.corrcoef(ret_w[1:], delta_vol)[0, 1])

            # Lévy area between return and vol channels (no augmentation, raw path)
            raw_sig = compute_signature(path_window, depth=2)
            d_raw = path_window.shape[1]  # 3
            s_rv = raw_sig[d_raw + 0 * d_raw + 1]  # S^2_{0,1}
            s_vr = raw_sig[d_raw + 1 * d_raw + 0]  # S^2_{1,0}
            levy_areas.append(s_rv - s_vr)

    levy_arr = np.array(levy_areas)
    corr_arr = np.array(direct_corrs)

    levy_mean = float(levy_arr.mean())
    levy_se = newey_west_se(levy_arr)
    levy_t = levy_mean / (levy_se + 1e-15)

    corr_mean = float(corr_arr.mean())
    corr_se = float(corr_arr.std() / np.sqrt(len(corr_arr)))

    result = {
        "test": "leverage_effect",
        "true_rho": rho_true,
        "levy_area_mean": levy_mean,
        "levy_area_se": levy_se,
        "levy_area_t_stat": levy_t,
        "levy_sign_correct": (levy_mean < 0) == (rho_true < 0),
        "direct_corr_mean": corr_mean,
        "direct_corr_se": corr_se,
        "n_windows": len(levy_arr),
    }

    logger.info("  Lévy area mean: %.4f (SE: %.4f, t=%.2f)", levy_mean, levy_se, levy_t)
    logger.info("  Direct corr mean: %.4f (SE: %.4f)", corr_mean, corr_se)
    logger.info("  Sign correct: %s", result["levy_sign_correct"])
    return result


# ---------------------------------------------------------------------------
# Lead-lag validation
# ---------------------------------------------------------------------------


def validate_leadlag(cfg: dict) -> dict:
    """Verify that the Lévy area recovers the volume→price lead-lag direction."""
    syn_cfg = cfg["data"]["synthetic"]
    true_lag = 1  # hardcoded for this validation

    logger.info("=== Lead-Lag Validation (true_lag=%d) ===", true_lag)
    panel = generate_leadlag_volume_price(
        n_paths=syn_cfg["n_paths"],
        n_steps=syn_cfg["n_steps"],
        lag=true_lag,
        seed=syn_cfg["seed"],
    )

    channels = ["log_return", "log_volume"]
    window = cfg["signature"]["window"]
    depth = 2

    levy_areas = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) < window + 1:
            continue

        vals = grp[channels].values.astype(float)
        T = len(vals)

        for t in range(window, T):
            path_w = vals[t - window : t]
            try:
                sig = compute_signature(path_w, depth=depth)
            except Exception:
                continue

            d = path_w.shape[1]  # 2
            s_rv = sig[d + 0 * d + 1]  # S^2_{price, volume}
            s_vr = sig[d + 1 * d + 0]  # S^2_{volume, price}
            levy_areas.append(s_rv - s_vr)

    levy_arr = np.array(levy_areas)
    levy_mean = float(levy_arr.mean())
    levy_se = newey_west_se(levy_arr)

    # When volume leads price, S^2_{vol, price} > S^2_{price, vol}
    # → Lévy area A_{price, vol} < 0 → volume leads
    detected_direction = "volume leads price" if levy_mean < 0 else "price leads volume"
    correct = detected_direction == "volume leads price"  # true_lag > 0 → vol leads

    result = {
        "test": "lead_lag",
        "true_lag": true_lag,
        "levy_area_mean": levy_mean,
        "levy_area_se": levy_se,
        "detected_direction": detected_direction,
        "direction_correct": correct,
        "n_windows": len(levy_arr),
    }

    logger.info("  Lévy area mean: %.6f (SE: %.6f)", levy_mean, levy_se)
    logger.info("  Detected: %s | Correct: %s", detected_direction, correct)
    return result


# ---------------------------------------------------------------------------
# Momentum validation
# ---------------------------------------------------------------------------


def validate_momentum(cfg: dict) -> dict:
    """Verify that level-1 signatures predict future returns in momentum paths."""
    syn_cfg = cfg["data"]["synthetic"]
    true_mom = syn_cfg["momentum_strength"]

    logger.info("=== Momentum Validation (true_α=%.3f) ===", true_mom)
    panel = generate_momentum_regime(
        n_paths=syn_cfg["n_paths"],
        n_steps=syn_cfg["n_steps"],
        momentum_strength=true_mom,
        seed=syn_cfg["seed"],
    )

    # Compute level-1 signature (= cumulative return) for each window
    window = cfg["signature"]["window"]
    channels = ["log_return"]

    pairs_x, pairs_y = [], []
    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        T = len(grp)
        if T < window + 1:
            continue
        rets = grp["log_return"].values.astype(float)

        for t in range(window, T - 1):
            # Build log-price path starting at 0; S^1 = X_T - X_0 = cumulative return.
            # Using raw return values as a path would give S^1 = last_ret - first_ret ≠ cumret.
            ret_w = rets[t - window : t]
            price_w = np.concatenate([[0.0], np.cumsum(ret_w)]).reshape(-1, 1)
            sig = compute_signature(price_w, depth=1)
            cum_ret = sig[0]  # = cumulative log-return (exact identity)
            next_ret = rets[t]
            pairs_x.append(cum_ret)
            pairs_y.append(next_ret)

    x = np.array(pairs_x)
    y = np.array(pairs_y)

    # OLS: y = α + β * x
    X_mat = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
    alpha_hat, beta_hat = float(beta[0]), float(beta[1])

    # Theoretical: since r_t = μ * r_{t-1} + ε and CumRet = Σ_{s=t-w}^{t-1} r_s,
    # E[r_t | CumRet] ≈ μ * r_{t-1}.  The regression coeff is approximately
    # μ / w (roughly), so we check sign agreement.
    result = {
        "test": "momentum",
        "true_momentum": true_mom,
        "ols_intercept": alpha_hat,
        "ols_beta": beta_hat,
        "beta_sign_correct": (beta_hat > 0) == (true_mom > 0),
        "n_obs": len(x),
    }

    logger.info("  OLS β=%.6f (true sign: %s)", beta_hat, "positive" if true_mom > 0 else "negative")
    logger.info("  Sign correct: %s", result["beta_sign_correct"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Synthetic validation experiments")
    parser.add_argument("--config", default="experiments/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["leverage"] = validate_leverage(cfg)
    results["leadlag"] = validate_leadlag(cfg)
    results["momentum"] = validate_momentum(cfg)

    out_path = out_dir / "synthetic_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    # Summary table
    print("\n=== SYNTHETIC VALIDATION SUMMARY ===")
    print(f"{'Test':<25} {'Pass':<8} {'Key metric'}")
    print("-" * 60)
    lev = results["leverage"]
    print(f"{'Leverage sign':<25} {'PASS' if lev['levy_sign_correct'] else 'FAIL':<8} "
          f"Lévy area={lev['levy_area_mean']:.4f} (true ρ={lev['true_rho']:.2f})")
    ll = results["leadlag"]
    print(f"{'Lead-lag direction':<25} {'PASS' if ll['direction_correct'] else 'FAIL':<8} "
          f"Lévy area={ll['levy_area_mean']:.6f}")
    mom = results["momentum"]
    print(f"{'Momentum sign':<25} {'PASS' if mom['beta_sign_correct'] else 'FAIL':<8} "
          f"β={mom['ols_beta']:.6f} (true α={mom['true_momentum']:.3f})")


if __name__ == "__main__":
    main()
