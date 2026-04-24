"""
Interpretability analysis experiment.

For each financial interpretability mapping (momentum, leverage, lead-lag):
    1. Extract the relevant signature term across all stocks and dates.
    2. Compute the corresponding hand-crafted feature.
    3. Report: Pearson/Spearman correlation, time-series of both signals.
    4. Save figures to experiments/figures/.

Usage
-----
    python experiments/run_interpretability.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synthetic import (
    generate_gbm_with_leverage,
    generate_leadlag_volume_price,
    generate_momentum_regime,
)
from pathsig.augmentations import lead_lag_augmentation, time_augmentation
from pathsig.interpretability import (
    full_interpretability_report,
    interpret_level1,
    interpret_level2_cross,
    interpret_price_volume_leadlag,
)
from pathsig.signatures import compute_signature, extract_signature_term

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _try_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="muted")
        return plt
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping figure generation.")
        return None


def analyse_momentum(panel: pd.DataFrame, window: int, fig_dir: Path) -> dict:
    """Compare level-1 signature (cumulative return) with standard momentum."""
    logger.info("Analysing momentum correspondence…")
    channels = ["log_return"]
    results_per_ticker = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        rets = grp["log_return"].values.astype(float)
        T = len(rets)
        if T < window + 1:
            continue

        sig_mom, hc_mom = [], []
        for t in range(window, T):
            path_w = rets[t - window : t].reshape(-1, 1)
            sig = compute_signature(path_w, depth=1)
            sig_mom.append(float(sig[0]))         # S^1_0 = cumulative return
            hc_mom.append(float(rets[t - window : t].sum()))  # same thing

        corr = float(np.corrcoef(sig_mom, hc_mom)[0, 1]) if len(sig_mom) > 1 else np.nan
        results_per_ticker.append({"ticker": ticker, "corr": corr})

    df = pd.DataFrame(results_per_ticker)
    mean_corr = float(df["corr"].mean())
    logger.info("  Mean level-1 ↔ momentum corr: %.6f", mean_corr)

    result = {
        "mapping": "level1 → momentum",
        "expected_corr": 1.0,
        "actual_corr_mean": mean_corr,
        "n_tickers": len(df),
        "note": "Level-1 IS the cumulative return — perfect correspondence by construction.",
    }
    return result


def analyse_leverage(panel: pd.DataFrame, window: int, fig_dir: Path) -> dict:
    """Compare level-2 Lévy area with direct leverage effect estimate."""
    logger.info("Analysing leverage effect correspondence…")
    levy_list, direct_corr_list = [], []

    channels = ["log_return", "realized_vol"]

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if "realized_vol" not in grp.columns:
            continue
        T = len(grp)
        if T < window + 1:
            continue

        rets = grp["log_return"].values.astype(float)
        vols = grp["realized_vol"].values.astype(float)

        for t in range(window, T):
            path_w = np.column_stack([rets[t - window : t], vols[t - window : t]])
            try:
                sig = compute_signature(path_w, depth=2)
            except Exception:
                continue

            d = 2
            s_rv = sig[d + 0 * d + 1]  # S^2_{return, vol}
            s_vr = sig[d + 1 * d + 0]  # S^2_{vol, return}
            levy_list.append(s_rv - s_vr)

            # Direct: corr(return, delta_vol)
            r_w = rets[t - window : t]
            v_w = vols[t - window : t]
            dv = np.diff(v_w)
            if np.std(dv) > 1e-12 and np.std(r_w[1:]) > 1e-12:
                direct_corr_list.append(np.corrcoef(r_w[1:], dv)[0, 1])

    levy_arr = np.array(levy_list)
    corr_arr = np.array(direct_corr_list)

    cross_corr = float(np.corrcoef(levy_arr[:len(corr_arr)], corr_arr)[0, 1]) \
        if len(corr_arr) > 1 else np.nan

    result = {
        "mapping": "levy_area → leverage_effect",
        "levy_area_mean": float(levy_arr.mean()),
        "direct_corr_mean": float(corr_arr.mean()),
        "cross_correlation": cross_corr,
        "n_windows": len(levy_arr),
    }
    logger.info("  Lévy area mean: %.4f, direct corr mean: %.4f, cross-corr: %.4f",
                result["levy_area_mean"], result["direct_corr_mean"], cross_corr)
    return result


def analyse_leadlag(panel: pd.DataFrame, window: int, fig_dir: Path) -> dict:
    """Compare Lévy area direction with known lead-lag structure."""
    logger.info("Analysing price-volume lead-lag correspondence…")
    levy_list = []

    for ticker, grp in panel.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if "log_volume" not in grp.columns:
            continue
        T = len(grp)
        if T < window + 1:
            continue

        rets = grp["log_return"].values.astype(float)
        lvol = grp["log_volume"].values.astype(float)

        for t in range(window, T):
            path_w = np.column_stack([rets[t - window : t], lvol[t - window : t]])
            try:
                sig = compute_signature(path_w, depth=2)
            except Exception:
                continue

            d = 2
            s_pv = sig[d + 0 * d + 1]  # S^2_{price, vol}
            s_vp = sig[d + 1 * d + 0]  # S^2_{vol, price}
            levy_list.append(s_pv - s_vp)

    levy_arr = np.array(levy_list)
    detected = "volume leads price" if levy_arr.mean() < 0 else "price leads volume"

    result = {
        "mapping": "levy_area → price_volume_leadlag",
        "levy_area_mean": float(levy_arr.mean()),
        "detected_direction": detected,
        "n_windows": len(levy_arr),
    }
    logger.info("  Lévy area mean: %.6f → %s", result["levy_area_mean"], detected)
    return result


def main():
    parser = argparse.ArgumentParser(description="Interpretability analysis experiment")
    parser.add_argument("--config", default="experiments/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["results_dir"])
    fig_dir = Path(cfg["output"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt = _try_matplotlib()
    syn = cfg["data"]["synthetic"]
    window = cfg["signature"]["window"]

    # Generate synthetic panel with leverage effect and lead-lag
    panel_lev = generate_gbm_with_leverage(
        n_paths=syn["n_paths"], n_steps=syn["n_steps"],
        leverage_corr=syn["leverage_corr"], seed=syn["seed"],
    )
    panel_lag = generate_leadlag_volume_price(
        n_paths=min(syn["n_paths"], 100), n_steps=syn["n_steps"], lag=1, seed=syn["seed"],
    )

    results = {}
    results["momentum"] = analyse_momentum(panel_lev, window, fig_dir)
    results["leverage"] = analyse_leverage(panel_lev, window, fig_dir)
    results["leadlag"] = analyse_leadlag(panel_lag, window, fig_dir)

    out_path = out_dir / "interpretability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    print("\n=== INTERPRETABILITY ANALYSIS SUMMARY ===")
    for key, val in results.items():
        print(f"\n[{key.upper()}]")
        for k, v in val.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
