"""
Ablation study experiment.

Systematically varies:
    1. Signature depth (1, 2, 3, 4)
    2. Window length (5, 10, 21, 63, 126 days)
    3. Augmentation sets
    4. Channel selection

Reports OOS R² for each configuration and saves results tables.

Usage
-----
    python experiments/run_ablation.py [--config path/to/config.yaml]
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

from data.synthetic import generate_gbm_with_leverage
from pathsig.evaluation import out_of_sample_r_squared
from pathsig.features import SignatureFeatureExtractor
from pathsig.models import CrossSectionalPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def quick_oos_r2(
    panel: pd.DataFrame,
    channels: list[str],
    window: int,
    depth: int,
    augmentations: list[str],
    min_train: int,
    id_col: str,
) -> float:
    """Extract features and run OOS prediction, returning OOS R²."""
    extractor = SignatureFeatureExtractor(
        channels=channels, window=window, depth=depth,
        augmentations=augmentations, normalize=True,
    )
    try:
        features = extractor.fit_transform(panel)
    except Exception as exc:
        logger.warning("Feature extraction failed: %s", exc)
        return np.nan

    feat_cols = [c for c in features.columns if c not in ("date", id_col)]
    if not feat_cols:
        return np.nan

    merged = features.merge(
        panel[["date", id_col, "ret_forward"]].dropna(),
        on=["date", id_col], how="inner",
    ).dropna(subset=feat_cols + ["ret_forward"])

    if len(merged["date"].unique()) < min_train + 10:
        return np.nan

    predictor = CrossSectionalPredictor(model_type="ridge", standardize=True)
    try:
        preds = predictor.expanding_window_predict(
            panel=merged,
            feature_cols=feat_cols,
            return_col="ret_forward",
            min_train_periods=min_train,
            id_col=id_col,
        )
    except Exception as exc:
        logger.warning("Prediction failed: %s", exc)
        return np.nan

    if len(preds) == 0:
        return np.nan

    return out_of_sample_r_squared(preds["predicted"].values, preds["realized"].values)


def main():
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--config", default="experiments/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    syn = cfg["data"]["synthetic"]
    logger.info("Generating synthetic panel…")
    panel = generate_gbm_with_leverage(
        n_paths=syn["n_paths"], n_steps=syn["n_steps"],
        leverage_corr=syn["leverage_corr"], seed=syn["seed"],
    )
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["ret_forward"] = panel.groupby("ticker")["log_return"].shift(-1)
    panel = panel.dropna(subset=["ret_forward"])

    id_col = cfg["data"].get("id_col", "ticker")
    base_channels = cfg["signature"]["channels"]
    min_train = cfg["model"].get("min_train_periods", 252)
    abl_cfg = cfg.get("ablation", {})

    all_results = {}

    # --- 1. Depth ablation ---
    depths = abl_cfg.get("depths", [1, 2, 3])
    logger.info("=== Depth ablation: depths=%s ===", depths)
    depth_results = {}
    for d in depths:
        r2 = quick_oos_r2(panel, base_channels, 21, d, ["time", "lead_lag"], min_train, id_col)
        depth_results[d] = r2
        logger.info("  depth=%d → OOS R²=%.4f%%", d, (r2 or 0) * 100)
    all_results["depth_ablation"] = depth_results

    # --- 2. Window ablation ---
    windows = abl_cfg.get("windows", [5, 10, 21, 63])
    logger.info("=== Window ablation: windows=%s ===", windows)
    window_results = {}
    for w in windows:
        r2 = quick_oos_r2(panel, base_channels, w, 2, ["time", "lead_lag"], min_train, id_col)
        window_results[w] = r2
        logger.info("  window=%d → OOS R²=%.4f%%", w, (r2 or 0) * 100)
    all_results["window_ablation"] = window_results

    # --- 3. Augmentation ablation ---
    aug_sets = abl_cfg.get("augmentation_sets", [[], ["time"], ["lead_lag"], ["time", "lead_lag"]])
    logger.info("=== Augmentation ablation ===")
    aug_results = {}
    for augs in aug_sets:
        key = "+".join(augs) if augs else "none"
        r2 = quick_oos_r2(panel, base_channels, 21, 2, augs, min_train, id_col)
        aug_results[key] = r2
        logger.info("  augs=%s → OOS R²=%.4f%%", key, (r2 or 0) * 100)
    all_results["augmentation_ablation"] = aug_results

    # --- 4. Channel ablation ---
    all_ch = base_channels
    channel_sets = [
        ["log_return"],
        ["log_return", "realized_vol"],
        ["log_return", "log_volume"],
        all_ch,
    ]
    logger.info("=== Channel ablation ===")
    channel_results = {}
    for ch in channel_sets:
        key = "+".join(ch)
        r2 = quick_oos_r2(panel, ch, 21, 2, ["time", "lead_lag"], min_train, id_col)
        channel_results[key] = r2
        logger.info("  channels=%s → OOS R²=%.4f%%", key, (r2 or 0) * 100)
    all_results["channel_ablation"] = channel_results

    # Save and print
    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    print("\n=== ABLATION STUDY RESULTS ===")
    for ablation_name, res in all_results.items():
        print(f"\n[{ablation_name.upper()}]")
        for k, v in res.items():
            print(f"  {k}: {(v or 0)*100:.4f}% OOS R²")


if __name__ == "__main__":
    main()
