"""
Main cross-sectional prediction experiment.

Runs expanding-window OOS prediction comparing:
    a) Signature features only
    b) Benchmark (hand-crafted) features only
    c) Combined: signatures + benchmarks

Outputs:
    - OOS R² for each model
    - Diebold-Mariano tests (signature vs benchmark)
    - Fama-MacBeth regression results
    - Portfolio sort analysis (quintile long-short returns)

Results saved to experiments/results/.

Usage
-----
    python experiments/run_cross_sectional.py [--config path/to/config.yaml]
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
from pathsig.evaluation import (
    diebold_mariano_test,
    fama_macbeth_regression,
    out_of_sample_r_squared,
    portfolio_sort_analysis,
    prediction_summary,
)
from pathsig.features import BenchmarkFeatureExtractor, SignatureFeatureExtractor
from pathsig.models import CrossSectionalPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(cfg: dict) -> pd.DataFrame:
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        syn = cfg["data"]["synthetic"]
        logger.info("Generating synthetic panel data…")
        panel = generate_gbm_with_leverage(
            n_paths=syn["n_paths"],
            n_steps=syn["n_steps"],
            leverage_corr=syn["leverage_corr"],
            seed=syn["seed"],
        )
        # Add forward return column (next-day return for each stock)
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
        panel["ret_forward"] = panel.groupby("ticker")["log_return"].shift(-1)
        return panel.dropna(subset=["ret_forward"]).reset_index(drop=True)
    elif mode == "real":
        from data.loader import load_panel_parquet, load_panel_csv
        path = Path(cfg["data"]["panel_path"])
        if path.suffix == ".parquet":
            panel = load_panel_parquet(
                path,
                start_date=cfg["data"].get("start_date"),
                end_date=cfg["data"].get("end_date"),
                min_obs_per_stock=cfg["data"].get("min_obs_per_stock", 0),
            )
        else:
            panel = load_panel_csv(
                path,
                start_date=cfg["data"].get("start_date"),
                end_date=cfg["data"].get("end_date"),
                min_obs_per_stock=cfg["data"].get("min_obs_per_stock", 0),
            )
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
        panel["ret_forward"] = panel.groupby("ticker")["log_return"].shift(-1)
        return panel.dropna(subset=["ret_forward"]).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown data mode '{mode}'.")


def extract_features(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig_cfg = cfg["signature"]

    sig_extractor = SignatureFeatureExtractor(
        channels=sig_cfg["channels"],
        window=sig_cfg["window"],
        depth=sig_cfg["depth"],
        augmentations=sig_cfg.get("augmentations", ["time", "lead_lag"]),
        normalize=sig_cfg.get("normalize", True),
        use_gpu=sig_cfg.get("use_gpu", False),
    )
    bench_extractor = BenchmarkFeatureExtractor(window=sig_cfg["window"])

    logger.info("Extracting signature features…")
    sig_features = sig_extractor.fit_transform(panel)

    logger.info("Extracting benchmark features…")
    bench_features = bench_extractor.fit_transform(panel)

    return sig_features, bench_features


def run_prediction(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    cfg: dict,
) -> pd.DataFrame:
    """Run expanding-window OOS prediction for a given feature set."""
    id_col = cfg["data"].get("id_col", "ticker")
    model_cfg = cfg["model"]

    # Merge features with forward returns
    merged = features.merge(
        panel[["date", id_col, "ret_forward"]].dropna(),
        on=["date", id_col],
        how="inner",
    )
    merged = merged.dropna(subset=feature_cols + ["ret_forward"])

    predictor = CrossSectionalPredictor(
        model_type=model_cfg["type"],
        standardize=model_cfg.get("standardize", True),
    )

    logger.info("Running OOS prediction: %s (%d feature cols)…", label, len(feature_cols))
    preds = predictor.expanding_window_predict(
        panel=merged,
        feature_cols=feature_cols,
        return_col="ret_forward",
        min_train_periods=model_cfg.get("min_train_periods", 252),
        id_col=id_col,
    )
    return preds


def main():
    parser = argparse.ArgumentParser(description="Cross-sectional prediction experiment")
    parser.add_argument("--config", default="experiments/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    id_col = cfg["data"].get("id_col", "ticker")

    # 1. Load data
    panel = load_data(cfg)
    logger.info("Panel: %d rows, %d stocks, %d dates.",
                len(panel), panel[id_col].nunique(), panel["date"].nunique())

    # 2. Extract features
    sig_features, bench_features = extract_features(panel, cfg)

    sig_feat_cols = [c for c in sig_features.columns if c not in ("date", id_col)]
    bench_feat_cols = [c for c in bench_features.columns if c not in ("date", id_col)]

    # 3. Run OOS predictions for all three model variants
    pred_sig = run_prediction(sig_features, panel, sig_feat_cols, "signature", cfg)
    pred_bench = run_prediction(bench_features, panel, bench_feat_cols, "benchmark", cfg)

    # Combined features
    combined = sig_features.merge(bench_features, on=["date", id_col], how="inner")
    combined_cols = sig_feat_cols + bench_feat_cols
    pred_combined = run_prediction(combined, panel, combined_cols, "combined", cfg)

    # 4. Evaluate
    eval_cfg = cfg["evaluation"]
    results: dict = {}

    for label, preds in [("signature", pred_sig), ("benchmark", pred_bench),
                          ("combined", pred_combined)]:
        summary = prediction_summary(preds)
        results[label] = summary
        logger.info("%s OOS R²: %.4f%%, hit rate: %.3f",
                    label, summary["oos_r2"] * 100, summary["hit_rate"])

    # 5. Diebold-Mariano: signature vs benchmark
    common_idx = pred_sig.merge(
        pred_bench, on=["date", id_col], suffixes=("_sig", "_bench")
    )
    if len(common_idx) > 0:
        e_sig = common_idx["realized_sig"].values - common_idx["predicted_sig"].values
        e_bench = common_idx["realized_bench"].values - common_idx["predicted_bench"].values
        dm = diebold_mariano_test(e_sig, e_bench, loss=eval_cfg.get("dm_loss", "squared"))
        results["dm_test_sig_vs_bench"] = dm
        logger.info("DM test (sig vs bench): stat=%.3f, p=%.3f, preferred=%s",
                    dm["statistic"], dm["p_value"], dm.get("preferred_model"))

    # 6. Fama-MacBeth on signature features
    try:
        merged_for_fm = sig_features.merge(
            panel[["date", id_col, "ret_forward"]].dropna(), on=["date", id_col], how="inner"
        ).dropna(subset=sig_feat_cols[:5] + ["ret_forward"])

        # Use first 5 sig features to keep FM tractable
        fm = fama_macbeth_regression(
            merged_for_fm, sig_feat_cols[:5], return_col="ret_forward",
            nw_lags=eval_cfg.get("nw_lags"),
        )
        results["fama_macbeth_sig"] = {
            "coefficients": fm["coefficients"],
            "t_statistics": fm["t_statistics"],
            "p_values": fm["p_values"],
            "n_dates": fm["n_dates"],
        }
        logger.info("Fama-MacBeth: %d dates, avg %d stocks.", fm["n_dates"], int(fm["avg_n_stocks"]))
    except Exception as exc:
        logger.warning("Fama-MacBeth failed: %s", exc)

    # 7. Portfolio sort on the strongest signature feature
    try:
        best_sig_col = sig_feat_cols[0]  # first feature; could rank by FM t-stat
        merged_ps = sig_features.merge(
            panel[["date", id_col, "ret_forward"]].dropna(), on=["date", id_col], how="inner"
        ).dropna(subset=[best_sig_col, "ret_forward"])

        ps = portfolio_sort_analysis(
            features=merged_ps[["date", best_sig_col]].rename(columns={best_sig_col: "signal"}),
            returns=merged_ps["ret_forward"],
            n_quantiles=eval_cfg.get("n_quantiles", 5),
            nw_lags=eval_cfg.get("nw_lags"),
        )
        results["portfolio_sort"] = {
            "mean_ls_ew": ps["mean_ls_ew"],
            "t_stat_ew": ps["t_stat_ew"],
            "p_value_ew": ps["p_value_ew"],
            "sharpe_ew": ps["sharpe_ew"],
            "n_dates": ps["n_dates"],
        }
        logger.info("Portfolio sort: mean L/S=%.4f%%, t=%.2f, Sharpe=%.2f",
                    ps["mean_ls_ew"] * 100, ps["t_stat_ew"], ps["sharpe_ew"])
    except Exception as exc:
        logger.warning("Portfolio sort failed: %s", exc)

    # 8. Save results
    out_path = out_dir / "cross_sectional_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {k: (v if isinstance(v, dict) else str(v)) for k, v in results.items()},
            f, indent=2, default=str,
        )
    logger.info("Results saved to %s", out_path)

    # Summary table
    print("\n=== CROSS-SECTIONAL PREDICTION SUMMARY ===")
    print(f"{'Model':<15} {'OOS R²':<12} {'Hit Rate':<12} {'Pearson r'}")
    print("-" * 55)
    for label in ["signature", "benchmark", "combined"]:
        r = results.get(label, {})
        print(f"{label:<15} {r.get('oos_r2', 0)*100:>8.4f}%   "
              f"{r.get('hit_rate', 0):>8.3f}    "
              f"{r.get('corr_pearson', 0):>8.4f}")


if __name__ == "__main__":
    main()
