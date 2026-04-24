"""
Feature extraction pipeline for cross-sectional equity return prediction.

Two extractors are provided:

``SignatureFeatureExtractor``
    Computes rolling-window truncated signatures of multivariate financial
    time series (log-return, realised vol, log-volume) with configurable
    augmentations.  Produces a panel DataFrame of signature features per
    stock-date.

``BenchmarkFeatureExtractor``
    Computes standard hand-crafted factor features (momentum, short-term
    reversal, realised vol, Amihud illiquidity, leverage effect proxy,
    volume trend) for direct comparison with signature features.
"""

from __future__ import annotations

import itertools
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pathsig.augmentations import apply_augmentations
from pathsig.signatures import (
    compute_signature,
    compute_signature_rolling,
    signature_dimension,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signature feature extractor
# ---------------------------------------------------------------------------


class SignatureFeatureExtractor:
    """Extract rolling-window signature features from a financial panel dataset.

    For each stock at each date t, the extractor:
        1. Takes the last ``window`` rows of the multivariate channel data
           (log_return, realized_vol, log_volume by default).
        2. Applies any requested augmentations (time, lead_lag, …).
        3. Computes the truncated signature up to ``depth``.
        4. Optionally cross-sectionally standardises the resulting features.

    Parameters
    ----------
    channels:
        Column names in the input DataFrame to use as path channels.
    window:
        Look-back window length in trading days (default 21 ≈ 1 month).
    depth:
        Signature truncation level.  Depth 2 gives O(d^2) features;
        depth 3 gives O(d^3) — use PCA for high d or deep level.
    augmentations:
        Ordered list of augmentation names applied before signature
        computation.  Choices: ``'time'``, ``'lead_lag'``, ``'invisibility'``,
        ``'cma'``.
    normalize:
        If True, standardise features cross-sectionally at each date.
    use_gpu:
        If True, use the signatory GPU backend.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> n = 100
    >>> df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=n),
    ...                    'ticker': 'AAPL',
    ...                    'log_return': np.random.randn(n) * 0.01,
    ...                    'realized_vol': np.abs(np.random.randn(n)) * 0.01,
    ...                    'log_volume': np.random.randn(n) + 10})
    >>> extractor = SignatureFeatureExtractor(window=21, depth=2)
    >>> features = extractor.fit_transform(df)
    >>> features.shape[1] > 3
    True
    """

    def __init__(
        self,
        channels: list[str] = None,
        window: int = 21,
        depth: int = 3,
        augmentations: list[str] = None,
        normalize: bool = True,
        use_gpu: bool = False,
    ) -> None:
        self.channels = channels or ["log_return", "realized_vol", "log_volume"]
        self.window = window
        self.depth = depth
        self.augmentations = augmentations if augmentations is not None else ["time", "lead_lag"]
        self.normalize = normalize
        self.use_gpu = use_gpu

        # Computed lazily in fit_transform
        self._feature_names: Optional[list[str]] = None
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augmented_d(self, d: int) -> int:
        """Return the number of channels after augmentation."""
        aug_d = d
        for aug in self.augmentations:
            if aug == "time":
                aug_d += 1
            elif aug in ("lead_lag",):
                aug_d *= 2
            elif aug in ("cma",):
                aug_d *= 2
            elif aug == "invisibility":
                aug_d += 1
        return aug_d

    def _compute_feature_names(self) -> list[str]:
        """Generate human-readable feature names with multi-index notation."""
        d_orig = len(self.channels)
        d_aug = self._augmented_d(d_orig)

        # Build augmented channel name list
        ch = list(self.channels)
        for aug in self.augmentations:
            if aug == "time":
                ch = ["t"] + ch
            elif aug == "lead_lag":
                ch = [f"lead_{c}" for c in ch] + [f"lag_{c}" for c in ch]
            elif aug == "cma":
                ch = ch + [f"cma_{c}" for c in ch]
            elif aug == "invisibility":
                ch = ["vis"] + ch

        names = []
        for k in range(1, self.depth + 1):
            for mi in itertools.product(range(d_aug), repeat=k):
                idx_str = ",".join(ch[i] for i in mi)
                names.append(f"sig{k}[{idx_str}]")
        return names

    def _extract_one(self, window_data: np.ndarray) -> np.ndarray:
        """Compute signature features for a single window."""
        aug_data = apply_augmentations(window_data, self.augmentations)
        return compute_signature(aug_data, depth=self.depth, use_gpu=self.use_gpu)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, panel_data: pd.DataFrame) -> pd.DataFrame:
        """Compute signature features for all stocks and dates.

        Parameters
        ----------
        panel_data:
            Long-format panel with at minimum columns ``['date', 'ticker']``
            (or ``'permno'``) plus the channel columns.  Rows must be sorted
            by (ticker, date).

        Returns
        -------
        pd.DataFrame
            Columns: ``['date', 'ticker']`` + signature feature columns.
            Dates without a full window of history are omitted.
        """
        required = {"date"} | set(self.channels)
        if not required.issubset(panel_data.columns):
            missing = required - set(panel_data.columns)
            raise ValueError(f"panel_data missing columns: {missing}")

        id_col = "ticker" if "ticker" in panel_data.columns else "permno"
        if id_col not in panel_data.columns:
            raise ValueError("panel_data must contain a 'ticker' or 'permno' column.")

        self._feature_names = self._compute_feature_names()
        records = []

        for entity, grp in panel_data.sort_values(["date"]).groupby(id_col):
            grp = grp.sort_values("date").reset_index(drop=True)
            vals = grp[self.channels].values.astype(float)
            dates = grp["date"].values
            T = len(grp)

            if T < self.window:
                logger.debug("Entity %s has fewer rows (%d) than window (%d); skipping.",
                             entity, T, self.window)
                continue

            for t in range(self.window - 1, T):
                window_data = vals[t - self.window + 1 : t + 1]
                try:
                    sig = self._extract_one(window_data)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Signature failed for %s at %s: %s", entity, dates[t], exc)
                    continue
                row = {id_col: entity, "date": dates[t]}
                for name, val in zip(self._feature_names, sig):
                    row[name] = val
                records.append(row)

        if not records:
            return pd.DataFrame()

        result = pd.DataFrame(records)

        if self.normalize:
            feat_cols = self._feature_names
            for date, grp in result.groupby("date"):
                idx = grp.index
                scaler = StandardScaler()
                result.loc[idx, feat_cols] = scaler.fit_transform(grp[feat_cols].values)

        return result

    def get_feature_names(self) -> list[str]:
        """Return the human-readable feature name list (available after fit_transform)."""
        if self._feature_names is None:
            self._feature_names = self._compute_feature_names()
        return self._feature_names

    @property
    def n_features(self) -> int:
        """Number of signature features produced."""
        d_aug = self._augmented_d(len(self.channels))
        return signature_dimension(d_aug, self.depth)


# ---------------------------------------------------------------------------
# Benchmark feature extractor
# ---------------------------------------------------------------------------


class BenchmarkFeatureExtractor:
    """Extract standard hand-crafted factor features for comparison.

    Features computed per stock-date:

    ``momentum``
        Cumulative log-return over the window (Jegadeesh & Titman 1993).
    ``short_reversal``
        Previous day's log-return (Jegadeesh 1990).
    ``realized_vol``
        Rolling annualised realised volatility = sqrt(sum(r^2) * 252 / window).
    ``vol_trend``
        OLS slope of log(realised_vol) over the window (vol momentum).
    ``leverage_proxy``
        Pearson correlation of daily returns with Δ(realized_vol) over the window.
        Negative values indicate the leverage effect.
    ``amihud``
        Amihud (2002) illiquidity ratio = mean(|r_t| / dollar_volume_t).
    ``volume_trend``
        OLS slope of log_volume over the window.
    ``skewness``
        Third standardised moment of daily returns.
    ``excess_kurtosis``
        Fourth standardised moment minus 3.

    Parameters
    ----------
    window:
        Look-back window length in trading days.
    """

    def __init__(self, window: int = 21) -> None:
        self.window = window

    def _features_one(
        self,
        returns: np.ndarray,
        vol: Optional[np.ndarray],
        log_volume: Optional[np.ndarray],
        dollar_volume: Optional[np.ndarray],
    ) -> dict:
        w = self.window
        out: dict = {}

        # ---- Momentum ----
        out["momentum"] = float(np.sum(returns))  # cumulative log-return

        # ---- Short-term reversal ----
        out["short_reversal"] = float(returns[-1]) if len(returns) > 0 else np.nan

        # ---- Realised volatility ----
        rv = float(np.sqrt(np.sum(returns**2) * 252 / w))
        out["realized_vol"] = rv

        # ---- Skewness & excess kurtosis ----
        r_demeaned = returns - returns.mean()
        r_std = returns.std() + 1e-12
        out["skewness"] = float(np.mean((r_demeaned / r_std) ** 3))
        out["excess_kurtosis"] = float(np.mean((r_demeaned / r_std) ** 4) - 3.0)

        # ---- Leverage proxy ----
        if vol is not None and len(vol) >= 2:
            delta_vol = np.diff(vol)
            ret_aligned = returns[1:]  # align with delta_vol
            if np.std(delta_vol) > 1e-12 and np.std(ret_aligned) > 1e-12:
                corr = float(np.corrcoef(ret_aligned, delta_vol)[0, 1])
            else:
                corr = np.nan
            out["leverage_proxy"] = corr
        else:
            out["leverage_proxy"] = np.nan

        # ---- Volume trend ----
        if log_volume is not None:
            t = np.arange(len(log_volume), dtype=float)
            if len(t) >= 2 and np.std(log_volume) > 1e-12:
                slope = float(np.polyfit(t, log_volume, 1)[0])
            else:
                slope = 0.0
            out["volume_trend"] = slope
        else:
            out["volume_trend"] = np.nan

        # ---- Amihud illiquidity ----
        if dollar_volume is not None and len(dollar_volume) > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                illiq = np.where(
                    dollar_volume > 0,
                    np.abs(returns) / dollar_volume,
                    np.nan,
                )
            out["amihud"] = float(np.nanmean(illiq)) * 1e6  # scale up
        else:
            out["amihud"] = np.nan

        return out

    def fit_transform(self, panel_data: pd.DataFrame) -> pd.DataFrame:
        """Compute benchmark features for all stocks and dates.

        Parameters
        ----------
        panel_data:
            Long-format panel with at minimum columns
            ``['date', 'ticker'/'permno', 'log_return']``.
            Optional columns: ``'realized_vol'``, ``'log_volume'``,
            ``'dollar_volume'``.

        Returns
        -------
        pd.DataFrame
            Columns: ``['date', 'ticker']`` + feature columns.
        """
        id_col = "ticker" if "ticker" in panel_data.columns else "permno"
        if id_col not in panel_data.columns:
            raise ValueError("panel_data must contain a 'ticker' or 'permno' column.")
        if "log_return" not in panel_data.columns:
            raise ValueError("panel_data must contain a 'log_return' column.")

        has_vol = "realized_vol" in panel_data.columns
        has_lv = "log_volume" in panel_data.columns
        has_dv = "dollar_volume" in panel_data.columns

        records = []
        for entity, grp in panel_data.sort_values(["date"]).groupby(id_col):
            grp = grp.sort_values("date").reset_index(drop=True)
            dates = grp["date"].values
            rets = grp["log_return"].values.astype(float)
            vol_arr = grp["realized_vol"].values.astype(float) if has_vol else None
            lv_arr = grp["log_volume"].values.astype(float) if has_lv else None
            dv_arr = grp["dollar_volume"].values.astype(float) if has_dv else None
            T = len(grp)

            for t in range(self.window - 1, T):
                sl = slice(t - self.window + 1, t + 1)
                row = {id_col: entity, "date": dates[t]}
                feats = self._features_one(
                    rets[sl],
                    vol_arr[sl] if vol_arr is not None else None,
                    lv_arr[sl] if lv_arr is not None else None,
                    dv_arr[sl] if dv_arr is not None else None,
                )
                row.update(feats)
                records.append(row)

        return pd.DataFrame(records) if records else pd.DataFrame()
