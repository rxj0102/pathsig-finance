"""
Data loading utilities for real financial panel data.

Supports CSV and Parquet formats.  The expected panel schema is:

    date        : date (YYYY-MM-DD) or parseable datetime
    ticker      : str  — stock identifier (or 'permno' for CRSP)
    log_return  : float — daily log-return ln(P_t / P_{t-1})
    realized_vol: float — annualised realised volatility estimate (optional)
    log_volume  : float — log of share/dollar volume (optional)
    dollar_volume: float — price * volume in dollars (optional)
    mktcap      : float — market capitalisation for value-weighting (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"date", "log_return"}
_OPTIONAL_COLS = {"realized_vol", "log_volume", "dollar_volume", "mktcap"}
_ID_COLS = {"ticker", "permno"}


def _validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce panel DataFrame to canonical schema."""
    missing_req = _REQUIRED_COLS - set(df.columns)
    if missing_req:
        raise ValueError(f"Panel is missing required columns: {missing_req}")

    if not _ID_COLS.intersection(df.columns):
        raise ValueError(
            f"Panel must have a stock identifier column: one of {_ID_COLS}."
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["log_return"] = pd.to_numeric(df["log_return"], errors="coerce")

    for col in _OPTIONAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove infinite / wildly unrealistic log-returns (|r| > 5 = 500% daily move)
    extreme = np.abs(df["log_return"]) > 5.0
    if extreme.any():
        logger.warning(
            "Dropping %d rows with |log_return| > 5 (likely data errors).", extreme.sum()
        )
        df = df[~extreme]

    df = df.sort_values(["date"]).reset_index(drop=True)
    return df


def load_panel_csv(
    path: Union[str, Path],
    columns: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_obs_per_stock: int = 0,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a financial panel from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.
    columns:
        Subset of columns to load (default: all).
    start_date, end_date:
        Optional date-range filter (inclusive).
    min_obs_per_stock:
        Drop stocks with fewer than this many observations.
    **read_csv_kwargs:
        Passed to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        Validated, sorted panel.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    logger.info("Loading panel from CSV: %s", path)
    df = pd.read_csv(path, usecols=columns, **read_csv_kwargs)
    df = _validate_panel(df)

    df = _apply_filters(df, start_date, end_date, min_obs_per_stock)
    logger.info("Loaded %d rows, %d unique dates.", len(df), df["date"].nunique())
    return df


def load_panel_parquet(
    path: Union[str, Path],
    columns: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_obs_per_stock: int = 0,
) -> pd.DataFrame:
    """Load a financial panel from a Parquet file.

    Parameters
    ----------
    path:
        Path to the Parquet file (or directory for partitioned datasets).
    columns:
        Subset of columns to load.
    start_date, end_date:
        Optional date-range filter.
    min_obs_per_stock:
        Drop stocks with fewer than this many observations.

    Returns
    -------
    pd.DataFrame
        Validated, sorted panel.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info("Loading panel from Parquet: %s", path)
    df = pd.read_parquet(path, columns=columns)
    df = _validate_panel(df)

    df = _apply_filters(df, start_date, end_date, min_obs_per_stock)
    logger.info("Loaded %d rows, %d unique dates.", len(df), df["date"].nunique())
    return df


def _apply_filters(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
    min_obs_per_stock: int,
) -> pd.DataFrame:
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    if min_obs_per_stock > 0:
        id_col = "ticker" if "ticker" in df.columns else "permno"
        counts = df.groupby(id_col).size()
        keep = counts[counts >= min_obs_per_stock].index
        before = len(df)
        df = df[df[id_col].isin(keep)]
        logger.info(
            "Dropped %d rows for stocks with < %d obs.", before - len(df), min_obs_per_stock
        )

    return df.reset_index(drop=True)


def compute_log_returns(
    price_panel: pd.DataFrame,
    price_col: str = "close",
    id_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute daily log-returns from a price panel.

    Parameters
    ----------
    price_panel:
        Panel with columns [date_col, id_col, price_col].
    price_col:
        Name of the price column.
    id_col:
        Stock identifier column.
    date_col:
        Date column.

    Returns
    -------
    pd.DataFrame
        Same panel with an added ``'log_return'`` column.
    """
    df = price_panel.copy().sort_values([id_col, date_col])
    df["log_return"] = df.groupby(id_col)[price_col].transform(
        lambda x: np.log(x).diff()
    )
    return df.dropna(subset=["log_return"]).reset_index(drop=True)


def compute_realized_vol(
    panel: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
    id_col: str = "ticker",
    date_col: str = "date",
    return_col: str = "log_return",
) -> pd.DataFrame:
    """Append a rolling realised-volatility column to a panel.

    Parameters
    ----------
    panel:
        Panel with at least [date_col, id_col, return_col].
    window:
        Rolling window length.
    annualize:
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.DataFrame
        Panel with added ``'realized_vol'`` column.
    """
    df = panel.copy().sort_values([id_col, date_col])
    scale = np.sqrt(252) if annualize else 1.0
    df["realized_vol"] = (
        df.groupby(id_col)[return_col]
        .transform(lambda x: x.rolling(window, min_periods=window // 2).std())
        * scale
    )
    return df
