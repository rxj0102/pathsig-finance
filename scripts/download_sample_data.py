"""
Download sample financial data via Yahoo Finance (yfinance).

Usage
-----
    python scripts/download_sample_data.py \
        --tickers AAPL MSFT GOOG AMZN META \
        --start 2015-01-01 --end 2023-12-31 \
        --output data/sample_yahoo.csv

Requirements
------------
    pip install yfinance
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "BAC", "GS",
    "XOM", "CVX", "JNJ", "PFE", "UNH",
]

_DEFAULT_START = "2015-01-01"
_DEFAULT_END   = "2023-12-31"


def download_yahoo(
    tickers: list[str],
    start: str,
    end: str,
    output_path: Path,
) -> pd.DataFrame:
    """Download adjusted close prices and compute panel features.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols.
    start, end:
        Date range (inclusive), format 'YYYY-MM-DD'.
    output_path:
        Path to save the resulting CSV file.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        ``['date', 'ticker', 'log_return', 'realized_vol', 'log_volume', 'dollar_volume']``
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error(
            "yfinance is not installed. Run: pip install yfinance\n"
            "Alternatively, use the synthetic data generators in data/synthetic.py."
        )
        sys.exit(1)

    logger.info("Downloading %d tickers from Yahoo Finance…", len(tickers))
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if raw.empty:
        logger.error("No data returned. Check ticker symbols and date range.")
        sys.exit(1)

    # Handle both single and multi-ticker download shapes
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"]
        volume = raw["Volume"]
    else:
        # Single ticker: reshape to multi-column
        close  = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    records = []
    for ticker in close.columns:
        c = close[ticker].dropna()
        v = volume[ticker].reindex(c.index).fillna(0)

        # Log returns
        lr = np.log(c).diff().dropna()

        for date, ret in lr.items():
            vol_today = v.get(date, 0)
            dollar_vol = float(c.get(date, np.nan)) * float(vol_today)
            records.append(
                {
                    "date": pd.Timestamp(date).date(),
                    "ticker": ticker,
                    "log_return": float(ret),
                    "close": float(c.get(date, np.nan)),
                    "volume": float(vol_today),
                    "dollar_volume": dollar_vol,
                    "log_volume": float(np.log(max(vol_today, 1))),
                }
            )

    panel = pd.DataFrame(records)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Rolling realised volatility (21-day, annualised)
    panel["realized_vol"] = (
        panel.groupby("ticker")["log_return"]
        .transform(lambda x: x.rolling(21, min_periods=10).std() * np.sqrt(252))
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    logger.info(
        "Saved %d rows (%d tickers, %d unique dates) to %s",
        len(panel), panel["ticker"].nunique(), panel["date"].nunique(), output_path,
    )

    return panel


def main():
    parser = argparse.ArgumentParser(
        description="Download sample equity data via Yahoo Finance"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=_DEFAULT_TICKERS,
        help="Ticker symbols (default: 15 large-caps)",
    )
    parser.add_argument("--start", default=_DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=_DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output", default="data/sample_yahoo.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    panel = download_yahoo(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        output_path=Path(args.output),
    )

    print(f"\nDownload complete: {len(panel):,} rows")
    print(f"Columns: {list(panel.columns)}")
    print(f"\nSample:\n{panel.head(3).to_string()}")
    print(
        f"\nTo use with pathsig-finance:\n"
        f"  from data.loader import load_panel_csv\n"
        f"  panel = load_panel_csv('{args.output}')"
    )


if __name__ == "__main__":
    main()
