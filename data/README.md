# Data Sources and Setup

This directory contains data loading utilities and synthetic generators.
**No real financial data is committed to the repository.**

## Supported Formats

The panel loader expects a long-format file with the following schema:

| Column | Type | Required | Description |
|---|---|---|---|
| `date` | date | ✓ | Trading date (YYYY-MM-DD) |
| `ticker` / `permno` | str / int | ✓ | Stock identifier |
| `log_return` | float | ✓ | Daily log-return ln(P_t/P_{t-1}) |
| `realized_vol` | float | — | Annualised realised volatility |
| `log_volume` | float | — | Natural log of share volume |
| `dollar_volume` | float | — | Price × volume in USD |
| `mktcap` | float | — | Market capitalisation (for value-weighting) |

## Obtaining Real Data

### Option 1: Yahoo Finance (free, easy)

```bash
python scripts/download_sample_data.py --tickers AAPL MSFT GOOG AMZN META \
    --start 2015-01-01 --end 2023-12-31 --output data/sample_yahoo.csv
```

This downloads adjusted-close prices and computes log-returns and volume.

### Option 2: CRSP (academic, comprehensive)

CRSP (Center for Research in Security Prices) is the standard data source
for academic equity research.  Access via WRDS (Wharton Research Data Services).

The loader expects a CRSP daily stock file with at minimum:
- `DATE`, `PERMNO`, `RET` (daily return), `VOL` (volume), `SHROUT` (shares outstanding)

```python
from data.loader import load_panel_csv

panel = load_panel_csv(
    "crsp_daily.csv",
    start_date="2000-01-01",
    end_date="2022-12-31",
    min_obs_per_stock=252,
)
```

### Option 3: Compustat / Refinitiv / Bloomberg

Any source providing OHLCV data can be converted to the panel format.
The key transformation is computing log-returns:

```python
from data.loader import compute_log_returns, compute_realized_vol

price_panel = ...  # your price data with 'close' column
panel = compute_log_returns(price_panel, price_col="close")
panel = compute_realized_vol(panel, window=21)
```

## Synthetic Data (No download required)

For testing and validation, use the synthetic generators:

```python
from data.synthetic import generate_gbm_with_leverage

panel = generate_gbm_with_leverage(
    n_paths=200, n_steps=252, leverage_corr=-0.7, seed=42
)
```

See `data/synthetic.py` for full documentation of each generator.

## Directory Structure

```
data/
├── __init__.py
├── loader.py         # CSV/Parquet loading and validation
├── synthetic.py      # Synthetic path generators
├── README.md         # This file
├── raw/              # (gitignored) raw downloaded data
└── processed/        # (gitignored) processed panel files
```
