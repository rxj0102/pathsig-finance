"""
Data loading and synthetic generation utilities for pathsig-finance.

Submodules
----------
loader     — CSV/Parquet interface for real financial panel data
synthetic  — Synthetic path generators with known mathematical properties
"""

from data.loader import load_panel_csv, load_panel_parquet
from data.synthetic import (
    generate_gbm_with_leverage,
    generate_leadlag_volume_price,
    generate_momentum_regime,
)

__all__ = [
    "load_panel_csv",
    "load_panel_parquet",
    "generate_gbm_with_leverage",
    "generate_leadlag_volume_price",
    "generate_momentum_regime",
]
