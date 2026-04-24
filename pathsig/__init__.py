"""
pathsig-finance: Rough path signatures for financial time series.

Core modules:
    signatures      — truncated signature computation (NumPy + optional GPU)
    augmentations   — path augmentation strategies (time, lead-lag, etc.)
    interpretability — formal mapping: signature terms → financial phenomena
    features        — feature extraction pipeline for cross-sectional prediction
    models          — penalized regression and ML prediction models
    evaluation      — OOS R², Diebold-Mariano, Fama-MacBeth, portfolio sorts
"""

from pathsig.signatures import (
    compute_signature,
    compute_signature_rolling,
    signature_dimension,
    extract_signature_term,
)
from pathsig.augmentations import (
    time_augmentation,
    lead_lag_augmentation,
    invisibility_augmentation,
    cumulative_moving_average_augmentation,
)
from pathsig.features import SignatureFeatureExtractor, BenchmarkFeatureExtractor
from pathsig.interpretability import full_interpretability_report

__version__ = "0.1.0"
__all__ = [
    "compute_signature",
    "compute_signature_rolling",
    "signature_dimension",
    "extract_signature_term",
    "time_augmentation",
    "lead_lag_augmentation",
    "invisibility_augmentation",
    "cumulative_moving_average_augmentation",
    "SignatureFeatureExtractor",
    "BenchmarkFeatureExtractor",
    "full_interpretability_report",
]
