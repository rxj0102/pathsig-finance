"""Tests for pathsig/augmentations.py."""

import numpy as np
import pytest

from pathsig.augmentations import (
    apply_augmentations,
    cumulative_moving_average_augmentation,
    invisibility_augmentation,
    lead_lag_augmentation,
    time_augmentation,
)
from pathsig.signatures import compute_signature, extract_signature_term


# ---------------------------------------------------------------------------
# time_augmentation
# ---------------------------------------------------------------------------


def test_time_augmentation_shape():
    path = np.ones((5, 3))
    aug = time_augmentation(path)
    assert aug.shape == (5, 4)


def test_time_augmentation_channel_values():
    path = np.zeros((5, 2))
    aug = time_augmentation(path)
    np.testing.assert_allclose(aug[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-12)


def test_time_augmentation_preserves_original():
    path = np.random.default_rng(0).standard_normal((10, 3))
    aug = time_augmentation(path)
    np.testing.assert_array_equal(aug[:, 1:], path)


# ---------------------------------------------------------------------------
# lead_lag_augmentation
# ---------------------------------------------------------------------------


def test_lead_lag_shape():
    path = np.ones((5, 2))
    aug = lead_lag_augmentation(path)
    assert aug.shape == (2 * 5 - 1, 4)


def test_lead_lag_1d_shape():
    path = np.arange(4, dtype=float).reshape(-1, 1)
    aug = lead_lag_augmentation(path)
    assert aug.shape == (7, 2)


def test_lead_lag_realized_variance():
    """Lévy area between lead and lag = realized variance (quadratic variation).

    Correct formula: A_{lead,lag} = S^2_{lead,lag} - S^2_{lag,lead} = QV.
    """
    rng = np.random.default_rng(42)
    returns = rng.standard_normal(50) * 0.01
    path = np.cumsum(returns).reshape(-1, 1)  # price path

    aug = lead_lag_augmentation(path)
    sig = compute_signature(aug, depth=2)

    d_aug = 2
    s_lead_lag = extract_signature_term(sig, d=d_aug, multi_index=(0, 1))
    s_lag_lead = extract_signature_term(sig, d=d_aug, multi_index=(1, 0))
    levy_area = s_lead_lag - s_lag_lead

    # Lévy area = sum of squared increments = realized variance of the price path
    realized_var = float(np.sum(np.diff(path[:, 0]) ** 2))
    np.testing.assert_allclose(levy_area, realized_var, rtol=1e-10)


def test_lead_lag_interleaving():
    """Verify the exact interleaving pattern for a small path."""
    path = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    aug = lead_lag_augmentation(path)
    # Row 0: (X_0, X_0) = (1, 10, 1, 10)
    np.testing.assert_array_equal(aug[0], [1.0, 10.0, 1.0, 10.0])
    # Row 1: lead advances to X_1, lag stays at X_0 = (2, 20, 1, 10)
    np.testing.assert_array_equal(aug[1], [2.0, 20.0, 1.0, 10.0])
    # Row 2: lag catches up to X_1 = (2, 20, 2, 20)
    np.testing.assert_array_equal(aug[2], [2.0, 20.0, 2.0, 20.0])


# ---------------------------------------------------------------------------
# invisibility_augmentation
# ---------------------------------------------------------------------------


def test_invisibility_shape():
    path = np.ones((7, 3))
    aug = invisibility_augmentation(path)
    assert aug.shape == (8, 4)


def test_invisibility_first_row():
    path = np.array([[5.0, 6.0], [7.0, 8.0]])
    aug = invisibility_augmentation(path)
    # First row: visibility=0, position=X_0
    np.testing.assert_array_equal(aug[0], [0.0, 5.0, 6.0])


def test_invisibility_visible_rows():
    path = np.array([[5.0, 6.0], [7.0, 8.0]])
    aug = invisibility_augmentation(path)
    # All rows after first: visibility=1
    np.testing.assert_array_equal(aug[1:, 0], [1.0, 1.0])


# ---------------------------------------------------------------------------
# cumulative_moving_average_augmentation
# ---------------------------------------------------------------------------


def test_cma_shape():
    path = np.ones((6, 2))
    aug = cumulative_moving_average_augmentation(path)
    assert aug.shape == (6, 4)


def test_cma_values():
    path = np.array([[1.0], [2.0], [3.0], [4.0]])
    aug = cumulative_moving_average_augmentation(path)
    expected_cma = np.array([[1.0], [1.5], [2.0], [2.5]])
    np.testing.assert_allclose(aug[:, 1:], expected_cma, atol=1e-12)


# ---------------------------------------------------------------------------
# apply_augmentations
# ---------------------------------------------------------------------------


def test_apply_augmentations_empty():
    path = np.ones((5, 2))
    result = apply_augmentations(path, [])
    np.testing.assert_array_equal(result, path)


def test_apply_augmentations_time():
    path = np.ones((5, 2))
    result = apply_augmentations(path, ["time"])
    assert result.shape == (5, 3)


def test_apply_augmentations_unknown():
    path = np.ones((5, 2))
    with pytest.raises(ValueError, match="Unknown augmentation"):
        apply_augmentations(path, ["nonexistent"])


def test_apply_augmentations_composition():
    """time then lead_lag: first 3 channels → 6 channels in lead-lag."""
    path = np.ones((4, 2))
    result = apply_augmentations(path, ["time", "lead_lag"])
    # After time: (4, 3); after lead_lag: (2*4-1, 6)
    assert result.shape == (7, 6)
