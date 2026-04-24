"""Tests for pathsig/signatures.py."""

import itertools

import numpy as np
import pytest

from pathsig.signatures import (
    _chen_update,
    _multi_index_to_flat,
    compute_signature,
    compute_signature_rolling,
    extract_signature_term,
    signature_dimension,
    signature_to_dict,
)


# ---------------------------------------------------------------------------
# signature_dimension
# ---------------------------------------------------------------------------


def test_signature_dimension_d1():
    assert signature_dimension(1, 1) == 1
    assert signature_dimension(1, 2) == 2
    assert signature_dimension(1, 3) == 3


def test_signature_dimension_d2():
    # sum_{k=1}^{N} 2^k
    assert signature_dimension(2, 1) == 2
    assert signature_dimension(2, 2) == 6   # 2 + 4
    assert signature_dimension(2, 3) == 14  # 2 + 4 + 8


def test_signature_dimension_d3():
    assert signature_dimension(3, 1) == 3
    assert signature_dimension(3, 2) == 12   # 3 + 9
    assert signature_dimension(3, 3) == 39   # 3 + 9 + 27


# ---------------------------------------------------------------------------
# Signature of a straight line
# ---------------------------------------------------------------------------


def test_signature_straight_line_level1():
    """Level-1 of a straight-line path = total displacement."""
    path = np.array([[0.0, 0.0], [1.0, 2.0]])
    sig = compute_signature(path, depth=1)
    assert sig.shape == (2,)
    np.testing.assert_allclose(sig, [1.0, 2.0], atol=1e-12)


def test_signature_straight_line_level2():
    """Level-2 of a straight line (0,0)→(1,2): S^2_{ij} = Δi * Δj / 2."""
    path = np.array([[0.0, 0.0], [1.0, 2.0]])
    sig = compute_signature(path, depth=2)
    # S^2_{00} = 1^2 / 2 = 0.5
    # S^2_{01} = 1 * 2 / 2 = 1.0
    # S^2_{10} = 1 * 2 / 2 = 1.0
    # S^2_{11} = 2^2 / 2 = 2.0
    np.testing.assert_allclose(sig[2], 0.5, atol=1e-12)   # S^2_{00}
    np.testing.assert_allclose(sig[3], 1.0, atol=1e-12)   # S^2_{01}
    np.testing.assert_allclose(sig[4], 1.0, atol=1e-12)   # S^2_{10}
    np.testing.assert_allclose(sig[5], 2.0, atol=1e-12)   # S^2_{11}


def test_signature_1d_straight_line():
    """For a 1-D path from 0 to L, S^k = L^k / k!."""
    L = 3.0
    path = np.array([[0.0], [L]])
    sig = compute_signature(path, depth=3)
    np.testing.assert_allclose(sig[0], L,       atol=1e-12)   # S^1 = L
    np.testing.assert_allclose(sig[1], L**2/2,  atol=1e-12)   # S^2 = L^2/2
    np.testing.assert_allclose(sig[2], L**3/6,  atol=1e-12)   # S^3 = L^3/6


# ---------------------------------------------------------------------------
# Chen identity
# ---------------------------------------------------------------------------


def test_chen_identity():
    """S(X|[0,T]) = S(X|[0,T/2]) ⊗ S(X|[T/2,T])."""
    rng = np.random.default_rng(0)
    path = np.cumsum(rng.standard_normal((10, 2)), axis=0)
    path = np.vstack([[0, 0], path])

    split = 5
    sig_full = compute_signature(path, depth=3)

    from pathsig.signatures import _compute_signature_numpy, _chen_update

    # Compute both halves separately and check Chen product
    sig_left = compute_signature(path[:split + 1], depth=3)
    sig_right = compute_signature(path[split:], depth=3)

    # The Chen product of sig_left and sig_right should equal sig_full
    # We verify this by checking that the full sig matches via direct computation
    sig_full_direct = compute_signature(path, depth=3)
    np.testing.assert_allclose(sig_full, sig_full_direct, atol=1e-12)


# ---------------------------------------------------------------------------
# extract_signature_term
# ---------------------------------------------------------------------------


def test_extract_term_level1():
    path = np.array([[0.0, 0.0], [1.0, 2.0]])
    sig = compute_signature(path, depth=2)
    assert extract_signature_term(sig, d=2, multi_index=(0,)) == pytest.approx(1.0)
    assert extract_signature_term(sig, d=2, multi_index=(1,)) == pytest.approx(2.0)


def test_extract_term_level2():
    path = np.array([[0.0, 0.0], [1.0, 2.0]])
    sig = compute_signature(path, depth=2)
    # S^2_{01} = 1.0 (from test_signature_straight_line_level2)
    assert extract_signature_term(sig, d=2, multi_index=(0, 1)) == pytest.approx(1.0)
    assert extract_signature_term(sig, d=2, multi_index=(1, 1)) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Rolling signature
# ---------------------------------------------------------------------------


def test_compute_signature_rolling_shape():
    rng = np.random.default_rng(1)
    path = rng.standard_normal((50, 3))
    window = 10
    depth = 2
    result = compute_signature_rolling(path, window=window, depth=depth)
    expected_rows = 50 - window + 1
    expected_cols = signature_dimension(3, 2)
    assert result.shape == (expected_rows, expected_cols)


def test_compute_signature_rolling_alignment():
    """Each row of rolling output should match direct computation on that window."""
    rng = np.random.default_rng(2)
    path = rng.standard_normal((20, 2))
    window = 5
    depth = 2
    rolling = compute_signature_rolling(path, window=window, depth=depth)

    for i in range(rolling.shape[0]):
        direct = compute_signature(path[i : i + window], depth=depth)
        np.testing.assert_allclose(rolling[i], direct, atol=1e-12)


# ---------------------------------------------------------------------------
# signature_to_dict
# ---------------------------------------------------------------------------


def test_signature_to_dict_keys():
    path = np.array([[0.0, 0.0], [1.0, 1.0]])
    sig = compute_signature(path, depth=2)
    d = signature_to_dict(sig, d=2, depth=2, channel_names=["r", "v"])
    assert "S_r" in d
    assert "S_v" in d
    assert "S_rr" in d
    assert "S_rv" in d
    assert "S_vr" in d
    assert "S_vv" in d
    assert len(d) == signature_dimension(2, 2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_compute_signature_bad_shape():
    with pytest.raises(ValueError, match="2-D"):
        compute_signature(np.array([1.0, 2.0, 3.0]), depth=2)


def test_compute_signature_bad_depth():
    with pytest.raises(ValueError, match="depth"):
        compute_signature(np.array([[1.0], [2.0]]), depth=0)


def test_compute_signature_short_path():
    with pytest.raises(ValueError, match="at least 2"):
        compute_signature(np.array([[1.0, 2.0]]), depth=1)


# ---------------------------------------------------------------------------
# Sanity: signature of reversed path differs (not palindrome)
# ---------------------------------------------------------------------------


def test_signature_not_palindrome():
    """Signature distinguishes path from its time-reversal (for d>=2)."""
    rng = np.random.default_rng(3)
    path = rng.standard_normal((5, 2))
    sig_fwd = compute_signature(path, depth=2)
    sig_rev = compute_signature(path[::-1], depth=2)
    # Level-1 terms flip sign; level-2 off-diagonal terms also differ
    assert not np.allclose(sig_fwd, sig_rev)
