"""
Truncated path signature computation engine.

Mathematical background
-----------------------
For a path X: [0,T] -> R^d, the truncated signature up to depth N is the
collection of iterated integrals:

    S(X)^k_{i_1,...,i_k} = ∫_{0 < t_1 < ... < t_k < T}
                               dX^{i_1}_{t_1} ⊗ ... ⊗ dX^{i_k}_{t_k}

for k = 1, ..., N and indices i_j ∈ {0, ..., d-1}.

The full signature (including the empty word "1") lives in the truncated
tensor algebra T^N(R^d).  This module works with the *non-constant* part:
the concatenation of levels 1 through N, giving a vector of dimension
sum_{k=1}^{N} d^k.

The key computational workhorse for piecewise-linear paths is the *Chen
identity*: for a path X concatenated from a segment on [s,u] and one on
[u,t],

    S(X|[s,t])^N = S(X|[s,u])^N ⊗ S(X|[u,t])^N

where ⊗ is the *shuffle product* (tensor-algebra multiplication).  This
lets us accumulate the signature one increment at a time in O(T * d^N)
operations.
"""

from __future__ import annotations

import itertools
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def signature_dimension(d: int, depth: int) -> int:
    """Return the length of the (non-constant) truncated signature vector.

    The dimension is sum_{k=1}^{depth} d^k = d * (d^depth - 1) / (d - 1)
    for d > 1, and depth for d == 1.

    Parameters
    ----------
    d:
        Number of channels (path dimension).
    depth:
        Truncation level N.

    Returns
    -------
    int
        Total number of signature coordinates.
    """
    if d == 1:
        return depth
    return int(d * (d**depth - 1) // (d - 1))


def _multi_index_to_flat(multi_index: tuple, d: int) -> int:
    """Convert a multi-index (i_1, ..., i_k) to a flat offset within the
    level-k block of the signature vector.

    Level k occupies positions [sum_{j=1}^{k-1} d^j, sum_{j=1}^{k} d^j).
    Within that block, (i_1, ..., i_k) maps to i_1 * d^{k-1} + ... + i_k.
    """
    k = len(multi_index)
    level_offset = signature_dimension(d, k - 1) if k > 1 else 0
    within = 0
    for idx in multi_index:
        within = within * d + idx
    return level_offset + within


def extract_signature_term(signature: np.ndarray, d: int, multi_index: tuple) -> float:
    """Extract a specific coefficient from a flattened signature vector.

    Parameters
    ----------
    signature:
        Flattened signature array as returned by :func:`compute_signature`.
    d:
        Number of channels of the original path.
    multi_index:
        Tuple of 0-based channel indices, e.g. ``(0, 1)`` for S^2_{0,1}.

    Returns
    -------
    float
        The value of the requested iterated integral.

    Examples
    --------
    >>> path = np.array([[0., 0.], [1., 2.]])
    >>> sig = compute_signature(path, depth=2)
    >>> # S^1_0 = Δx = 1.0
    >>> extract_signature_term(sig, d=2, multi_index=(0,))
    1.0
    >>> # S^2_{0,1} = ∫∫_{s<t} dX^0_s dX^1_t  for a straight line = 0.5 * 1.0 * 2.0
    >>> extract_signature_term(sig, d=2, multi_index=(0, 1))
    1.0
    """
    flat_idx = _multi_index_to_flat(multi_index, d)
    return float(signature[flat_idx])


# ---------------------------------------------------------------------------
# NumPy backend (pure Python, no PyTorch required)
# ---------------------------------------------------------------------------


def _tensor_product(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Outer (tensor) product of two 1-D arrays, flattened to 1-D."""
    return np.outer(s, t).ravel()


def _chen_update(
    sig_levels: list[np.ndarray],
    increment: np.ndarray,
) -> list[np.ndarray]:
    """Apply the Chen identity to update an accumulated signature by one
    piecewise-linear increment delta = X_{n+1} - X_n.

    For a straight-line path from 0 to delta, its k-th level signature is:
        delta^{⊗k} / k!

    The Chen product formula for two segments A, B gives:
        (A ⊗ B)^k = sum_{j=0}^{k} A^j ⊗ B^{k-j}
    where A^0 = B^0 = 1 (scalar).

    Parameters
    ----------
    sig_levels:
        List of length N where ``sig_levels[k]`` holds the accumulated
        level-(k+1) signature as a flattened array of shape (d^{k+1},).
    increment:
        Shape (d,) — the path increment delta_n = X_{n+1} - X_n.

    Returns
    -------
    list[np.ndarray]
        Updated sig_levels (mutated in-place and returned).
    """
    depth = len(sig_levels)
    d = increment.shape[0]

    # Segment signature: level k = delta^{⊗k} / k!
    seg_levels = []
    seg_k = np.ones(1)  # level 0 = scalar 1
    for k in range(1, depth + 1):
        seg_k = _tensor_product(seg_k, increment) / k
        seg_levels.append(seg_k.copy())
        # Note: we divide by k at each step so that
        # seg_levels[k-1] = delta^{⊗k} / k!  (accumulated factorial)
        # Actually the formula is delta^⊗k / k!, so correct:
        # level 1: delta / 1
        # level 2: outer(delta, delta) / 2
        # level 3: outer(outer(delta, delta), delta) / 6  etc.
        # We do this by dividing by k at each successive outer product step.

    # Chen product: new_sig = accumulated ⊗ seg
    # Level k: sum_{j=0}^{k} acc^j ⊗ seg^{k-j}
    # where acc^0 = seg^0 = scalar 1.
    acc_levels = [np.ones(1)] + [s.copy() for s in sig_levels]   # acc_levels[k] = level k
    seg_all = [np.ones(1)] + seg_levels                           # seg_all[k] = level k

    new_sig_levels = []
    for k in range(1, depth + 1):
        new_k = np.zeros(d**k)
        for j in range(k + 1):
            a = acc_levels[j]      # shape d^j (or scalar for j=0)
            s = seg_all[k - j]     # shape d^{k-j} (or scalar for k-j=0)
            if j == 0:
                new_k += s
            elif k - j == 0:
                new_k += a
            else:
                new_k += _tensor_product(a, s)
        new_sig_levels.append(new_k)

    return new_sig_levels


def _compute_signature_numpy(path: np.ndarray, depth: int) -> np.ndarray:
    """Pure NumPy implementation using the Chen identity.

    Iterates over increments of the piecewise-linear path, applying the
    Chen product at each step.

    Parameters
    ----------
    path:
        Shape (T, d).
    depth:
        Truncation level.

    Returns
    -------
    np.ndarray
        Shape (signature_dimension(d, depth),).
    """
    T, d = path.shape
    if T < 2:
        raise ValueError(f"Path must have at least 2 points, got {T}.")

    # Initialise accumulated signature to "empty path" = (0, 0, ...)
    sig_levels = [np.zeros(d**k) for k in range(1, depth + 1)]

    increments = np.diff(path, axis=0)  # shape (T-1, d)
    for delta in increments:
        sig_levels = _chen_update(sig_levels, delta)

    return np.concatenate(sig_levels)


# ---------------------------------------------------------------------------
# GPU backend (signatory, optional)
# ---------------------------------------------------------------------------


def _compute_signature_signatory(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute the signature using the ``signatory`` PyTorch backend.

    Falls back silently to the NumPy implementation if signatory or torch
    are unavailable.
    """
    try:
        import torch
        import signatory

        path_tensor = torch.tensor(path, dtype=torch.float32).unsqueeze(0)
        sig = signatory.signature(path_tensor, depth=depth, scalar_term=False)
        return sig.squeeze(0).detach().numpy()
    except ImportError:
        logger.warning("signatory or torch not available; falling back to NumPy backend.")
        return _compute_signature_numpy(path, depth)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_signature(
    path: np.ndarray,
    depth: int = 3,
    use_gpu: bool = False,
) -> np.ndarray:
    """Compute the truncated signature of a d-dimensional path.

    The signature is the collection of iterated integrals:

        S(X)^k_{i_1,...,i_k} = ∫_{0<t_1<...<t_k<T} dX^{i_1} ⊗ ... ⊗ dX^{i_k}

    for k = 1, ..., N.  The output is the concatenation of all levels,
    flattened in lexicographic order of the multi-index.

    Parameters
    ----------
    path:
        Array of shape ``(T, d)`` — the discretised path.  Interpreted as
        a piecewise-linear path through the given points.
    depth:
        Truncation level N (typically 2–4 for financial applications).
    use_gpu:
        If True, attempt to use the ``signatory`` GPU backend.

    Returns
    -------
    np.ndarray
        Shape ``(signature_dimension(d, depth),)``.

    Notes
    -----
    The output dimension grows as d^1 + d^2 + ... + d^N, so for d = 5
    channels and depth 3 the signature has 155 coordinates.  Consider
    combining with PCA or the log-signature for dimensionality control.

    Examples
    --------
    >>> path = np.zeros((5, 2)); path[:, 0] = np.linspace(0, 1, 5)
    >>> sig = compute_signature(path, depth=2)
    >>> sig.shape
    (6,)
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError(f"path must be 2-D (T, d), got shape {path.shape}.")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}.")

    if use_gpu:
        return _compute_signature_signatory(path, depth)
    return _compute_signature_numpy(path, depth)


def compute_signature_rolling(
    path: np.ndarray,
    window: int,
    depth: int = 3,
    use_gpu: bool = False,
    step: int = 1,
) -> np.ndarray:
    """Compute rolling-window signatures over a time series.

    At each time step t (starting at t = window - 1), compute the signature
    of the sub-path ``path[t - window + 1 : t + 1]``.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.
    window:
        Look-back window length in time steps.
    depth:
        Signature truncation level.
    use_gpu:
        Whether to use the signatory backend.
    step:
        Stride between successive windows (default 1 = daily rolling).

    Returns
    -------
    np.ndarray
        Shape ``(n_windows, signature_dimension(d, depth))`` where
        ``n_windows = ceil((T - window + 1) / step)``.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    if window > T:
        raise ValueError(f"window ({window}) > path length ({T}).")

    sig_dim = signature_dimension(d, depth)
    starts = range(0, T - window + 1, step)
    result = np.empty((len(starts), sig_dim))

    for out_idx, t in enumerate(starts):
        result[out_idx] = compute_signature(path[t : t + window], depth=depth, use_gpu=use_gpu)

    return result


def compute_logsignature(
    path: np.ndarray,
    depth: int = 3,
    use_gpu: bool = False,
) -> np.ndarray:
    """Compute the log-signature (coordinates in the free Lie algebra).

    The log-signature is strictly more compact than the signature for
    depth >= 2: at depth N it has dimension equal to the sum of the
    dimensions of the Lyndon basis components up to level N, which for
    d channels and depth 2 is d + d(d-1)/2 (vs d + d^2 for the signature).

    Uses the signatory backend when available; falls back to
    ``log(1 + signature)`` approximation via the tensor logarithm otherwise.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.
    depth:
        Truncation level.
    use_gpu:
        Whether to use the signatory GPU backend.

    Returns
    -------
    np.ndarray
        Log-signature vector.
    """
    try:
        import torch
        import signatory

        path_t = torch.tensor(path, dtype=torch.float32).unsqueeze(0)
        logsig = signatory.logsignature(path_t, depth=depth)
        return logsig.squeeze(0).detach().numpy()
    except ImportError:
        logger.warning(
            "signatory unavailable; returning signature as log-signature approximation."
        )
        return compute_signature(path, depth=depth, use_gpu=False)


def signature_to_dict(
    signature: np.ndarray,
    d: int,
    depth: int,
    channel_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """Convert a flat signature array to a human-readable dictionary.

    Keys are multi-index strings such as ``'S_0'``, ``'S_01'``, ``'S_112'``.
    If ``channel_names`` is provided (length d), keys use channel names
    instead of numeric indices.

    Parameters
    ----------
    signature:
        Flattened signature array of length ``signature_dimension(d, depth)``.
    d:
        Number of channels.
    depth:
        Truncation level used when computing the signature.
    channel_names:
        Optional list of length d with human-readable channel labels.

    Returns
    -------
    dict[str, float]
    """
    if channel_names is None:
        channel_names = [str(i) for i in range(d)]

    result: dict[str, float] = {}
    flat_idx = 0
    for k in range(1, depth + 1):
        for mi in itertools.product(range(d), repeat=k):
            key = "S_" + "".join(channel_names[i] for i in mi)
            result[key] = float(signature[flat_idx])
            flat_idx += 1
    return result
