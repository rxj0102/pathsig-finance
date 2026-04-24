"""
Path augmentation strategies for financial time series.

Raw price paths are invariant to time reparameterisation under the signature
map — two paths that traverse the same points at different speeds have
identical signatures.  Augmentations add channels that break this invariance
and encode financially relevant structure.

Each augmentation is a map  path: (T, d) -> augmented_path: (T', d')
that can be composed before signature computation.
"""

from __future__ import annotations

import numpy as np


def time_augmentation(path: np.ndarray) -> np.ndarray:
    """Prepend a linearly increasing time channel normalised to [0, 1].

    Breaks time-reparameterisation invariance: two paths traversing the
    same points at different speeds will now have different signatures,
    because the time channel explicitly encodes when each move occurs.

    The level-2 signature cross-term  S^2_{time, channel_i} captures
    whether channel i tends to move early or late in the window — a
    form of intra-window timing signal.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.

    Returns
    -------
    np.ndarray
        Shape ``(T, d + 1)``.  Column 0 is the time channel in [0, 1];
        columns 1..d are the original channels.

    Examples
    --------
    >>> p = np.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> time_augmentation(p).shape
    (3, 3)
    >>> time_augmentation(p)[:, 0]   # time channel
    array([0. , 0.5, 1. ])
    """
    path = np.asarray(path, dtype=float)
    T = path.shape[0]
    time_col = np.linspace(0.0, 1.0, T).reshape(-1, 1)
    return np.concatenate([time_col, path], axis=1)


def lead_lag_augmentation(path: np.ndarray) -> np.ndarray:
    """Lead-lag transform: interleave each point with its lagged copy.

    For a 1-D path (X_0, X_1, ..., X_T), the lead-lag path is the 2-D path:
        (X_0, X_0), (X_1, X_0), (X_1, X_1), (X_2, X_1), ..., (X_T, X_{T-1})

    generalised to d channels: the lead channels are the original d channels;
    the lag channels are the same channels delayed by one step.

    Mathematical property
    --------------------
    The *Lévy area* between lead channel i and lag channel i equals the
    quadratic variation (realised variance) of channel i:

        A_{lead_i, lag_i} = S^2_{lead_i, lag_i} - S^2_{lag_i, lead_i}
                          = sum_k (ΔX^i_k)^2
                          = [X^i, X^i]_T   (quadratic variation)

    Proof: for each step k the lead-lag path sweeps a right-angle turn in
    the (lead, lag) plane.  The Lévy area of that turn equals (ΔX^i_k)^2,
    and summing over steps gives the full quadratic variation.

    Note: the individual term S^2_{lead_i, lag_i} = (1/2)*CumRet_i^2 +
    (1/2)*QV_i, so to isolate QV one must use the antisymmetric combination
    (the Lévy area).  This provides a signature-based realised-variance
    estimator requiring no extra volatility channel.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.

    Returns
    -------
    np.ndarray
        Shape ``(2*T - 1, 2*d)``.  First d columns = lead; last d = lag.

    References
    ----------
    Flint & Lyons (2016). Pathwise currency hedging. arXiv:1405.7687.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    # Interleave: at odd indices the lead advances, lag stays.
    # Result has 2T - 1 rows.
    out = np.empty((2 * T - 1, 2 * d))
    for k in range(T - 1):
        out[2 * k, :d] = path[k]          # (X_k, X_{k-1}) — lag catches up
        out[2 * k, d:] = path[k]
        out[2 * k + 1, :d] = path[k + 1]  # lead advances
        out[2 * k + 1, d:] = path[k]
    out[-1, :d] = path[-1]
    out[-1, d:] = path[-1]
    return out


def invisibility_augmentation(path: np.ndarray) -> np.ndarray:
    """Add a basepoint-encoding channel via the 'invisibility reset'.

    Prepends a virtual starting segment that carries the initial condition:

        (0, X_0), (1, X_0), (1, X_1), ..., (1, X_T)

    The first step (from 0 to 1 in the new channel while staying at X_0)
    encodes the starting position X_0 into the signature, which is otherwise
    insensitive to where in state space the path begins.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.

    Returns
    -------
    np.ndarray
        Shape ``(T + 1, d + 1)``.  Column 0 is the visibility flag.

    References
    ----------
    Morrill et al. (2020). Neural CDEs for long time series. arXiv:2009.09433.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((T + 1, d + 1))
    # invisible point: visibility=0, at X_0
    out[0, 0] = 0.0
    out[0, 1:] = path[0]
    # visible points: visibility=1
    out[1:, 0] = 1.0
    out[1:, 1:] = path
    return out


def cumulative_moving_average_augmentation(path: np.ndarray) -> np.ndarray:
    """Append a channel containing the cumulative moving average of each column.

    The CMA channel tracks the expanding mean up to each time point:
        CMA^i_t = (1 / t) * sum_{s=1}^{t} X^i_s

    Signature cross-terms between the original channel and the CMA channel
    capture mean-reversion signals: S^2_{price, CMA} > 0 indicates the
    price tended to move in the same direction as its own expanding mean
    (trend-following), while S^2_{price, CMA} < 0 suggests mean-reversion.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.

    Returns
    -------
    np.ndarray
        Shape ``(T, 2*d)``.  Columns 0..d-1 are the original channels;
        columns d..2d-1 are the corresponding cumulative moving averages.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    cma = np.cumsum(path, axis=0) / np.arange(1, T + 1).reshape(-1, 1)
    return np.concatenate([path, cma], axis=1)


def logsig_augmentation(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute the log-signature of the path.

    The log-signature lives in the free Lie algebra and is strictly more
    compact than the signature at depth >= 2.  For d channels and depth 2
    the log-signature has d + d(d-1)/2 coordinates (the Lie bracket terms
    collapse to antisymmetric combinations), compared to d + d^2 for the
    full signature.

    Concretely, at depth 2:
        logsig^1_i = sig^1_i  (level-1 terms are unchanged)
        logsig^2_{i,j} = sig^2_{i,j} - sig^2_{j,i}  (Lévy area)

    Higher-level terms follow from the Baker-Campbell-Hausdorff formula.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.  This function computes the log-signature of
        the *entire* path (not a rolling version).
    depth:
        Truncation level.

    Returns
    -------
    np.ndarray
        Log-signature vector; dimension depends on the Lyndon basis size.

    Notes
    -----
    Requires the ``signatory`` package for depth >= 3.  For depth <= 2
    a pure NumPy implementation is used.
    """
    from pathsig.signatures import compute_signature, signature_dimension

    path = np.asarray(path, dtype=float)
    _, d = path.shape

    try:
        import torch
        import signatory

        path_t = torch.tensor(path, dtype=torch.float32).unsqueeze(0)
        logsig = signatory.logsignature(path_t, depth=depth)
        return logsig.squeeze(0).detach().numpy()
    except ImportError:
        pass

    # Depth-2 NumPy fallback: use the relation logsig = sig - (1/2) sig^2 + ...
    # At depth 2 the antisymmetrisation gives the Lévy area.
    if depth > 2:
        raise ImportError(
            "signatory is required for log-signature at depth > 2. "
            "Install with: pip install signatory"
        )

    sig = compute_signature(path, depth=2)
    level1 = sig[:d]
    level2 = sig[d : d + d * d].reshape(d, d)
    # Lévy areas: antisymmetric part of level 2
    levy_areas = []
    for i in range(d):
        for j in range(i + 1, d):
            levy_areas.append(level2[i, j] - level2[j, i])
    return np.concatenate([level1, levy_areas])


def apply_augmentations(path: np.ndarray, augmentations: list[str]) -> np.ndarray:
    """Apply a named sequence of augmentations to a path.

    Parameters
    ----------
    path:
        Shape ``(T, d)``.
    augmentations:
        List of augmentation names drawn from:
        ``['time', 'lead_lag', 'invisibility', 'cma']``.
        Applied left-to-right.

    Returns
    -------
    np.ndarray
        Augmented path (shape varies by augmentation choice).

    Raises
    ------
    ValueError
        If an unknown augmentation name is supplied.
    """
    _dispatch = {
        "time": time_augmentation,
        "lead_lag": lead_lag_augmentation,
        "invisibility": invisibility_augmentation,
        "cma": cumulative_moving_average_augmentation,
    }
    result = np.asarray(path, dtype=float)
    for name in augmentations:
        if name not in _dispatch:
            raise ValueError(
                f"Unknown augmentation '{name}'. Choose from {list(_dispatch)}."
            )
        result = _dispatch[name](result)
    return result
