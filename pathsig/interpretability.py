"""
Formal mapping between signature terms and financial phenomena.

This module is the core interpretability contribution of pathsig-finance.
It documents and implements the exact correspondence between truncated
signature coordinates and classical financial features:

    Level 1  →  Momentum (cumulative net displacement = total return)
    Level 2 diagonal (lead-lag augmented)  →  Realised variance
    Level 2 off-diagonal (price × vol)     →  Leverage effect (Black 1976)
    Level 2 off-diagonal (price ↔ volume)  →  Lead-lag / informed trading
    Level 3 terms  →  Skewness proxies, higher-order interactions

Each function below accepts a pre-computed signature and returns a
human-readable dictionary with the financial interpretation alongside
the raw numerical value.

References
----------
Chevyrev & Kormilitzin (2016). A primer on the signature method in
    machine learning. arXiv:1603.03788.
Lyons (1998). Differential equations driven by rough signals.
    Revista Matemática Iberoamericana, 14(2).
Kidger & Lyons (2021). Signatory: differentiable computations of the
    signature and log-signature transforms. ICLR 2021.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from pathsig.signatures import (
    compute_signature,
    extract_signature_term,
    signature_dimension,
    signature_to_dict,
)
from pathsig.augmentations import lead_lag_augmentation, time_augmentation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level-1 interpretability: Momentum
# ---------------------------------------------------------------------------


def interpret_level1(
    signature: np.ndarray,
    d: int,
    channel_names: Optional[list[str]] = None,
) -> dict:
    """Interpret level-1 signature terms as momentum / net displacement.

    Mathematical correspondence
    ---------------------------
    The level-1 term for channel i is:

        S^1_i = ∫_0^T dX^i_t = X^i_T - X^i_0

    This is the net displacement of channel i over the window.  For a
    log-return channel (X^i_t = log P_t) this equals the cumulative
    log-return — precisely the momentum signal of Jegadeesh & Titman (1993).

    Parameters
    ----------
    signature:
        Flattened signature array (all levels concatenated).
    d:
        Number of channels in the original (possibly augmented) path.
    channel_names:
        Optional list of length d with human-readable channel labels.

    Returns
    -------
    dict
        ``{'channel_name': {'value': float, 'interpretation': str, ...}, ...}``
    """
    if channel_names is None:
        channel_names = [f"X{i}" for i in range(d)]

    result = {}
    for i, name in enumerate(channel_names):
        val = extract_signature_term(signature, d=d, multi_index=(i,))
        result[name] = {
            "value": val,
            "level": 1,
            "multi_index": (i,),
            "formula": f"S^1_{i} = X^{i}_T - X^{i}_0",
            "financial_meaning": (
                "Cumulative log-return (momentum)" if "return" in name.lower()
                else "Volume net change" if "volume" in name.lower() or "vol" in name.lower()
                else "Net displacement"
            ),
            "interpretation": (
                f"{'Positive' if val > 0 else 'Negative'} momentum: "
                f"net displacement = {val:.6f}"
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Level-2 diagonal: Realised variance (via lead-lag)
# ---------------------------------------------------------------------------


def interpret_level2_diagonal(
    signature: np.ndarray,
    d_original: int,
    channel_names: Optional[list[str]] = None,
) -> dict:
    """Interpret the cross-terms between lead and lag channels as realised variance.

    Mathematical correspondence
    ---------------------------
    For the lead-lag augmented path of a d-dimensional path X, the
    augmented dimension is 2d: the first d channels are the "lead"
    (= X itself) and the last d are the "lag" (= X shifted by one step).

    The *Lévy area* between lead channel i and lag channel i equals the
    quadratic variation:

        A_{lead_i, lag_i} = S^2_{lead_i, lag_i} - S^2_{lag_i, lead_i}
                          = sum_k (ΔX^i_k)^2
                          = [X^i, X^i]_T   (quadratic variation)

    Proof: each step k sweeps a right-angle turn in the (lead, lag) plane.
    The Lévy area of that turn is (ΔX^i_k)^2; summing gives QV.

    Note: S^2_{lead_i, lag_i} alone equals (1/2)*CumRet_i^2 + (1/2)*QV_i;
    the antisymmetric combination (Lévy area) isolates QV.

    Parameters
    ----------
    signature:
        Signature of the **lead-lag augmented** path, dimension 2d_original.
    d_original:
        Number of channels in the original (pre-lead-lag) path.
    channel_names:
        Optional labels for the original channels.

    Returns
    -------
    dict
        Per-channel realised-variance estimates and their interpretations.
    """
    if channel_names is None:
        channel_names = [f"X{i}" for i in range(d_original)]

    d_aug = 2 * d_original  # dimension after lead-lag
    result = {}
    for i, name in enumerate(channel_names):
        lead_i = i
        lag_i = i + d_original
        # Lévy area = S^2_{lead,lag} - S^2_{lag,lead} = QV
        s_lead_lag = extract_signature_term(signature, d=d_aug, multi_index=(lead_i, lag_i))
        s_lag_lead = extract_signature_term(signature, d=d_aug, multi_index=(lag_i, lead_i))
        realized_var = s_lead_lag - s_lag_lead  # = QV = sum_k (ΔX_k)^2
        realized_var = max(realized_var, 0.0)   # numerical safety
        result[name] = {
            "levy_area": float(s_lead_lag - s_lag_lead),
            "S2_lead_lag": float(s_lead_lag),
            "S2_lag_lead": float(s_lag_lead),
            "realized_variance": float(realized_var),
            "realized_volatility": float(np.sqrt(realized_var)),
            "level": 2,
            "multi_index_lead_lag": (lead_i, lag_i),
            "multi_index_lag_lead": (lag_i, lead_i),
            "formula": (
                f"A_{{{lead_i},{lag_i}}} = S^2_{{{lead_i},{lag_i}}} "
                f"- S^2_{{{lag_i},{lead_i}}} = [X^{i}, X^{i}]_T"
            ),
            "financial_meaning": "Realised variance (quadratic variation via Lévy area)",
            "interpretation": (
                f"Realised vol estimate: {np.sqrt(realized_var):.6f} "
                f"(annualised ≈ {np.sqrt(realized_var * 252):.4f})"
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Level-2 off-diagonal: The Leverage Effect
# ---------------------------------------------------------------------------


def interpret_level2_cross(
    signature: np.ndarray,
    d: int,
    channel_i: int,
    channel_j: int,
    channel_names: Optional[list[str]] = None,
) -> dict:
    """Interpret the level-2 cross-term and Lévy area between two channels.

    Mathematical correspondence
    ---------------------------
    The ordered level-2 terms S^2_{i,j} and S^2_{j,i} capture the
    *asymmetric* temporal relationship between channels i and j:

        S^2_{i,j} = ∫∫_{s<t} dX^i_s dX^j_t
                  = (how much X^i moves first, then X^j follows)

    The antisymmetric part is the Lévy area:

        A_{i,j} = S^2_{i,j} - S^2_{j,i}
                = ∫_0^T X^i_t dX^j_t - ∫_0^T X^j_t dX^i_t   (Itô formula)

    The Leverage Effect (Black 1976)
    --------------------------------
    For channels i = log-return and j = realised volatility:

        A_{return, vol} < 0  ⟺  negative return → increased volatility

    This is the leverage effect: falling prices raise financial leverage,
    increasing equity risk.  A_{return, vol} quantifies its magnitude.

    Parameters
    ----------
    signature:
        Flattened signature array.
    d:
        Number of channels.
    channel_i, channel_j:
        0-based indices of the two channels.
    channel_names:
        Optional labels (length d).

    Returns
    -------
    dict
        Cross-term values, Lévy area, and financial interpretation.
    """
    if channel_names is None:
        channel_names = [f"X{k}" for k in range(d)]

    name_i = channel_names[channel_i]
    name_j = channel_names[channel_j]

    s_ij = extract_signature_term(signature, d=d, multi_index=(channel_i, channel_j))
    s_ji = extract_signature_term(signature, d=d, multi_index=(channel_j, channel_i))
    levy_area = s_ij - s_ji
    symmetric = 0.5 * (s_ij + s_ji)

    # Infer financial meaning from channel names
    name_pair = {name_i.lower(), name_j.lower()}
    if any("return" in n or "price" in n for n in name_pair) and any(
        "vol" in n or "sigma" in n for n in name_pair
    ):
        if levy_area < 0:
            effect = "Negative leverage effect: past negative returns → higher volatility."
        elif levy_area > 0:
            effect = "Positive cross-term: positive returns → higher volatility (unusual)."
        else:
            effect = "No detectable leverage effect."
        financial_meaning = "Leverage effect (Black 1976)"
    elif any("volume" in n or "vol" in n for n in name_pair) and any(
        "return" in n or "price" in n for n in name_pair
    ):
        financial_meaning = "Price-volume lead-lag relationship"
        effect = interpret_price_volume_leadlag.__doc__  # placeholder
    else:
        financial_meaning = f"Cross-channel interaction: {name_i} × {name_j}"
        effect = f"Lévy area = {levy_area:.6f}"

    return {
        f"S2_{name_i}_{name_j}": s_ij,
        f"S2_{name_j}_{name_i}": s_ji,
        "levy_area": levy_area,
        "symmetric_part": symmetric,
        "level": 2,
        "multi_index_ij": (channel_i, channel_j),
        "multi_index_ji": (channel_j, channel_i),
        "formula": (
            f"A_{{{name_i},{name_j}}} = S^2_{{{channel_i},{channel_j}}} "
            f"- S^2_{{{channel_j},{channel_i}}}"
        ),
        "financial_meaning": financial_meaning,
        "interpretation": effect,
    }


# ---------------------------------------------------------------------------
# Price-Volume lead-lag
# ---------------------------------------------------------------------------


def interpret_price_volume_leadlag(
    signature: np.ndarray,
    d: int,
    price_channel: int,
    volume_channel: int,
    channel_names: Optional[list[str]] = None,
) -> dict:
    """Quantify the temporal ordering of price and volume movements.

    Mathematical correspondence
    ---------------------------
    The asymmetry between S^2_{price, volume} and S^2_{volume, price}
    reveals which series tends to lead the other:

        If S^2_{vol, price} > S^2_{price, vol}:
            Volume moves first, price follows.
            → Consistent with informed trading / order-flow driven moves
              (Easley & O'Hara 1987, Glosten & Milgrom 1985).

        If S^2_{price, vol} > S^2_{vol, price}:
            Price moves first, volume follows.
            → Consistent with momentum / trend-chasing behaviour.

    The Lévy area A_{price, vol} = S^2_{price,vol} - S^2_{vol,price}
    gives a signed, scalar measure of the lead-lag strength.

    Parameters
    ----------
    signature:
        Flattened signature array.
    d:
        Number of channels.
    price_channel:
        Index of the price / log-return channel.
    volume_channel:
        Index of the log-volume channel.
    channel_names:
        Optional labels (length d).

    Returns
    -------
    dict
        Lévy area, lead-lag direction, and financial interpretation.
    """
    if channel_names is None:
        channel_names = [f"X{k}" for k in range(d)]

    s_pv = extract_signature_term(
        signature, d=d, multi_index=(price_channel, volume_channel)
    )
    s_vp = extract_signature_term(
        signature, d=d, multi_index=(volume_channel, price_channel)
    )
    levy_area = s_pv - s_vp

    if levy_area > 0:
        direction = "price leads volume"
        econ_interp = "Momentum / trend-following: price moves attract trading activity."
    elif levy_area < 0:
        direction = "volume leads price"
        econ_interp = "Informed trading: order flow precedes price discovery."
    else:
        direction = "simultaneous"
        econ_interp = "No detectable lead-lag relationship."

    return {
        "S2_price_volume": s_pv,
        "S2_volume_price": s_vp,
        "levy_area": levy_area,
        "lead_lag_direction": direction,
        "level": 2,
        "formula": "A_{price,vol} = S^2_{price,vol} - S^2_{vol,price}",
        "financial_meaning": "Price-volume lead-lag (informed trading vs momentum)",
        "interpretation": econ_interp,
    }


# ---------------------------------------------------------------------------
# Level-3 interpretability: Higher-order interactions
# ---------------------------------------------------------------------------


def interpret_level3_highlights(
    signature: np.ndarray,
    d: int,
    channel_names: Optional[list[str]] = None,
) -> dict:
    """Provide approximate interpretations for prominent level-3 terms.

    Level-3 terms S^3_{i,j,k} = ∫∫∫_{s<u<t} dX^i_s dX^j_u dX^k_t encode
    third-order interactions.  For financial channels, key examples are:

    * S^3_{r,r,r}  (return cubed): skewness of the return distribution
    * S^3_{r,σ,r}  (return-vol-return): interaction of trend with volatility
    * S^3_{r,r,σ}  (return-return-vol): volatility-of-volatility (vol-of-vol)

    These are harder to interpret precisely, but the diagonal term
    S^3_{r,r,r} ≈ (1/6) * sum_k (ΔX^r_k)^3  relates to return skewness.

    Parameters
    ----------
    signature:
        Signature array (must include level-3 terms).
    d:
        Number of channels.
    channel_names:
        Optional labels.

    Returns
    -------
    dict
        Key level-3 terms and their approximate financial interpretations.
    """
    if channel_names is None:
        channel_names = [f"X{k}" for k in range(d)]

    result = {}
    # Extract all level-3 terms for all channel triples
    for i in range(d):
        for j in range(d):
            for k in range(d):
                val = extract_signature_term(signature, d=d, multi_index=(i, j, k))
                key = f"S3_{channel_names[i]}_{channel_names[j]}_{channel_names[k]}"

                if i == j == k:
                    interp = (
                        f"Approximate (1/6) * E[(ΔX^{i})^3]: "
                        f"cubic return / skewness proxy for {channel_names[i]}"
                    )
                elif i == k and i != j:
                    interp = (
                        f"Return skewness conditioned on {channel_names[j]} state: "
                        f"interaction of {channel_names[i]} and {channel_names[j]}"
                    )
                else:
                    interp = (
                        f"Third-order interaction: {channel_names[i]} → "
                        f"{channel_names[j]} → {channel_names[k]}"
                    )

                result[key] = {"value": val, "level": 3,
                               "multi_index": (i, j, k), "interpretation": interp}
    return result


# ---------------------------------------------------------------------------
# Full interpretability report
# ---------------------------------------------------------------------------


def full_interpretability_report(
    path: np.ndarray,
    depth: int = 3,
    channel_names: Optional[list[str]] = None,
    augmentations: Optional[list[str]] = None,
    price_channel: int = 0,
    volume_channel: Optional[int] = None,
    vol_channel: Optional[int] = None,
) -> dict:
    """Generate a complete interpretability report for a multivariate financial path.

    Applies the requested augmentations, computes the truncated signature,
    and maps each level of the signature to the corresponding financial
    phenomenon.

    Parameters
    ----------
    path:
        Shape ``(T, d)`` — raw financial time series window (e.g. log-returns,
        realised vol, log-volume).
    depth:
        Signature truncation level (typically 2 or 3).
    channel_names:
        Human-readable names for each column of ``path`` (length d).
    augmentations:
        List of augmentation names to apply before signature computation.
        E.g. ``['time', 'lead_lag']``.  Applied left-to-right.
    price_channel:
        Index (in the *original* path) of the price / log-return channel.
    volume_channel:
        Index of the log-volume channel (if present).
    vol_channel:
        Index of the realised-volatility channel (if present).

    Returns
    -------
    dict with keys:
        ``'augmented_path_shape'``,
        ``'signature_dimension'``,
        ``'level1_momentum'``,
        ``'level2_realized_variance'`` (if lead-lag augmented),
        ``'leverage_effect'`` (if vol_channel given),
        ``'price_volume_leadlag'`` (if volume_channel given),
        ``'level3_highlights'`` (if depth >= 3),
        ``'full_signature_dict'``.
    """
    path = np.asarray(path, dtype=float)
    if channel_names is None:
        channel_names = [f"X{i}" for i in range(path.shape[1])]

    if augmentations is None:
        augmentations = []

    # Apply augmentations
    aug_path = path.copy()
    for aug in augmentations:
        if aug == "time":
            aug_path = time_augmentation(aug_path)
        elif aug == "lead_lag":
            aug_path = lead_lag_augmentation(aug_path)
        else:
            from pathsig.augmentations import apply_augmentations
            aug_path = apply_augmentations(aug_path, [aug])

    d_aug = aug_path.shape[1]
    sig = compute_signature(aug_path, depth=depth)
    sig_dim = signature_dimension(d_aug, depth)

    report: dict = {
        "augmented_path_shape": aug_path.shape,
        "signature_dimension": sig_dim,
        "augmentations_applied": augmentations,
    }

    # Level 1: use original channels for naming, but computed from augmented sig
    # If time was prepended, channel 0 is time — adjust names accordingly
    aug_channel_names: list[str] = []
    for aug in augmentations:
        if aug == "time":
            aug_channel_names = ["time"] + channel_names
            channel_names = aug_channel_names
        elif aug == "lead_lag":
            aug_channel_names = (
                [f"lead_{n}" for n in channel_names]
                + [f"lag_{n}" for n in channel_names]
            )
            channel_names = aug_channel_names
    if not aug_channel_names:
        aug_channel_names = channel_names

    report["level1_momentum"] = interpret_level1(sig, d_aug, aug_channel_names)

    # Level 2 diagonal: realised variance (only meaningful after lead-lag)
    if "lead_lag" in augmentations:
        d_orig = path.shape[1]
        report["level2_realized_variance"] = interpret_level2_diagonal(
            sig, d_orig, channel_names=channel_names[:d_orig]
        )

    # Leverage effect
    if vol_channel is not None:
        report["leverage_effect"] = interpret_level2_cross(
            sig, d_aug, price_channel, vol_channel, aug_channel_names
        )

    # Price-volume lead-lag
    if volume_channel is not None:
        report["price_volume_leadlag"] = interpret_price_volume_leadlag(
            sig, d_aug, price_channel, volume_channel, aug_channel_names
        )

    # Level-3 highlights
    if depth >= 3:
        report["level3_highlights"] = interpret_level3_highlights(
            sig, d_aug, aug_channel_names
        )

    report["full_signature_dict"] = signature_to_dict(sig, d_aug, depth, aug_channel_names)

    return report
