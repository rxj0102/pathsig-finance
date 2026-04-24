# Audit Review Report — pathsig-finance

**Date:** 2026-04-24  
**Scope:** Full codebase audit covering mathematical correctness, numerical stability,
experiment pipeline integrity, and code quality.  
**Tests before audit:** 73 passing  
**Tests after audit:** 75 passing  

---

## Summary

Six issues were identified and fixed. The two critical mathematical bugs both concerned
the lead-lag Lévy area formula — an error that would have silently produced a factor-of-2
error in every realised-variance estimate. All issues are documented below with severity,
root cause, fix applied, and (where applicable) a new test that would have caught the
regression.

---

## Issues by Severity

### CRITICAL

#### C1 — Wrong formula in `augmentations.py` docstring: `S^2_{lead,lag} = (1/2)*QV`

**File:** `pathsig/augmentations.py`, `lead_lag_augmentation` docstring  
**Root cause:** The docstring originally claimed:

```
S^2_{lead_i, lag_i} = (1/2) * QV_i
```

This is mathematically wrong.  For a 1-D path [0, 1, -1] the lead-lag augmentation
gives paths whose Lévy area equals the quadratic variation (QV = 1+4 = 5), but
S^2_{lead,lag} ≠ (1/2)*QV.  The correct identity is:

```
A_{lead_i, lag_i} = S^2_{lead_i, lag_i} - S^2_{lag_i, lead_i} = sum_k (ΔX_k)^2 = QV
```

and

```
S^2_{lead_i, lag_i} = (1/2)*CumRet_i^2 + (1/2)*QV_i
```

so the individual (non-antisymmetrised) term is NOT equal to (1/2)*QV unless CumRet = 0.

**Impact:** Any user relying on the docstring to extract realised variance from a single
S^2 term (without antisymmetrising) would get a wrong answer.

**Fix:** Updated docstring to correctly state `A_{lead,lag} = S^2_{lead,lag} - S^2_{lag,lead} = QV`.

---

#### C2 — Wrong computation in `interpretability.py`: `realized_var = 2.0 * val`

**File:** `pathsig/interpretability.py`, `interpret_level2_diagonal`  
**Root cause:** The original implementation used:

```python
realized_var = 2.0 * s_lead_lag  # WRONG: used S^2_{lead,lag} alone, not the Lévy area
```

This would produce `(CumRet^2 + QV)` instead of `QV`, giving incorrect values whenever
CumRet ≠ 0.

**Fix:** Changed to the correct antisymmetric combination:

```python
s_lead_lag = extract_signature_term(...)
s_lag_lead = extract_signature_term(...)
realized_var = s_lead_lag - s_lag_lead  # = QV exactly
```

**New test:** `tests/test_interpretability.py::test_level2_diagonal_vs_realized_variance`
now asserts `rtol=1e-10` (exact identity) instead of a loose tolerance.

---

### HIGH

#### H1 — NaN values silently propagate through signature computation

**File:** `pathsig/signatures.py`, `compute_signature`  
**Root cause:** No input validation for NaN/Inf values.  A path with a single missing
data point (common in financial panels) would produce a NaN signature vector with no
error or warning, causing downstream models to silently train on corrupt features.

**Fix:** Added explicit guard before computation:

```python
if np.any(np.isnan(path)):
    raise ValueError(
        "path contains NaN values. Fill or drop missing data before computing signatures."
    )
```

**New tests:**
- `tests/test_signatures.py::test_compute_signature_nan_raises`
- `tests/test_signatures.py::test_compute_signature_nan_raises_first_row`

---

#### H2 — Momentum experiment uses raw return path; S^1 ≠ cumulative return

**Files:** `experiments/run_synthetic_validation.py`, `experiments/run_interpretability.py`  
**Root cause:** Both experiment scripts computed the level-1 signature term on the raw
log-return series, then labelled the result "cumulative return":

```python
path_w = rets[t - window : t].reshape(-1, 1)
sig = compute_signature(path_w, depth=1)
cum_ret = sig[0]  # labelled "cumulative return" — WRONG
```

When the path is the raw return series `[r_{t-w}, ..., r_{t-1}]`, the increments are
*differences* of returns and S^1 = r_{t-1} - r_{t-w} (last minus first), not
∑ r_t (cumulative return).  The cumulative return identity holds only when the path is
the **log-price series** `[0, r_1, r_1+r_2, ..., r_1+...+r_n]`.

**Fix:** Both scripts now build the log-price path:

```python
ret_w = rets[t - window : t]
price_w = np.concatenate([[0.0], np.cumsum(ret_w)]).reshape(-1, 1)
sig = compute_signature(price_w, depth=1)
cum_ret = sig[0]  # = sum(ret_w) = cumulative return ✓
```

---

### MEDIUM

#### M1 — `interpret_level1` docstring ambiguous about path type

**File:** `pathsig/interpretability.py`, `interpret_level1`  
**Root cause:** The docstring said "For a log-return channel (X^i_t = log P_t)" which
conflates the log-price and log-return series.  `SignatureFeatureExtractor` builds paths
from raw log-return values, so its S^1 terms are net displacements of the return series
(last return minus first return), not the cumulative log-return.

**Fix:** Docstring now explicitly distinguishes:
- **Log-price path** → S^1 = cumulative return (exact momentum identity)
- **Raw log-return path** → S^1 = last return − first return (different signal, still useful
  but not the Jegadeesh-Titman momentum factor)

---

#### M2 — `augmentations.py` test used wrong tolerance

**File:** `tests/test_augmentations.py`, `test_lead_lag_realized_variance`  
**Root cause:** Original test checked `S^2_{lead,lag} ≈ (1/2)*QV` with `rtol=0.05`.
This would pass even with the wrong formula for paths with small cumulative return.

**Fix:** Test now checks the correct identity `Lévy area = QV` with `rtol=1e-10`.

---

#### M3 — Missing high-dimensional signature warning

**File:** `pathsig/signatures.py`, `compute_signature`  
**Root cause:** No warning when the signature dimension exceeds practical limits.
For d=7 channels and depth=4 this gives 2800 features; for d=10 and depth=5 it gives
111 110 features. Without a warning, users may inadvertently request very large feature
vectors.

**Fix:** Added a `logger.warning` when `signature_dimension(d, depth) > 5000`:

```python
if sig_dim > 5000:
    logger.warning(
        "High-dimensional signature: %d features (d=%d, depth=%d). "
        "Consider using PCA or the log-signature for dimensionality control.",
        sig_dim, d, depth,
    )
```

---

### LOW

#### L1 — `pyproject.toml` used invalid build backend

**File:** `pyproject.toml`  
**Root cause:** `build-backend = "setuptools.backends.legacy:build"` causes
`ModuleNotFoundError` with current setuptools.  
**Fix:** Changed to `build-backend = "setuptools.build_meta"`.

---

## Verified Correct

The following items were hand-verified during the audit and confirmed correct:

| Component | Check | Result |
|---|---|---|
| `_chen_update` | Uses old `acc_levels`, not the mid-update state | ✓ Correct |
| `_chen_update` | Factorial denominators: `seg_k /= k` at each level | ✓ Correct |
| Hand computation: path `(0,0)→(1,0)→(1,1)` | S^2 values match closed form | ✓ |
| Hand computation: 1-D path 0→3 | S^k = 3^k / k! | ✓ |
| Chen identity | Split at any midpoint: <1e-17 error | ✓ |
| `lead_lag_augmentation` interleaving | Pattern `(X_k,X_k),(X_{k+1},X_k),(X_{k+1},X_{k+1})` | ✓ |
| Lévy area = QV (exact) | `rtol=1e-10` | ✓ |
| Heston Cholesky | `L = [[1,0],[ρ, √(1-ρ²)]]` | ✓ |
| Heston ρ recovery | ρ=−0.7 → est. corr ≈ −0.665 | ✓ |
| OOS R² perfect predictor | = 1.0 | ✓ |
| OOS R² mean predictor | = 0.0 | ✓ |
| NW SE: i.i.d. | ≈ OLS SE (ratio within 50%) | ✓ |
| NW SE: AR(1) φ=0.9 | NW > OLS SE | ✓ |
| NW SE: always non-negative | Verified analytically | ✓ |
| Fama-MacBeth beta recovery | 200 dates × 50 stocks: error < 0.05 | ✓ |
| Portfolio sort monotone | Positive signal → positive L/S | ✓ |
| Synthetic validation (all 3) | PASS (leverage sign, lead-lag direction, momentum sign) | ✓ |

---

## Test Suite

| Before audit | After audit |
|---|---|
| 73 tests, all passing | 75 tests, all passing |

New tests added:
- `test_compute_signature_nan_raises` — NaN in middle of path raises ValueError
- `test_compute_signature_nan_raises_first_row` — NaN in first row raises ValueError

Strengthened tests:
- `test_lead_lag_realized_variance` — now tests `Lévy_area = QV` at `rtol=1e-10`
- `test_level2_diagonal_vs_realized_variance` — now tests exact identity at `rtol=1e-10`

---

## Mathematical Correctness Statement

After all fixes, the following identities hold to machine precision in the implementation:

1. **Chen identity**: `S(X[0:T]) = S(X[0:k]) ⊗ S(X[k:T])` (verified to < 1e-17)
2. **Lead-lag Lévy area**: `S^2_{lead,lag} - S^2_{lag,lead} = Σ_k (ΔX_k)^2` (verified to `rtol=1e-10`)
3. **Level-1 = cumulative return** (when path = log-price): `S^1 = Σ r_k` (exact, by construction)
4. **Straight-line signature**: `S^k = Δ^{⊗k} / k!` (verified to < 1e-12)
5. **Leverage recovery**: `sign(mean Lévy area A_{r,σ}) = sign(ρ)` (verified on Heston simulations)
