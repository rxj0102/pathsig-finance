"""
Synthetic financial path generators with known mathematical properties.

These generators are the primary testing and validation tool.  Each
creates panel data where the "ground truth" parameter is known, so we
can verify that the signature-based estimator recovers it.

Three families:

``generate_gbm_with_leverage``
    Heston-like stochastic volatility with known leverage correlation ρ.
    Used to verify that the Lévy area S^2_{return, vol} - S^2_{vol, return}
    recovers the sign and approximate magnitude of ρ.

``generate_leadlag_volume_price``
    Volume leads price by a known lag L.
    Used to verify that the Lévy area between price and volume channels
    correctly identifies the lead-lag direction.

``generate_momentum_regime``
    Returns have positive autocorrelation of known strength μ.
    Used to verify that level-1 signature features predict next-period
    returns with the correct coefficient.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_gbm_with_leverage(
    n_paths: int = 500,
    n_steps: int = 252,
    mu: float = 0.08,
    sigma0: float = 0.2,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    leverage_corr: float = -0.7,
    dt: float = 1.0 / 252,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate equity paths from a Heston stochastic-volatility model.

    The model is:

        dS_t / S_t = μ dt + σ_t dW^1_t
        dσ²_t      = κ(θ - σ²_t) dt + ξ σ_t dW^2_t
        corr(dW^1_t, dW^2_t) = ρ   (leverage_corr)

    where the second equation is the Heston (1993) variance dynamics.

    The leverage correlation ρ < 0 means that negative equity returns
    (dW^1 < 0) tend to be accompanied by upward volatility innovations
    (dW^2 > 0).  This is Black's (1976) leverage effect.

    Validation target
    -----------------
    For the discretised path (log_return, Δσ), the Lévy area:

        A_{ret, Δσ} = S^2_{ret, Δσ} - S^2_{Δσ, ret}

    should be negative when ρ < 0.  The Pearson correlation of
    (log_return_t, Δσ_{t+1}) across all stocks and steps provides
    a direct estimate of ρ for comparison.

    Parameters
    ----------
    n_paths:
        Number of simulated stocks.
    n_steps:
        Number of daily time steps (252 = 1 trading year).
    mu:
        Drift of the log-price process (annualised).
    sigma0:
        Initial volatility.
    kappa:
        Mean-reversion speed of the variance process.
    theta:
        Long-run mean of variance (σ² → θ as t → ∞).
    xi:
        Volatility-of-volatility.
    leverage_corr:
        Correlation ρ between price and variance Brownian motions.
    dt:
        Time step in years (default 1/252 = 1 trading day).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        ``['date', 'ticker', 'log_return', 'realized_vol', 'log_volume',
          'true_sigma', 'true_rho']``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_steps, freq="B")

    # Cholesky decomposition for correlated BMs
    L = np.array([
        [1.0, 0.0],
        [leverage_corr, np.sqrt(1.0 - leverage_corr**2)],
    ])

    records = []
    for path_idx in range(n_paths):
        ticker = f"SYN_{path_idx:04d}"
        sigma_sq = sigma0**2
        log_price = 0.0  # start at 0 in log space

        prev_sigma = np.sqrt(sigma_sq)
        for t in range(n_steps):
            sigma = max(np.sqrt(abs(sigma_sq)), 1e-6)

            # Correlated standard normals
            Z = rng.standard_normal(2)
            W = L @ Z

            # Euler-Maruyama step
            log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W[0]
            d_sigma_sq = kappa * (theta - sigma_sq) * dt + xi * sigma * np.sqrt(dt) * W[1]
            sigma_sq = max(sigma_sq + d_sigma_sq, 1e-8)
            log_price += log_ret

            new_sigma = np.sqrt(sigma_sq)

            # Synthetic log-volume: correlated with abs(return), mean-reverting
            log_vol_noise = 0.3 * abs(log_ret) / (sigma * np.sqrt(dt) + 1e-8) + \
                            rng.standard_normal() * 0.1
            log_volume = 10.0 + log_vol_noise

            records.append(
                {
                    "date": dates[t],
                    "ticker": ticker,
                    "log_return": float(log_ret),
                    "realized_vol": float(new_sigma),
                    "log_volume": float(log_volume),
                    "true_sigma": float(new_sigma),
                    "true_rho": float(leverage_corr),
                }
            )
            prev_sigma = new_sigma

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d Heston paths × %d steps (ρ=%.2f).", n_paths, n_steps, leverage_corr
    )
    return df


def generate_leadlag_volume_price(
    n_paths: int = 500,
    n_steps: int = 252,
    lag: int = 1,
    beta: float = 0.05,
    sigma_v: float = 0.1,
    sigma_r: float = 0.01,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate paths where volume leads price by a known lag.

    Model:

        V_t = V_{t-1} + η_t,      η_t ~ N(0, σ_v²)
        r_t = β * V_{t-lag} + ε_t, ε_t ~ N(0, σ_r²)

    where V_t is log-volume and r_t is the daily log-return.

    Validation target
    -----------------
    When lag > 0, volume leads price.  The Lévy area:

        A_{price, vol} = S^2_{price,vol} - S^2_{vol,price}

    should be negative (volume moves first, so the lag component
    S^2_{vol, price} is larger).

    Parameters
    ----------
    n_paths:
        Number of simulated stocks.
    n_steps:
        Number of time steps.
    lag:
        Number of periods by which volume leads price.
    beta:
        Volume-to-price coefficient.
    sigma_v:
        Standard deviation of the volume innovation.
    sigma_r:
        Idiosyncratic return noise.
    seed:
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        ``['date', 'ticker', 'log_return', 'log_volume',
          'true_lag', 'true_beta']``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_steps, freq="B")

    records = []
    for path_idx in range(n_paths):
        ticker = f"SYNLAG_{path_idx:04d}"

        # Generate volume path
        V = np.zeros(n_steps + lag)
        V[0] = 10.0
        for t in range(1, n_steps + lag):
            V[t] = V[t - 1] + rng.normal(0, sigma_v)

        # Generate returns: driven by lagged volume
        R = np.zeros(n_steps)
        for t in range(n_steps):
            driving_vol = V[t]  # V[t] lags return by `lag` in the panel
            R[t] = beta * driving_vol + rng.normal(0, sigma_r)

        # Slice to n_steps
        V_obs = V[lag : lag + n_steps]

        for t in range(n_steps):
            records.append(
                {
                    "date": dates[t],
                    "ticker": ticker,
                    "log_return": float(R[t]),
                    "log_volume": float(V_obs[t]),
                    "realized_vol": float(sigma_r),
                    "true_lag": lag,
                    "true_beta": beta,
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d lead-lag paths × %d steps (lag=%d).", n_paths, n_steps, lag
    )
    return df


def generate_momentum_regime(
    n_paths: int = 500,
    n_steps: int = 252,
    momentum_strength: float = 0.05,
    sigma: float = 0.01,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate paths with positive return autocorrelation of known strength.

    Model (AR(1) in returns):

        r_t = μ_strength * r_{t-1} + ε_t,   ε_t ~ N(0, σ²)

    where ``momentum_strength`` is the AR(1) coefficient (must be in (-1, 1)).

    Validation target
    -----------------
    Level-1 signature features (cumulative return) should positively
    predict next-period returns, with a coefficient proportional to
    ``momentum_strength``.  OLS of r_{t+1} on Σ_{s≤t} r_s (within the
    window) should recover a positive, significant coefficient.

    Parameters
    ----------
    n_paths:
        Number of simulated stocks.
    n_steps:
        Number of time steps.
    momentum_strength:
        AR(1) coefficient.  Set to 0 for a martingale (no momentum).
    sigma:
        Idiosyncratic noise standard deviation.
    seed:
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        ``['date', 'ticker', 'log_return', 'realized_vol', 'log_volume',
          'true_momentum']``.
    """
    if abs(momentum_strength) >= 1.0:
        raise ValueError(
            f"momentum_strength must be in (-1, 1) for stationarity; got {momentum_strength}."
        )

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-03", periods=n_steps, freq="B")

    records = []
    for path_idx in range(n_paths):
        ticker = f"SYNMOM_{path_idx:04d}"
        r_prev = 0.0

        for t in range(n_steps):
            r_t = momentum_strength * r_prev + rng.normal(0, sigma)
            log_volume = 10.0 + 0.5 * abs(r_t) / sigma + rng.normal(0, 0.1)

            records.append(
                {
                    "date": dates[t],
                    "ticker": ticker,
                    "log_return": float(r_t),
                    "realized_vol": float(sigma),
                    "log_volume": float(log_volume),
                    "true_momentum": momentum_strength,
                }
            )
            r_prev = r_t

    df = pd.DataFrame(records)
    logger.info(
        "Generated %d momentum paths × %d steps (α=%.3f).",
        n_paths, n_steps, momentum_strength,
    )
    return df


def generate_multivariate_brownian(
    n_paths: int = 200,
    n_steps: int = 252,
    d: int = 3,
    correlation_matrix: Optional[np.ndarray] = None,
    dt: float = 1.0 / 252,
    seed: Optional[int] = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate correlated d-dimensional Brownian motion paths.

    Used for testing signature computations against known theoretical values.

    Parameters
    ----------
    n_paths:
        Number of independent paths.
    n_steps:
        Number of steps per path.
    d:
        Dimension (number of channels).
    correlation_matrix:
        d×d correlation matrix.  Defaults to identity (independent channels).
    dt:
        Time step.
    seed:
        RNG seed.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        ``(paths_array, panel_dataframe)`` where ``paths_array`` has shape
        ``(n_paths, n_steps, d)`` and ``panel_dataframe`` is in long format.
    """
    rng = np.random.default_rng(seed)

    if correlation_matrix is None:
        L = np.eye(d)
    else:
        corr = np.asarray(correlation_matrix, dtype=float)
        L = np.linalg.cholesky(corr)

    paths = np.zeros((n_paths, n_steps + 1, d))
    for i in range(n_paths):
        Z = rng.standard_normal((n_steps, d))
        increments = (Z @ L.T) * np.sqrt(dt)
        paths[i, 1:, :] = np.cumsum(increments, axis=0)

    # Build panel
    dates = pd.date_range("2000-01-03", periods=n_steps + 1, freq="B")
    records = []
    for i in range(n_paths):
        for t in range(1, n_steps + 1):
            row = {"date": dates[t], "ticker": f"BM_{i:04d}"}
            for ch in range(d):
                row[f"channel_{ch}"] = float(paths[i, t, ch])
            row["log_return"] = float(paths[i, t, 0] - paths[i, t - 1, 0])
            records.append(row)

    return paths[:, 1:, :], pd.DataFrame(records)
