"""Ornstein-Uhlenbeck calibration for the intraday-vs-day-ahead spread.

Model: dX_t = kappa * (mu - X_t) dt + sigma dW_t.

Its exact discrete-time transition (for a regularly-sampled series with
step `dt`) is the Gaussian AR(1) process

    X_{t+1} = a + b * X_t + eps_t,   eps_t ~ N(0, sigma_eps^2)

with  b = exp(-kappa * dt),  a = mu * (1 - b), and
sigma_eps^2 = sigma^2 * (1 - exp(-2 * kappa * dt)) / (2 * kappa). Both
calibration methods below therefore fit the *same* AR(1) regression to get
(a, b) -- under the exact Gaussian transition density, the maximum
likelihood estimator for (a, b) coincides with the ordinary least-squares
estimator, since maximizing the Gaussian log-likelihood over the mean
parameters for a fixed noise variance is exactly a least-squares problem.
The two methods differ only in how they estimate the innovation variance
`sigma_eps^2` from the regression residuals: OLS uses the usual n-2
(unbiased) denominator, MLE the exact n (biased) denominator. This is a
genuine, if subtle, difference worth exposing as a config choice
(`ou.estimation.method`) rather than a purely cosmetic one.

Calibration is intended to run on either a granular per-product intraday
price path (once a paid EPEX ID1/ID3 feed is plugged into
`fr_power.data.intraday`) or, in this repository's default degraded
state, the hourly intraday-vs-day-ahead spread series itself
(`target_spread_eur_mwh` from `fr_power.features.build`). A raw price
*level* is not economically mean-reverting in the sense this model
assumes and should not be calibrated directly -- see the `calibrate` CLI
command, which refuses to run OU calibration when only the day-ahead
price level is available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OUParameters:
    """Calibrated OU parameters plus fit diagnostics."""

    kappa: float  # mean-reversion speed, per period (dt units)
    mu: float  # long-run mean
    sigma: float  # volatility, per sqrt(period)
    half_life: float  # ln(2) / kappa, in periods
    stationary_std: float  # sigma / sqrt(2 * kappa); the process's unconditional std
    r_squared: float  # AR(1) regression fit quality
    n_observations: int
    method: Literal["ols", "mle"]
    dt: float


def _prepare_pairs(series: pd.Series, min_observations: int) -> tuple[np.ndarray, np.ndarray]:
    clean = series.dropna()
    if len(clean) < min_observations:
        raise ValueError(
            f"Need at least {min_observations} observations to calibrate an OU process, "
            f"got {len(clean)}"
        )
    values = clean.to_numpy(dtype=float)
    return values[:-1], values[1:]


def _fit_ar1(x_t: np.ndarray, x_next: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least-squares fit of X_{t+1} = a + b * X_t + eps. Returns (a, b, residuals)."""
    design = np.column_stack([np.ones_like(x_t), x_t])
    coeffs, *_ = np.linalg.lstsq(design, x_next, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    resid = x_next - design @ coeffs
    return a, b, resid


def _ou_params_from_ar1(
    a: float, b: float, resid: np.ndarray, dt: float, ddof: int, method: Literal["ols", "mle"]
) -> OUParameters:
    if not (0.0 < b < 1.0):
        raise ValueError(
            f"Estimated AR(1) coefficient b={b:.4f} implies no valid mean reversion "
            "(need 0 < b < 1); this series does not look mean-reverting over the given dt."
        )

    kappa = -np.log(b) / dt
    mu = a / (1 - b)
    n = len(resid)
    resid_var = float(np.sum(resid**2) / max(n - ddof, 1))
    sigma = float(np.sqrt(resid_var * 2 * kappa / (1 - np.exp(-2 * kappa * dt))))
    half_life = float(np.log(2) / kappa)
    stationary_std = float(sigma / np.sqrt(2 * kappa))

    return OUParameters(
        kappa=float(kappa),
        mu=float(mu),
        sigma=sigma,
        half_life=half_life,
        stationary_std=stationary_std,
        r_squared=float("nan"),  # filled in by callers that have x_next available
        n_observations=n,
        method=method,
        dt=dt,
    )


def _r_squared(x_next: np.ndarray, resid: np.ndarray) -> float:
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((x_next - x_next.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def estimate_ou_ols(series: pd.Series, dt: float = 1.0, min_observations: int = 20) -> OUParameters:
    """Calibrate via OLS, with the unbiased (n-2) innovation-variance estimator."""
    x_t, x_next = _prepare_pairs(series, min_observations)
    a, b, resid = _fit_ar1(x_t, x_next)
    params = _ou_params_from_ar1(a, b, resid, dt, ddof=2, method="ols")
    return _with_r_squared(params, x_next, resid)


def estimate_ou_mle(series: pd.Series, dt: float = 1.0, min_observations: int = 20) -> OUParameters:
    """Calibrate via the exact discrete-time Gaussian MLE (n-denominator variance estimator)."""
    x_t, x_next = _prepare_pairs(series, min_observations)
    a, b, resid = _fit_ar1(x_t, x_next)
    params = _ou_params_from_ar1(a, b, resid, dt, ddof=0, method="mle")
    return _with_r_squared(params, x_next, resid)


def _with_r_squared(params: OUParameters, x_next: np.ndarray, resid: np.ndarray) -> OUParameters:
    return replace(params, r_squared=_r_squared(x_next, resid))


def estimate_ou(
    series: pd.Series,
    method: Literal["ols", "mle"] = "ols",
    dt: float = 1.0,
    min_observations: int = 20,
) -> OUParameters:
    """Dispatch to `estimate_ou_ols` or `estimate_ou_mle` by name (see `ou.estimation.method`)."""
    if method == "ols":
        return estimate_ou_ols(series, dt, min_observations)
    if method == "mle":
        return estimate_ou_mle(series, dt, min_observations)
    raise ValueError(f"Unknown OU estimation method: {method!r}")


def compute_z_score(series: pd.Series, params: OUParameters) -> pd.Series:
    """Standardized deviation from the OU long-run mean, in units of the stationary std."""
    return ((series - params.mu) / params.stationary_std).rename("z_score")
