"""Forecast accuracy metrics and the Diebold-Mariano pairwise comparison test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _aligned_arrays(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop any pair with a NaN on either side; align by index when both are Series."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        aligned = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
        return aligned["y_true"].to_numpy(dtype=float), aligned["y_pred"].to_numpy(dtype=float)
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    a, b = _aligned_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    a, b = _aligned_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mdae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    a, b = _aligned_arrays(y_true, y_pred)
    return float(np.median(np.abs(a - b)))


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Convenience bundle of the three headline point-forecast metrics."""
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "mdae": mdae(y_true, y_pred)}


def diebold_mariano_test(
    errors_a: pd.Series | np.ndarray,
    errors_b: pd.Series | np.ndarray,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy between two forecasts.

    Parameters
    ----------
    errors_a, errors_b : forecast errors (actual - predicted) for models A and
        B, aligned to the same timestamps.
    h : forecast horizon, used for the Newey-West truncation lag (h - 1). Use
        h=1 for one-step-ahead, non-overlapping forecasts (no autocovariance
        correction needed).
    power : 1 for absolute-error loss, 2 (default) for squared-error loss.

    Returns
    -------
    (dm_statistic, two_sided_p_value). A significantly negative statistic
    (p < 0.05) favors model A (lower average loss); a significantly positive
    statistic favors model B.
    """
    a, b = _aligned_arrays(errors_a, errors_b)
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 overlapping observations for the DM test")

    d = np.abs(a) ** power - np.abs(b) ** power
    d_mean = d.mean()

    long_run_var = np.var(d, ddof=0)
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        long_run_var += 2 * cov

    var_d_mean = long_run_var / n
    if var_d_mean <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d_mean)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)
