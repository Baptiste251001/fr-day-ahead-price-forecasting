"""Forecast accuracy metrics and the Diebold-Mariano test.

All comparisons are made on the rows where *every* model has a prediction, so that no
model is flattered by being scored on an easier subset of hours.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def common_rows(actual: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    """Actual (column `actual`) and predictions, restricted to fully-observed rows."""
    return pd.concat([actual.rename("actual"), predictions], axis=1).dropna()


def point_metrics(actual: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, median absolute error and MAE relative to the naive model (rMAE)."""
    df = common_rows(actual, predictions)
    errors = df[predictions.columns].sub(df["actual"], axis=0)
    out = pd.DataFrame(
        {
            "mae": errors.abs().mean(),
            "rmse": np.sqrt((errors**2).mean()),
            "median_ae": errors.abs().median(),
        }
    )
    if "naive" in out.index:
        out["rmae_vs_naive"] = out["mae"] / out.loc["naive", "mae"]
    out.attrs["n_periods"] = len(df)
    return out


def mae_by_year(actual: pd.Series, predictions: pd.DataFrame, min_days: int = 30) -> pd.DataFrame:
    """MAE per calendar year, skipping years with fewer than `min_days` scored days."""
    df = common_rows(actual, predictions)
    errors = df[predictions.columns].sub(df["actual"], axis=0).abs()
    years = errors.index.year
    n_days = pd.Series(errors.index.date, index=errors.index).groupby(years).nunique()
    return errors.groupby(years).mean()[n_days >= min_days].rename_axis("year")


def diebold_mariano(
    actual: pd.Series, pred_a: pd.Series, pred_b: pd.Series, power: int = 1
) -> tuple[float, float]:
    """DM test of equal accuracy between forecasts A and B.

    The 24 hourly forecasts of a day are issued together, so their errors are strongly
    correlated. Following common EPF practice, the loss differential is first averaged per
    delivery day, and the test is run on that daily series. `power=1` uses absolute
    errors, `power=2` squared errors.

    Returns (statistic, two-sided p-value). A negative statistic means A is more accurate.
    """
    df = pd.concat([actual, pred_a, pred_b], axis=1, keys=["y", "a", "b"]).dropna()
    loss_diff = (df["y"] - df["a"]).abs() ** power - (df["y"] - df["b"]).abs() ** power
    daily = loss_diff.groupby(loss_diff.index.date).mean().to_numpy()
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return 0.0, 1.0
    stat = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def dm_table(actual: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    """Pairwise DM tests (absolute loss) between every pair of models."""
    df = common_rows(actual, predictions)
    rows = []
    for a, b in combinations(predictions.columns, 2):
        stat, p = diebold_mariano(df["actual"], df[a], df[b])
        rows.append({"model_a": a, "model_b": b, "dm_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def conformal_intervals(
    actual: pd.Series,
    forecast: pd.Series,
    coverage: float = 0.9,
    window_days: int = 60,
    min_days: int = 20,
) -> pd.DataFrame:
    """Rolling split-conformal prediction intervals around a point forecast.

    For each delivery period of the day (hour or quarter-hour), the half-width is the
    `coverage` quantile of that period's absolute errors over the previous `window_days`
    days. Day D's interval only uses errors of days up
    to D-1, all known at decision time (D-1's prices are published on D-2). Returns columns
    `lower` and `upper`, NaN until `min_days` days of errors are available.
    """
    errors = (actual - forecast).abs().dropna()
    slot = errors.index.hour * 60 + errors.index.minute  # period of the day
    table = errors.groupby([errors.index.date, slot]).mean().unstack()
    half_width = table.rolling(window_days, min_periods=min_days).quantile(coverage).shift(1)
    keys = pd.MultiIndex.from_arrays(
        [forecast.index.date, forecast.index.hour * 60 + forecast.index.minute]
    )
    width = half_width.stack().reindex(keys).to_numpy()
    return pd.DataFrame(
        {"lower": forecast - width, "upper": forecast + width}, index=forecast.index
    )


def interval_metrics(actual: pd.Series, intervals: pd.DataFrame, coverage: float = 0.9) -> dict:
    """Empirical coverage, mean width and interval (Winkler) score; lower score is better."""
    df = pd.concat([actual.rename("y"), intervals], axis=1).dropna()
    y, lo, hi = df["y"], df["lower"], df["upper"]
    alpha = 1 - coverage
    score = (hi - lo) + 2 / alpha * ((lo - y).clip(lower=0) + (y - hi).clip(lower=0))
    return {
        "coverage": float(((y >= lo) & (y <= hi)).mean()),
        "mean_width": float((hi - lo).mean()),
        "interval_score": float(score.mean()),
        "n_periods": len(df),
    }
