"""Tests for the LightGBM forecaster."""

import numpy as np
import pandas as pd

from fr_power.models.gbm import LightGBMForecaster


def _dataset(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.where(x1 > 0, 10.0, -10.0) + 0.5 * x2 + rng.normal(scale=0.1, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=index)
    return X, pd.Series(y, index=index)


def test_lightgbm_forecaster_fits_and_predicts() -> None:
    X, y = _dataset()
    model = LightGBMForecaster(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=7,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
    )
    model.fit(X, y)
    preds = model.predict(X)

    residuals = (y - preds).dropna()
    assert residuals.abs().mean() < 2.0


def test_lightgbm_forecaster_handles_nan_features() -> None:
    X, y = _dataset()
    X_with_nan = X.copy()
    X_with_nan.iloc[::10, 0] = np.nan  # every 10th row has a missing feature

    model = LightGBMForecaster(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=7,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
    )
    model.fit(X_with_nan, y)
    preds = model.predict(X_with_nan)

    assert preds.notna().all()  # LightGBM handles NaN features natively, no dropped rows


def test_lightgbm_forecaster_drops_nan_targets_during_fit() -> None:
    X, y = _dataset(n=200)
    y_with_nan = y.copy()
    y_with_nan.iloc[:5] = np.nan

    model = LightGBMForecaster(
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=7,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
    )
    model.fit(X, y_with_nan)  # should not raise despite NaN targets
    preds = model.predict(X)
    assert len(preds) == len(X)
