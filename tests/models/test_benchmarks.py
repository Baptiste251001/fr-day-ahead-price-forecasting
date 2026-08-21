"""Tests for the naive persistence and LEAR benchmark forecasters."""

import numpy as np
import pandas as pd
import pytest

from fr_power.models.benchmarks import LearForecaster, NaivePersistenceForecaster, naive_lag_column


def test_naive_lag_column_maps_target_to_prefix() -> None:
    assert naive_lag_column("target_spread_eur_mwh", 168) == "spread_lag_168h"
    assert naive_lag_column("target_day_ahead_price_eur_mwh", 24) == "day_ahead_price_lag_24h"


def test_naive_lag_column_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unknown target column"):
        naive_lag_column("something_else", 24)


def test_naive_persistence_forecaster_reads_lag_column() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="h", tz="Europe/Paris")
    X = pd.DataFrame({"day_ahead_price_lag_24h": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    y = pd.Series([np.nan] * 5, index=index)

    model = NaivePersistenceForecaster("day_ahead_price_lag_24h")
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_naive_persistence_forecaster_requires_lag_column_present() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="Europe/Paris")
    X = pd.DataFrame({"other_col": [1.0, 2.0, 3.0]}, index=index)
    model = NaivePersistenceForecaster("day_ahead_price_lag_24h")
    with pytest.raises(ValueError, match="day_ahead_price_lag_24h"):
        model.fit(X, pd.Series(index=index, dtype=float))


def _linear_dataset(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.1, size=n)
    y = 3.0 * x1 - 2.0 * x2 + noise
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3_irrelevant": rng.normal(size=n)}, index=index)
    return X, pd.Series(y, index=index)


def test_lear_forecaster_recovers_linear_signal() -> None:
    X, y = _linear_dataset()
    model = LearForecaster(alpha_grid=[0.001, 0.01, 0.1, 1.0], cv_folds=3)
    model.fit(X, y)
    preds = model.predict(X)

    residuals = (y - preds).dropna()
    assert residuals.abs().mean() < 0.5  # should fit the strong linear signal well


def test_lear_forecaster_predicts_nan_for_incomplete_rows() -> None:
    X, y = _linear_dataset()
    model = LearForecaster(alpha_grid=[0.01, 0.1], cv_folds=3)
    model.fit(X, y)

    X_missing = X.copy()
    X_missing.iloc[0, 0] = np.nan
    preds = model.predict(X_missing)

    assert np.isnan(preds.iloc[0])
    assert not np.isnan(preds.iloc[1])


def test_lear_forecaster_raises_with_too_few_rows() -> None:
    X, y = _linear_dataset(n=3)
    model = LearForecaster(alpha_grid=[0.1], cv_folds=5)
    with pytest.raises(ValueError, match="Not enough complete rows"):
        model.fit(X, y)
