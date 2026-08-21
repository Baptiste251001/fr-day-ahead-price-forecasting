"""Tests for the LSTM quantile forecaster.

Kept deliberately small (short sequences, tiny network, few epochs) so
these run in a few seconds on CPU -- they check structural/point-in-time
correctness, not forecast quality.
"""

import numpy as np
import pandas as pd
import pytest

from fr_power.models.lstm import LSTMQuantileForecaster


def _dataset(n: int = 150, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 * x1 + rng.normal(scale=0.2, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=index)
    return X, pd.Series(y, index=index)


def _small_model(quantiles: list[float] | None = None) -> LSTMQuantileForecaster:
    return LSTMQuantileForecaster(
        sequence_length_hours=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=16,
        learning_rate=0.01,
        max_epochs=5,
        early_stopping_patience=3,
        quantiles=quantiles or [0.1, 0.5, 0.9],
        random_seed=0,
    )


def test_lstm_predict_quantiles_shape_and_columns() -> None:
    X, y = _dataset()
    model = _small_model()
    model.fit(X, y)

    quantile_df = model.predict_quantiles(X)

    assert list(quantile_df.columns) == ["q10", "q50", "q90"]
    assert len(quantile_df) == len(X)
    assert quantile_df.iloc[: model.sequence_length_hours].isna().all().all()
    assert quantile_df.iloc[model.sequence_length_hours :].notna().all().all()


def test_lstm_predict_returns_median_quantile() -> None:
    X, y = _dataset()
    model = _small_model()
    model.fit(X, y)

    point = model.predict(X)
    quantile_df = model.predict_quantiles(X)

    pd.testing.assert_series_equal(point, quantile_df["q50"].rename("prediction"))


def test_lstm_skips_windows_with_nan_features() -> None:
    X, y = _dataset(n=100)
    model = _small_model()
    model.fit(X, y)

    X_with_nan = X.copy()
    X_with_nan.iloc[50, 0] = np.nan  # poisons every window that includes row 50

    preds = model.predict(X_with_nan)

    affected = preds.iloc[51 : 51 + model.sequence_length_hours]
    assert affected.isna().all()
    assert preds.iloc[55] is not None and not np.isnan(preds.iloc[55])


def test_lstm_raises_when_too_few_windows() -> None:
    X, y = _dataset(n=3)
    model = _small_model()
    with pytest.raises(ValueError, match="Not enough complete windows"):
        model.fit(X, y)


def test_lstm_median_index_falls_back_when_no_exact_median_quantile() -> None:
    model = _small_model(quantiles=[0.2, 0.8])
    assert model.median_index_ == 1
