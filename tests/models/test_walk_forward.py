"""Tests for the generic walk-forward evaluation harness."""

import numpy as np
import pandas as pd

from fr_power.models.walk_forward import walk_forward_predict


class _RecordingMeanForecaster:
    """Predicts the training-target mean; records how many rows it was trained on."""

    instances: list["_RecordingMeanForecaster"] = []

    def __init__(self) -> None:
        self.n_train_rows: int | None = None
        self.mean_: float = 0.0
        _RecordingMeanForecaster.instances.append(self)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.n_train_rows = len(X)
        self.mean_ = float(y.mean())

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.mean_, index=X.index)


def _dataset(n: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float)}, index=index)
    return df


def test_walk_forward_predict_covers_out_of_sample_period_only() -> None:
    df = _dataset(n=240)
    predictions = walk_forward_predict(
        df,
        feature_cols=["x"],
        target_col="y",
        model_factories={"mean": _RecordingMeanForecaster},
        initial_train_hours=100,
        retrain_every_hours=50,
        test_window_hours=50,
    )

    preds = predictions["mean"]
    assert preds.iloc[:100].isna().all()  # initial training period never predicted
    assert preds.iloc[100:].notna().all()


def test_walk_forward_predict_retrains_with_expanding_window() -> None:
    _RecordingMeanForecaster.instances = []
    df = _dataset(n=240)
    walk_forward_predict(
        df,
        feature_cols=["x"],
        target_col="y",
        model_factories={"mean": _RecordingMeanForecaster},
        initial_train_hours=100,
        retrain_every_hours=50,
        test_window_hours=50,
    )

    train_sizes = [m.n_train_rows for m in _RecordingMeanForecaster.instances]
    assert train_sizes == [100, 150, 200]  # expanding, not fixed-size, window
    assert train_sizes == sorted(train_sizes)


def test_walk_forward_predict_multiple_models_independent() -> None:
    df = _dataset(n=200)

    class _ConstantForecaster:
        def __init__(self, value: float) -> None:
            self.value = value

        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            pass

        def predict(self, X: pd.DataFrame) -> pd.Series:
            return pd.Series(self.value, index=X.index)

    predictions = walk_forward_predict(
        df,
        feature_cols=["x"],
        target_col="y",
        model_factories={
            "a": lambda: _ConstantForecaster(1.0),
            "b": lambda: _ConstantForecaster(2.0),
        },
        initial_train_hours=100,
        retrain_every_hours=100,
        test_window_hours=100,
    )

    assert (predictions["a"].dropna() == 1.0).all()
    assert (predictions["b"].dropna() == 2.0).all()


def test_walk_forward_predict_returns_all_nan_when_no_test_period() -> None:
    df = _dataset(n=50)
    predictions = walk_forward_predict(
        df,
        feature_cols=["x"],
        target_col="y",
        model_factories={"mean": _RecordingMeanForecaster},
        initial_train_hours=100,  # exceeds available rows
        retrain_every_hours=10,
        test_window_hours=10,
    )
    assert predictions["mean"].isna().all()
