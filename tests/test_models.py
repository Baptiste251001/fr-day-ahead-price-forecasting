"""Models, walk-forward harness and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fr_power.config import Config
from fr_power.evaluation import (
    conformal_intervals,
    diebold_mariano,
    dm_table,
    interval_metrics,
    mae_by_year,
    point_metrics,
)
from fr_power.features import TARGET, build_feature_matrix, feature_columns
from fr_power.models import (
    NaiveForecaster,
    add_ensemble,
    build_models,
    walk_forward_predict,
)


@pytest.fixture
def matrix(raw: dict[str, pd.DataFrame], config: Config) -> pd.DataFrame:
    return build_feature_matrix(raw, config.features)


def test_naive_uses_last_week_on_mondays_and_weekends_else_yesterday() -> None:
    X = pd.DataFrame(
        {"day_of_week": [0, 1, 4, 5, 6], "price_lag_24h": [1.0] * 5, "price_lag_168h": [7.0] * 5}
    )
    model = NaiveForecaster()
    model.fit(X, pd.Series(0.0, index=X.index))
    assert model.predict(X).tolist() == [7.0, 1.0, 1.0, 7.0, 7.0]


def test_every_model_beats_noise(matrix: pd.DataFrame, config: Config) -> None:
    X, y = matrix[feature_columns(matrix)], matrix[TARGET]
    for name, make_model in build_models(config).items():
        model = make_model()
        model.fit(X.iloc[:900], y.iloc[:900])
        preds = model.predict(X.iloc[900:])
        assert preds.notna().mean() > 0.9, name
        assert (preds - y.iloc[900:]).abs().mean() < y.std(), name


def test_walk_forward_never_trains_on_the_future(matrix: pd.DataFrame) -> None:
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class Recorder:
        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            self.train_end = X.index.max()

        def predict(self, X: pd.DataFrame) -> pd.Series:
            seen.append((self.train_end, X.index.min()))
            return pd.Series(1.0, index=X.index)

    X, y = matrix[feature_columns(matrix)], matrix[TARGET]
    preds = walk_forward_predict(
        X, y, {"rec": Recorder}, initial_train_days=10, retrain_every_days=7
    )

    assert len(seen) == int(np.ceil((60 - 10) / 7))
    assert all(train_end < test_start for train_end, test_start in seen)
    assert all(test_start.hour == 0 for _, test_start in seen)  # folds start at midnight
    assert preds["rec"].iloc[: 10 * 24].isna().all() and preds["rec"].iloc[10 * 24 :].notna().all()


def test_point_metrics_on_known_errors() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    actual = pd.Series([10.0, 10.0, 10.0, 10.0], index=idx)
    preds = pd.DataFrame(
        {"naive": [12.0, 8.0, 12.0, 8.0], "good": [11.0, 9.0, 10.0, np.nan]}, index=idx
    )

    metrics = point_metrics(actual, preds)

    assert metrics.attrs["n_hours"] == 3  # scored only where every model has a forecast
    assert metrics.loc["naive", "mae"] == 2.0
    assert metrics.loc["good", "rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics.loc["good", "rmae_vs_naive"] == pytest.approx((2 / 3) / 2)
    assert mae_by_year(actual, preds, min_days=0).loc[2024, "naive"] == 2.0
    assert mae_by_year(actual, preds).empty  # 3 hours is not a meaningful year


def test_diebold_mariano_detects_a_better_forecast() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    actual = pd.Series(rng.normal(50, 10, len(idx)), index=idx)
    good = actual + rng.normal(0, 1, len(idx))
    bad = actual + rng.normal(0, 5, len(idx))

    stat, p = diebold_mariano(actual, good, bad)
    assert stat < 0 and p < 0.01

    _, p_same = diebold_mariano(actual, good, good)
    assert p_same == 1.0

    table = dm_table(actual, pd.DataFrame({"good": good, "bad": bad}))
    assert table[["model_a", "model_b"]].values.tolist() == [["good", "bad"]]


def test_walk_forward_folds_stay_on_midnight_across_dst() -> None:
    idx = pd.date_range("2024-03-01", periods=24 * 60, freq="h", tz="Europe/Paris")
    X = pd.DataFrame(
        {"day_of_week": idx.dayofweek, "price_lag_24h": 1.0, "price_lag_168h": 2.0}, index=idx
    )
    preds = walk_forward_predict(X, pd.Series(0.0, index=idx), {"naive": NaiveForecaster}, 10, 7)
    first_predicted = preds["naive"].notna().groupby(preds.index.date).first()
    assert first_predicted.loc[pd.Timestamp("2024-04-01").date()]  # a fold start after DST
    assert preds["naive"].notna().groupby(preds.index.date).nunique().eq(1).all()  # whole days


def test_ensemble_is_the_hourly_median() -> None:
    preds = pd.DataFrame({"a": [1.0, 5.0], "b": [2.0, np.nan], "c": [10.0, 1.0], "naive": 0.0})
    out = add_ensemble(preds, ["a", "b", "c"])
    assert out["ensemble"].iloc[0] == 2.0
    assert np.isnan(out["ensemble"].iloc[1])  # a member without a forecast -> no ensemble


def test_conformal_intervals_use_only_past_days_and_reach_coverage() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=24 * 200, freq="h", tz="Europe/Paris")
    forecast = pd.Series(50.0, index=idx)
    actual = forecast + rng.normal(0, 10, len(idx))

    intervals = conformal_intervals(actual, forecast, coverage=0.9, window_days=60)
    assert intervals.iloc[: 24 * 20].isna().all().all()  # not enough history yet

    # Changing day D's outcome must not change day D's interval.
    day = idx[24 * 100 : 24 * 101]
    shocked = actual.copy()
    shocked[day] += 1_000
    pd.testing.assert_frame_equal(
        intervals.loc[day], conformal_intervals(shocked, forecast, 0.9, 60).loc[day]
    )

    metrics = interval_metrics(actual, intervals, coverage=0.9)
    assert 0.86 < metrics["coverage"] < 0.94
    assert metrics["mean_width"] == pytest.approx(2 * 1.645 * 10, rel=0.1)
