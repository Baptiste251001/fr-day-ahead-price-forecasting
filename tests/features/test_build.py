"""Tests for the point-in-time feature matrix builder.

The core concern here is point-in-time correctness: every lagged/rolling
feature must be built only from data strictly earlier than the row it
appears on. We use sentinel data (values equal to a known, traceable
integer position) so leakage would show up as a wrong number, not just a
wrong shape.
"""

import numpy as np
import pandas as pd
import pytest

from fr_power.config import FeaturesConfig
from fr_power.features.build import (
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
    compute_renewable_forecast_mw,
    compute_residual_load_mw,
    expand_daily_to_hourly,
    lag_realized_series,
)


def _grid(n: int = 200, start: str = "2024-06-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="Europe/Paris")


def test_add_lag_features_shifts_correctly() -> None:
    index = _grid(50)
    series = pd.Series(np.arange(len(index), dtype=float), index=index)

    lagged = add_lag_features(series, [24], "x")

    assert lagged["x_lag_24h"].iloc[24] == series.iloc[0]
    assert np.isnan(lagged["x_lag_24h"].iloc[23])
    assert lagged["x_lag_24h"].iloc[49] == series.iloc[25]


def test_add_rolling_features_excludes_current_value() -> None:
    index = _grid(50)
    series = pd.Series(np.arange(len(index), dtype=float), index=index)

    rolled = add_rolling_features(series, [5], "x", min_lag_hours=1)

    # rollmean at t=10 should be mean of series[5..9] (shift(1) then rolling(5))
    expected = series.iloc[5:10].mean()
    assert rolled["x_rollmean_5h"].iloc[10] == expected
    # must never include series.iloc[10] itself
    assert expected != series.iloc[10]


def test_lag_realized_series_shifts_and_renames() -> None:
    index = _grid(50)
    df = pd.DataFrame({"Nuclear": np.arange(len(index), dtype=float) * 10}, index=index)

    lagged = lag_realized_series(df, 24, "generation")

    assert list(lagged.columns) == ["generation_Nuclear_lag24h"]
    assert lagged["generation_Nuclear_lag24h"].iloc[24] == df["Nuclear"].iloc[0]
    assert np.isnan(lagged["generation_Nuclear_lag24h"].iloc[0])


def test_compute_renewable_forecast_mw_sums_wind_and_solar() -> None:
    index = _grid(3)
    df = pd.DataFrame(
        {
            "Solar": [0.0, 100.0, 500.0],
            "Wind Onshore": [10.0, 20.0, 30.0],
            "Wind Offshore": [1.0, 2.0, 3.0],
        },
        index=index,
    )
    out = compute_renewable_forecast_mw(df)
    assert out["wind_forecast_mw"].tolist() == [11.0, 22.0, 33.0]
    assert out["solar_forecast_mw"].tolist() == [0.0, 100.0, 500.0]


def test_compute_residual_load_mw() -> None:
    index = _grid(3)
    load = pd.Series([1000.0, 1000.0, 1000.0], index=index)
    renewables = pd.DataFrame(
        {"wind_forecast_mw": [100.0, 0.0, 50.0], "solar_forecast_mw": [50.0, 0.0, 0.0]}, index=index
    )
    residual = compute_residual_load_mw(load, renewables)
    assert residual.tolist() == [850.0, 1000.0, 950.0]


def test_expand_daily_to_hourly_broadcasts_each_day() -> None:
    hourly_index = _grid(48, start="2024-06-01")  # 2024-06-01 and 2024-06-02
    daily = pd.DataFrame({"hdd": [7.0, 3.0]}, index=pd.to_datetime(["2024-06-01", "2024-06-02"]))

    expanded = expand_daily_to_hourly(daily, hourly_index)

    assert (expanded["hdd"].iloc[:24] == 7.0).all()
    assert (expanded["hdd"].iloc[24:] == 3.0).all()


@pytest.fixture
def features_config() -> FeaturesConfig:
    return FeaturesConfig(
        price_lags_hours=[24, 48],
        price_rolling_windows_hours=[24],
        spread_lags_hours=[24],
        generation_availability_lag_hours=24,
        weather_availability_lag_hours=24,
    )


def _synthetic_inputs(n: int = 200) -> dict:
    index = _grid(n)
    day_ahead_prices = pd.DataFrame(
        {"day_ahead_price_eur_mwh": np.arange(n, dtype=float)}, index=index
    )
    wind_solar_forecast = pd.DataFrame(
        {"Solar": np.full(n, 10.0), "Wind Onshore": np.full(n, 20.0)}, index=index
    )
    load_forecast = pd.DataFrame({"Forecasted Load": np.full(n, 1000.0)}, index=index)
    actual_generation_by_type = pd.DataFrame(
        {"Nuclear": np.arange(n, dtype=float) * 10}, index=index
    )
    net_position = pd.DataFrame({"net_position_mw": np.zeros(n)}, index=index)

    daily_dates = pd.date_range("2024-05-25", periods=20, freq="D")
    weather = pd.DataFrame({"hdd": np.arange(len(daily_dates), dtype=float)}, index=daily_dates)

    intraday = pd.DataFrame(
        {"intraday_price_eur_mwh": day_ahead_prices["day_ahead_price_eur_mwh"] + 5.0}, index=index
    )
    return {
        "day_ahead_prices": day_ahead_prices,
        "wind_solar_forecast": wind_solar_forecast,
        "load_forecast": load_forecast,
        "actual_generation_by_type": actual_generation_by_type,
        "net_position": net_position,
        "weather": weather,
        "intraday": intraday,
    }


def test_build_feature_matrix_generation_lag_matches_point_in_time(
    features_config: FeaturesConfig,
) -> None:
    inputs = _synthetic_inputs()
    matrix = build_feature_matrix(
        inputs["day_ahead_prices"],
        inputs["wind_solar_forecast"],
        inputs["load_forecast"],
        inputs["actual_generation_by_type"],
        inputs["net_position"],
        features_config,
        weather=inputs["weather"],
        intraday=None,
    )

    raw_generation = inputs["actual_generation_by_type"]["Nuclear"]
    col = "generation_Nuclear_lag24h"
    for t in [30, 100, 199]:
        assert matrix[col].iloc[t] == raw_generation.iloc[t - 24]
    # never equal to same-hour realized generation (that would be a leak)
    assert (matrix[col].dropna() != raw_generation.reindex(matrix[col].dropna().index)).all()


def test_build_feature_matrix_weather_lag_uses_previous_day(
    features_config: FeaturesConfig,
) -> None:
    inputs = _synthetic_inputs()
    matrix = build_feature_matrix(
        inputs["day_ahead_prices"],
        inputs["wind_solar_forecast"],
        inputs["load_forecast"],
        inputs["actual_generation_by_type"],
        inputs["net_position"],
        features_config,
        weather=inputs["weather"],
        intraday=None,
    )

    daily_hdd = inputs["weather"]["hdd"]
    target_ts = pd.Timestamp("2024-06-03 05:00", tz="Europe/Paris")
    expected = daily_hdd.loc[pd.Timestamp("2024-06-02")]  # previous local calendar day
    assert matrix.loc[target_ts, "weather_hdd_lag24h"] == expected


def test_build_feature_matrix_without_intraday_excludes_current_price_feature(
    features_config: FeaturesConfig,
) -> None:
    inputs = _synthetic_inputs()
    matrix = build_feature_matrix(
        inputs["day_ahead_prices"],
        inputs["wind_solar_forecast"],
        inputs["load_forecast"],
        inputs["actual_generation_by_type"],
        inputs["net_position"],
        features_config,
        weather=inputs["weather"],
        intraday=None,
    )

    assert matrix.attrs["target_column"] == "target_day_ahead_price_eur_mwh"
    assert "day_ahead_price_eur_mwh" not in matrix.columns
    pd.testing.assert_series_equal(
        matrix["target_day_ahead_price_eur_mwh"],
        inputs["day_ahead_prices"]["day_ahead_price_eur_mwh"],
        check_names=False,
    )


def test_build_feature_matrix_with_intraday_uses_spread_target(
    features_config: FeaturesConfig,
) -> None:
    inputs = _synthetic_inputs()
    matrix = build_feature_matrix(
        inputs["day_ahead_prices"],
        inputs["wind_solar_forecast"],
        inputs["load_forecast"],
        inputs["actual_generation_by_type"],
        inputs["net_position"],
        features_config,
        weather=inputs["weather"],
        intraday=inputs["intraday"],
    )

    assert matrix.attrs["target_column"] == "target_spread_eur_mwh"
    assert "day_ahead_price_eur_mwh" in matrix.columns
    assert (matrix["target_spread_eur_mwh"] == 5.0).all()
