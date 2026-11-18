"""15-minute stage: resolution handling, shape features (no look-ahead), final forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fr_power import data
from fr_power.quarter_hour import (
    SHAPE_TARGET,
    build_shape_matrix,
    quarter_hour_predictions,
    quarter_prices,
    to_quarter_hour,
    within_day_ramp,
)

DAY = pd.Timestamp("2025-10-20", tz=data.TIMEZONE)  # a delivery day after the switch


def _hour_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert("UTC").floor("h").tz_convert(data.TIMEZONE)  # DST-safe


def _hourly_forecast(raw: dict[str, pd.DataFrame]) -> pd.Series:
    price = quarter_prices(raw)
    return price.resample("1h").mean()  # stands in for an hourly model forecast


def test_hourly_values_cover_their_four_quarters() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz=data.TIMEZONE)
    out = to_quarter_hour(pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index))
    assert out["x"].tolist() == [1.0] * 4 + [2.0] * 4 + [3.0]


def test_prices_are_hourly_before_the_switch_and_quarterly_after(raw_mtu) -> None:  # type: ignore[no-untyped-def]
    price = quarter_prices(raw_mtu)
    spread = price.resample("1h").agg(lambda s: s.max() - s.min())
    assert (spread[spread.index < data.MTU_15MIN_START] == 0).all()
    assert (spread[spread.index >= data.MTU_15MIN_START] > 0).mean() > 0.9


def test_ramp_stays_within_the_delivery_day() -> None:
    index = pd.date_range("2025-10-20", periods=48, freq="h", tz=data.TIMEZONE)
    hourly = pd.Series(np.arange(48.0), index=index)
    bumped = hourly.copy()
    bumped.iloc[24:] += 1_000  # forecasts for the next day
    np.testing.assert_array_equal(within_day_ramp(hourly)[:24], within_day_ramp(bumped)[:24])
    assert within_day_ramp(hourly).iloc[23] == 1.0  # one-sided slope at the day's end


def test_shape_features_only_use_information_known_before_the_gate(raw_mtu) -> None:  # type: ignore[no-untyped-def]
    forecast = _hourly_forecast(raw_mtu)
    base = build_shape_matrix(raw_mtu, forecast, profile_days=7)
    features = [c for c in base.columns if c != SHAPE_TARGET]
    on_day = base.index.normalize() == DAY

    perturbed = {k: v.copy() for k, v in raw_mtu.items()}
    prices = perturbed[data.DA_PRICES]
    prices.loc[prices.index >= DAY, "day_ahead_price_eur_mwh"] += 1_000  # day D and later
    wind_solar = perturbed[data.WIND_SOLAR_FORECAST]
    wind_solar.loc[wind_solar.index >= DAY] += 1_000  # TSO forecasts for D: after the gate
    later = forecast.copy()
    later[later.index >= DAY + pd.Timedelta(days=1)] += 1_000  # forecasts for D+1
    after = build_shape_matrix(perturbed, later, profile_days=7)

    pd.testing.assert_frame_equal(base.loc[on_day, features], after.loc[on_day, features])
    assert not base.loc[on_day, SHAPE_TARGET].equals(after.loc[on_day, SHAPE_TARGET])


def test_quarter_forecasts_are_hourly_level_plus_shape(raw_mtu) -> None:  # type: ignore[no-untyped-def]
    price = quarter_prices(raw_mtu)
    hourly = pd.DataFrame({"lasso": price.resample("1h").mean() + 5.0})
    shape = pd.Series(2.0, index=price.index[price.index >= DAY])
    out = quarter_hour_predictions(price, hourly, shape)

    before = out.index < data.MTU_15MIN_START
    level = hourly["lasso"].reindex(_hour_start(out.index[before]))
    np.testing.assert_allclose(out.loc[before, "lasso"], level)  # hourly products: no shape
    assert out.loc[(out.index >= data.MTU_15MIN_START) & (out.index < DAY), "lasso"].isna().all()
    after = out.index >= DAY
    expected = hourly["lasso"].reindex(_hour_start(out.index[after])).to_numpy() + 2.0
    np.testing.assert_allclose(out.loc[after, "lasso"], expected)

    tuesday = out.index[(out.index.dayofweek == 1) & after][0]
    assert out.loc[tuesday, "naive"] == price[tuesday - pd.Timedelta(days=1)]
    monday = out.index[(out.index.dayofweek == 0) & after][0]
    assert out.loc[monday, "naive"] == price[monday - pd.Timedelta(days=7)]
