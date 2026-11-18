"""Point-in-time feature matrix for forecasting the FR day-ahead price.

Information set
---------------
The forecast for the 24 hourly prices of delivery day D is made on D-1, shortly before the
day-ahead auction gate closure (12:00 CET). At that moment we know:

* every day-ahead price up to and including day D-1 (published on D-2), so price lags and
  rolling statistics are computed on prices shifted by at least 24h;
* the TSO's day-ahead load forecast and wind & solar forecasts *for day D* -- used as-is;
* realized generation by type and observed temperature only up to the morning of D-1. To be
  safe for every delivery hour they are lagged by `realized_lag_hours` (>= 48h), i.e. the
  same hour two days earlier: a persistence proxy for nuclear/hydro availability and weather;
* calendar information (hour, weekday, month, French public holidays).

ENTSO-E data moved from hourly to 15-minute resolution for several series (EU 15-minute
market time unit), so every input is first averaged onto an hourly grid.
"""

from __future__ import annotations

import re

import pandas as pd
from holidays import France

from fr_power.config import FeaturesConfig
from fr_power.data import (
    DA_PRICES,
    GENERATION,
    LOAD_FORECAST,
    TIMEZONE,
    WEATHER,
    WIND_SOLAR_FORECAST,
)

TARGET = "target_price_eur_mwh"


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Average a (possibly 15-minute) tz-aware series onto a regular hourly Paris-time grid."""
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert(TIMEZONE)
    return df.resample("1h").mean()


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    fr_holidays = France(years=range(index.year.min(), index.year.max() + 1))
    return pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_holiday": [int(d in fr_holidays) for d in index.date],
        },
        index=index,
    )


def fill_leading_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Set to 0 the leading all-NaN stretch of each column, leaving interior gaps as NaN.

    ENTSO-E added production types over time (offshore wind from 2023, storage from 2025):
    before a column first reports, the technology was simply not there (~0 MW). Without
    this, a single late-starting column would make every early row incomplete.
    """
    out = df.copy()
    for col in out.columns:
        first_valid = out[col].first_valid_index()
        if first_valid is not None:
            out.loc[out.index < first_valid, col] = 0.0
    return out


def daily_to_hourly(daily: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Broadcast a daily series onto every hour of its local calendar date."""
    lookup = daily.set_axis(pd.DatetimeIndex(daily.index).date)
    return lookup.reindex(hourly_index.date).set_axis(hourly_index)


def _slug(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def build_feature_matrix(raw: dict[str, pd.DataFrame], config: FeaturesConfig) -> pd.DataFrame:
    """Assemble the hourly feature matrix (features + `TARGET` column) from raw datasets."""
    price = to_hourly(raw[DA_PRICES])["day_ahead_price_eur_mwh"]
    grid = price.index

    wind_solar = to_hourly(raw[WIND_SOLAR_FORECAST]).reindex(grid)
    wind = wind_solar[[c for c in wind_solar.columns if "wind" in c.lower()]].sum(
        axis=1, min_count=1
    )
    solar = wind_solar[[c for c in wind_solar.columns if "solar" in c.lower()]].sum(
        axis=1, min_count=1
    )
    load = to_hourly(raw[LOAD_FORECAST]).reindex(grid).iloc[:, 0]

    lag = config.realized_lag_hours
    generation = fill_leading_gaps(to_hourly(raw[GENERATION]).reindex(grid)).shift(lag)
    generation.columns = [f"gen_{_slug(c)}_lag{lag}h" for c in generation.columns]
    weather = daily_to_hourly(raw[WEATHER], grid).shift(lag)
    weather.columns = [f"{c}_lag{lag}h" for c in weather.columns]

    price_feats = {f"price_lag_{h}h": price.shift(h) for h in config.price_lags_hours}
    known = price.shift(24)  # most recent prices fully known before the auction
    for w in config.price_rolling_windows_hours:
        price_feats[f"price_mean_{w}h"] = known.rolling(w, min_periods=w // 2).mean()
        price_feats[f"price_std_{w}h"] = known.rolling(w, min_periods=w // 2).std()

    forecasts = pd.DataFrame(
        {
            "load_forecast_mw": load,
            "wind_forecast_mw": wind,
            "solar_forecast_mw": solar,
            "residual_load_forecast_mw": load - wind.fillna(0) - solar.fillna(0),
        },
        index=grid,
    )
    return pd.concat(
        [
            calendar_features(grid),
            forecasts,
            pd.DataFrame(price_feats, index=grid),
            generation,
            weather,
            price.rename(TARGET),
        ],
        axis=1,
    )


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    return [c for c in matrix.columns if c != TARGET]
