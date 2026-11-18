"""From hourly to 15-minute forecasts.

Since 1 October 2025 the SDAC day-ahead auction clears 96 quarter-hour prices per day.
Before that, it cleared 24 hourly prices: the four quarters of an hour shared one price.

Fifteen-minute prices therefore exist for about a year only, too little to train the full
models on. The forecast is built in two stages:

1. **Hourly level.** The models of `models.py` forecast the hourly average price, trained on
   the whole history since 2019.
2. **Intra-hour shape.** A Lasso forecasts each quarter's deviation from its hourly mean,
   trained on 15-minute data only. Before the switch the shape is exactly zero.

Shape features, all known on D-1 before the 12:00 gate:

* the quarter's position in the hour (-1.5, -0.5, 0.5, 1.5);
* the same quarter's deviation on D-1, and its mean over the last 7 and 28 days;
* the ramp of the hourly price forecast around that hour, times the quarter position (prices
  rising through the hour push early quarters down and late ones up). The ramp only uses
  forecasts for day D itself: the forecast for D+1 00:00 would include day D's prices;
* the within-hour variation of the TSO load forecast for D (published before the gate) and
  of the wind and solar forecasts for D-1 (those for D come after the gate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fr_power.config import LassoConfig
from fr_power.data import DA_PRICES, LOAD_FORECAST, MTU_15MIN_START, TIMEZONE, WIND_SOLAR_FORECAST
from fr_power.models import LassoForecaster, walk_forward_predict

SHAPE_TARGET = "intra_hour_deviation"
QUARTERS_PER_DAY = 96


def to_quarter_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Put a tz-aware hourly or 15-minute frame on a regular 15-minute Paris-time grid.

    Hourly values apply to the four quarters of their hour.
    """
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert(TIMEZONE)
    return df.resample("15min").mean().ffill(limit=3)


def quarter_prices(raw: dict[str, pd.DataFrame]) -> pd.Series:
    """Price of every 15-minute delivery period (the hourly price before the switch)."""
    return to_quarter_hour(raw[DA_PRICES])["day_ahead_price_eur_mwh"].rename("price_15min")


def within_hour(series: pd.Series) -> pd.Series:
    """Deviation of each quarter from the mean of its hour."""
    return series - series.resample("1h").transform("mean")


def broadcast_hourly(hourly: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Repeat hourly values on the four quarters of each hour of `grid`."""
    union = hourly.reindex(hourly.index.union(grid))
    return union.ffill(limit=3).reindex(grid)


def within_day_ramp(hourly: pd.Series) -> pd.Series:
    """Centred hourly slope (EUR/MWh per hour) using only values of the same local day."""
    by_day = hourly.groupby(hourly.index.date)
    nxt, prv = by_day.shift(-1), by_day.shift(1)
    return ((nxt - prv) / 2).fillna(nxt - hourly).fillna(hourly - prv)


def build_shape_matrix(
    raw: dict[str, pd.DataFrame], hourly_forecast: pd.Series, profile_days: int = 28
) -> pd.DataFrame:
    """15-minute features and `SHAPE_TARGET`, from the 15-minute switch onwards."""
    price = quarter_prices(raw)
    grid = price.index[price.index >= MTU_15MIN_START]
    deviation = within_hour(price)
    deviation[price.index < MTU_15MIN_START] = np.nan  # hourly products: no shape to learn

    lags = pd.concat(
        {k: deviation.shift(QUARTERS_PER_DAY * k) for k in range(1, profile_days + 1)}, axis=1
    )
    qpos = pd.Series(price.index.minute / 15 - 1.5, index=price.index)
    ramp = broadcast_hourly(within_day_ramp(hourly_forecast).to_frame(), price.index).iloc[:, 0]

    load = to_quarter_hour(raw[LOAD_FORECAST]).iloc[:, 0].reindex(price.index)
    # Wind and solar forecasts for D come after the gate: use those for D-1.
    wind_solar = to_quarter_hour(raw[WIND_SOLAR_FORECAST]).reindex(price.index)
    wind_solar = wind_solar.shift(QUARTERS_PER_DAY)

    def total(word: str) -> pd.Series:
        cols = [c for c in wind_solar.columns if word in c.lower()]
        return wind_solar[cols].sum(axis=1, min_count=1)

    matrix = pd.DataFrame(
        {
            "quarter_position": qpos,
            "deviation_lag_24h": deviation.shift(QUARTERS_PER_DAY),
            "deviation_mean_7d": lags.iloc[:, :7].mean(axis=1),
            f"deviation_mean_{profile_days}d": lags.mean(axis=1),
            "price_ramp_x_position": ramp * qpos,
            "load_forecast_within_hour": within_hour(load),
            "solar_forecast_prev_day_within_hour": within_hour(total("solar")),
            "wind_forecast_prev_day_within_hour": within_hour(total("wind")),
            SHAPE_TARGET: deviation,
        }
    )
    return matrix.reindex(grid)


def quarter_hour_predictions(
    price: pd.Series, hourly_predictions: pd.DataFrame, shape: pd.Series
) -> pd.DataFrame:
    """15-minute forecasts: hourly level plus intra-hour shape, and a 15-minute naive.

    Before the switch the shape is zero (hourly products). After it, quarters without a
    shape forecast (the shape model's first training window) stay NaN. The naive benchmark
    repeats each quarter's price from one week earlier on Mondays and weekends, one day
    earlier otherwise, as the hourly naive does.
    """
    grid = price.index
    shape_q = shape.reindex(grid)
    shape_q[grid < MTU_15MIN_START] = 0.0
    out = broadcast_hourly(hourly_predictions, grid).add(shape_q, axis=0)
    weekly = pd.Series(grid.dayofweek, index=grid).isin([0, 5, 6])
    daily_lag = price.shift(QUARTERS_PER_DAY)
    out["naive"] = price.shift(7 * QUARTERS_PER_DAY).where(weekly, daily_lag)
    return out


def predict_shape(
    matrix: pd.DataFrame,
    lasso: LassoConfig,
    initial_train_days: int,
    retrain_every_days: int,
    random_seed: int = 42,
) -> pd.Series:
    """Walk-forward out-of-sample forecasts of the intra-hour deviation (NaN if no data)."""
    if matrix.empty or matrix[SHAPE_TARGET].notna().sum() == 0:
        return pd.Series(np.nan, index=matrix.index, name="shape")
    features = [c for c in matrix.columns if c != SHAPE_TARGET]
    predictions = walk_forward_predict(
        matrix[features],
        matrix[SHAPE_TARGET],
        {"shape": lambda: LassoForecaster(lasso, random_seed)},
        initial_train_days=initial_train_days,
        retrain_every_days=retrain_every_days,
    )
    return predictions["shape"]
