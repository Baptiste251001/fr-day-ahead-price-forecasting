"""Point-in-time feature matrix for FR intraday-vs-day-ahead spread forecasting.

Information timeline
---------------------
This project targets the ~13:00 CET/CEST D-1 decision point: for every hour
of day D, the day-ahead (DA) auction has just cleared and its price is
public. At that same moment, what is knowable about day D is:

  * The DA price itself, for every hour of day D
    (`day_ahead_price_eur_mwh`).
  * ENTSO-E's day-ahead wind & solar generation forecast for day D and
    day-ahead load forecast for day D -- these are themselves D-1
    forecasts *of* D, published alongside the auction result, so they are
    used as-is, unlagged (`wind_forecast_mw`, `solar_forecast_mw`,
    `load_forecast_mw`, and the derived `residual_load_forecast_mw`).
  * ENTSO-E's day-ahead net cross-border position for day D
    (`query_net_position(..., dayahead=True)`) -- also a D-1-published
    quantity for delivery day D, used as-is when supplied. This series is
    optional (`net_position=None` omits it): a live run found ENTSO-E's FR
    feed for it discontinued from 2025-09-30 onward, and `configs/default.yaml`
    disables fetching it by default as a result -- see `build_feature_matrix`.
  * Realized (actual) generation by production type is NOT known for day D
    at this point -- it is only published as the operating day unfolds.
    It is used lagged by `features.generation_availability_lag_hours`
    (default 24h): the most recent generation known as of D-1 13:00, as a
    proxy for structural conditions (nuclear/hydro availability) likely to
    persist into day D.
  * Actual temperature for day D is likewise not known at D-1 13:00, and no
    free, long-history archive of *historical weather forecasts* exists.
    This project therefore uses lagged *actual* temperature/HDD/CDD
    (`features.weather_availability_lag_hours`, default 24h) as a
    persistence proxy for expected weather on day D. This is a documented
    simplification -- see reports/conclusions.md (Phase 6) for its
    implications.
  * Calendar features (hour, day of week, month, FR holidays, DST
    transition flag) are always known in advance.
  * Lagged/rolling features of the DA price -- and, when intraday data is
    available, of the intraday-vs-DA spread -- use only strictly earlier
    delivery hours, so they never look ahead.

Target
------
If intraday data is available, the target is the intraday-vs-day-ahead
spread (`target_spread_eur_mwh` = intraday price - DA price) for each
hourly product. If no intraday data is configured (the common case; see
`fr_power.data.intraday`), the pipeline degrades to forecasting the DA
price level itself (`target_day_ahead_price_eur_mwh`), and the *current
hour's* DA price is excluded from the feature set in that case (only its
lags/rolling stats are used) so the model is never handed its own target.
"""

from __future__ import annotations

import logging

import pandas as pd

from fr_power.config import FeaturesConfig
from fr_power.features.calendar import add_calendar_features

logger = logging.getLogger(__name__)


def compute_renewable_forecast_mw(wind_solar_forecast: pd.DataFrame) -> pd.DataFrame:
    """Sum wind (onshore + offshore) and solar day-ahead forecast columns into MW totals."""
    wind_cols = [c for c in wind_solar_forecast.columns if "wind" in c.lower()]
    solar_cols = [c for c in wind_solar_forecast.columns if "solar" in c.lower()]
    index = wind_solar_forecast.index
    return pd.DataFrame(
        {
            "wind_forecast_mw": wind_solar_forecast[wind_cols].sum(axis=1) if wind_cols else 0.0,
            "solar_forecast_mw": wind_solar_forecast[solar_cols].sum(axis=1) if solar_cols else 0.0,
        },
        index=index,
    )


def compute_residual_load_mw(
    load_forecast_mw: pd.Series, renewable_forecast: pd.DataFrame
) -> pd.Series:
    """Residual load = load forecast - wind forecast - solar forecast (a key EPF driver)."""
    return (
        load_forecast_mw
        - renewable_forecast["wind_forecast_mw"]
        - renewable_forecast["solar_forecast_mw"]
    ).rename("residual_load_forecast_mw")


def add_lag_features(series: pd.Series, lags_hours: list[int], prefix: str) -> pd.DataFrame:
    """Lag `series` by each of `lags_hours` hourly delivery periods."""
    return pd.DataFrame(
        {f"{prefix}_lag_{lag}h": series.shift(lag) for lag in lags_hours}, index=series.index
    )


def add_rolling_features(
    series: pd.Series, windows_hours: list[int], prefix: str, min_lag_hours: int = 1
) -> pd.DataFrame:
    """Rolling mean/std of `series`, computed only on data available `min_lag_hours` before t."""
    shifted = series.shift(min_lag_hours)
    out = {}
    for window in windows_hours:
        out[f"{prefix}_rollmean_{window}h"] = shifted.rolling(window).mean()
        out[f"{prefix}_rollstd_{window}h"] = shifted.rolling(window).std()
    return pd.DataFrame(out, index=series.index)


def fill_untracked_generation_categories(actual_generation_by_type: pd.DataFrame) -> pd.DataFrame:
    """Fill the *leading* all-NaN stretch of each generation-by-type column with 0.

    ENTSO-E's set of reported production types for FR has grown over time (e.g. grid-scale
    battery storage and offshore wind only began reporting partway through 2019-2026): a
    column that is NaN for its first N rows then transitions to real values almost certainly
    represents "this technology had ~zero installed capacity / wasn't tracked yet", not a
    reporting gap for an active technology -- filling that leading stretch with 0 is the
    economically correct read, and it avoids it poisoning every one of those early rows via
    NaN-propagation once lagged (`lag_realized_series`) and later required to be complete by
    strict models (e.g. LEAR's `notna().all(axis=1)` mask).

    Only the strict leading stretch (before the column's first real value) is touched --
    scattered/interior NaN once a technology is active (a genuine reporting gap, and common
    enough in real ENTSO-E data even for well-established categories) are left as NaN.
    """
    filled = actual_generation_by_type.copy()
    for col in filled.columns:
        series = filled[col]
        first_valid = series.first_valid_index()
        if first_valid is None:
            continue
        leading = series.index < first_valid
        filled.loc[leading, col] = series.loc[leading].fillna(0.0)
    return filled


def lag_realized_series(df: pd.DataFrame, availability_lag_hours: int, prefix: str) -> pd.DataFrame:
    """Shift a realized (actual, not day-ahead-forecast) series to reflect what is knowable
    `availability_lag_hours` before each delivery hour."""
    shifted = df.shift(availability_lag_hours)
    shifted.columns = [f"{prefix}_{c}_lag{availability_lag_hours}h" for c in df.columns]
    return shifted


def expand_daily_to_hourly(daily_df: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Broadcast a daily-resolution series (e.g. weather) onto every hour of its local calendar date."""
    local_dates = pd.Index(hourly_index.tz_convert("Europe/Paris").normalize().date)
    daily_by_date = daily_df.copy()
    daily_by_date.index = daily_df.index.normalize()
    lookup = daily_by_date.set_axis(daily_by_date.index.date)
    expanded = lookup.reindex(local_dates)
    expanded.index = hourly_index
    return expanded


def build_feature_matrix(
    day_ahead_prices: pd.DataFrame,
    wind_solar_forecast: pd.DataFrame,
    load_forecast: pd.DataFrame,
    actual_generation_by_type: pd.DataFrame,
    config: FeaturesConfig,
    net_position: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    intraday: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the point-in-time feature matrix. See module docstring for the information timeline.

    The DA price index is treated as the authoritative hourly delivery grid; all other inputs
    are reindexed onto it (introducing NaNs where a source has gaps, handled downstream).

    `net_position` is optional: ENTSO-E's FR net cross-border position feed was found (via a
    live run) to have been discontinued around 2025-09-30 -- with it required, every model
    needing complete rows (LEAR, LSTM) would silently produce zero predictions for the entire
    tail of the dataset, the most recent and most relevant period. `configs/default.yaml`
    disables `entsoe.cross_border_flows` by default for this reason; the column is only added
    when `net_position` is supplied (e.g. for a deliberately pre-2025-10 backtest).
    """
    da_price = day_ahead_prices["day_ahead_price_eur_mwh"]
    grid = da_price.index

    wind_solar_forecast = wind_solar_forecast.reindex(grid)
    load_forecast = load_forecast.reindex(grid)
    actual_generation_by_type = fill_untracked_generation_categories(
        actual_generation_by_type.reindex(grid)
    )

    calendar = add_calendar_features(grid)
    renewables = compute_renewable_forecast_mw(wind_solar_forecast)
    load_series = load_forecast[load_forecast.columns[0]].rename("load_forecast_mw")
    residual_load = compute_residual_load_mw(load_series, renewables)

    price_lags = add_lag_features(da_price, config.price_lags_hours, "day_ahead_price")
    price_rolling = add_rolling_features(
        da_price, config.price_rolling_windows_hours, "day_ahead_price"
    )
    generation_lagged = lag_realized_series(
        actual_generation_by_type, config.generation_availability_lag_hours, "generation"
    )

    frames = [
        calendar,
        renewables,
        load_series,
        residual_load,
        price_lags,
        price_rolling,
        generation_lagged,
    ]

    if net_position is not None:
        frames.append(net_position.reindex(grid))
    else:
        logger.warning(
            "No net cross-border position data supplied; net_position_mw will be absent "
            "from the feature matrix."
        )

    if weather is not None:
        weather_hourly = expand_daily_to_hourly(weather, grid)
        weather_lagged = lag_realized_series(
            weather_hourly, config.weather_availability_lag_hours, "weather"
        )
        frames.append(weather_lagged)
    else:
        logger.warning(
            "No weather data supplied; HDD/CDD features will be absent from the feature matrix."
        )

    if intraday is not None and "intraday_price_eur_mwh" in intraday.columns:
        intraday_aligned = intraday.reindex(grid)
        spread = (intraday_aligned["intraday_price_eur_mwh"] - da_price).rename(
            "target_spread_eur_mwh"
        )
        spread_lags = add_lag_features(spread, config.spread_lags_hours, "spread")
        spread_rolling = add_rolling_features(spread, config.price_rolling_windows_hours, "spread")
        frames.extend([da_price, spread, spread_lags, spread_rolling])
        target_column = "target_spread_eur_mwh"
    else:
        logger.warning(
            "No intraday data available -- target degrades to forecasting the day-ahead price "
            "level (`target_day_ahead_price_eur_mwh`) instead of the intraday-vs-day-ahead spread."
        )
        frames.append(da_price.rename("target_day_ahead_price_eur_mwh"))
        target_column = "target_day_ahead_price_eur_mwh"

    matrix = pd.concat(frames, axis=1)
    matrix.attrs["target_column"] = target_column
    return matrix
