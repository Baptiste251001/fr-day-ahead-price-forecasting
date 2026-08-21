"""ENTSO-E Transparency Platform data ingestion for the FR bidding zone.

Pulls, for France:
  - day-ahead auction prices,
  - day-ahead wind & solar generation forecast,
  - day-ahead load (demand) forecast,
  - actual generation by production type (nuclear/hydro/wind/solar/thermal shares),
  - the net cross-border position (FR's net scheduled import/export).

Day-ahead forecasts are pulled deliberately (not realized values) because
they are what is actually known at the ~13:00 D-1 decision point that this
project's information timeline is built around; see Phase 2 for how these
feed the point-in-time feature matrix.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from entsoe import EntsoePandasClient

from fr_power.config import EntsoeSeriesConfig, Secrets
from fr_power.data.cache import cached_fetch

logger = logging.getLogger(__name__)

COUNTRY_CODE = "FR"


def build_client(secrets: Secrets) -> EntsoePandasClient:
    """Construct an authenticated ENTSO-E client from loaded secrets."""
    if not secrets.entsoe_api_token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN is not set. Copy .env.example to .env and add your "
            "ENTSO-E Transparency Platform token (see .env.example for how to "
            "request one)."
        )
    return EntsoePandasClient(api_key=secrets.entsoe_api_token)


def fetch_day_ahead_prices(client: Any, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """FR day-ahead auction clearing price, EUR/MWh, hourly."""
    series = client.query_day_ahead_prices(COUNTRY_CODE, start=start, end=end)
    return series.rename("day_ahead_price_eur_mwh").to_frame()


def fetch_day_ahead_load_forecast(
    client: Any, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """FR day-ahead total load forecast, MW."""
    df: pd.DataFrame = client.query_load_forecast(COUNTRY_CODE, start=start, end=end)
    return df


def fetch_day_ahead_wind_solar_forecast(
    client: Any, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """FR day-ahead wind & solar generation forecast, MW, by production type."""
    df: pd.DataFrame = client.query_wind_and_solar_forecast(
        COUNTRY_CODE, start=start, end=end, process_type="A01"
    )
    return df


def fetch_actual_generation_by_type(
    client: Any, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """FR actual generation, MW, net of consumption, by production type (nuclear/hydro/etc.)."""
    df: pd.DataFrame = client.query_generation(COUNTRY_CODE, start=start, end=end, nett=True)
    return df


def fetch_net_cross_border_position(
    client: Any, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """FR day-ahead net position, MW (positive = net import, negative = net export)."""
    series = client.query_net_position(COUNTRY_CODE, start=start, end=end, dayahead=True)
    return series.rename("net_position_mw").to_frame()


# (config flag name, cache file name, fetch function)
_SERIES: list[tuple[str, str, Callable[[Any, pd.Timestamp, pd.Timestamp], pd.DataFrame]]] = [
    ("day_ahead_prices", "entsoe_day_ahead_prices", fetch_day_ahead_prices),
    (
        "day_ahead_wind_solar_forecast",
        "entsoe_day_ahead_wind_solar_forecast",
        fetch_day_ahead_wind_solar_forecast,
    ),
    ("day_ahead_load_forecast", "entsoe_day_ahead_load_forecast", fetch_day_ahead_load_forecast),
    (
        "actual_generation_by_type",
        "entsoe_actual_generation_by_type",
        fetch_actual_generation_by_type,
    ),
    ("cross_border_flows", "entsoe_net_cross_border_position", fetch_net_cross_border_position),
]


def fetch_all(
    client: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path,
    series_flags: EntsoeSeriesConfig,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch (or load from cache) every ENTSO-E series enabled in `series_flags`."""
    results: dict[str, pd.DataFrame] = {}
    for flag_name, cache_name, fetch_fn in _SERIES:
        if not getattr(series_flags, flag_name):
            logger.info("Skipping ENTSO-E series '%s' (disabled in config).", flag_name)
            continue
        logger.info("Fetching ENTSO-E series '%s'.", flag_name)
        results[flag_name] = cached_fetch(
            cache_dir,
            cache_name,
            functools.partial(fetch_fn, client, start, end),
            force_refresh=force_refresh,
        )
    return results
