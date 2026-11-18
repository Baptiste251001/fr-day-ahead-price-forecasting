"""Raw data ingestion: ENTSO-E Transparency Platform + Open-Meteo, cached as parquet.

Every pull goes through `cached_fetch`, so re-running the pipeline never re-hits an API
unnecessarily and tests never need network access.

Series pulled for the FR bidding zone:
  * day-ahead auction price (the forecasting target): hourly products until 30 September
    2025, 15-minute products from 1 October 2025 (SDAC 15-minute market time unit),
  * day-ahead load forecast and day-ahead wind & solar forecast (known before the auction),
  * actual generation by production type (realized, so only used with a lag),
  * daily population-weighted temperature for a basket of French cities (Open-Meteo),
  * daily closes of the Dutch TTF natural-gas front-month future (Yahoo Finance).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from fr_power.config import WeatherConfig

logger = logging.getLogger(__name__)

COUNTRY_CODE = "FR"
TIMEZONE = "Europe/Paris"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
# First delivery day of 15-minute products on the SDAC day-ahead auction.
MTU_15MIN_START = pd.Timestamp("2025-10-01", tz=TIMEZONE)

# Cache file name (without .parquet) for each raw dataset.
DA_PRICES = "entsoe_day_ahead_prices"
LOAD_FORECAST = "entsoe_day_ahead_load_forecast"
WIND_SOLAR_FORECAST = "entsoe_day_ahead_wind_solar_forecast"
GENERATION = "entsoe_actual_generation_by_type"
WEATHER = "weather_fr_population_weighted"
GAS = "gas_ttf_daily"
RAW_DATASETS = [
    DA_PRICES,
    LOAD_FORECAST,
    WIND_SOLAR_FORECAST,
    GENERATION,
    WEATHER,
    GAS,
]


# --------------------------------------------------------------------------- cache


def cache_path(cache_dir: str | Path, name: str) -> Path:
    return Path(cache_dir) / f"{name}.parquet"


def read_cache(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    return pd.read_parquet(path) if path.exists() else None


def write_cache(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Cached %d rows to %s", len(df), path)


def cached_fetch(
    cache_dir: str | Path,
    name: str,
    fetch_fn: Callable[[], pd.DataFrame],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return dataset `name` from the cache, fetching and caching it first if absent."""
    path = cache_path(cache_dir, name)
    if not force_refresh and (cached := read_cache(path)) is not None:
        return cached
    df = fetch_fn()
    write_cache(df, path)
    return df


def load_raw(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every cached raw dataset; raise if any is missing (run `fr-power fetch`)."""
    raw = {name: read_cache(cache_path(raw_dir, name)) for name in RAW_DATASETS}
    missing = [name for name, df in raw.items() if df is None]
    if missing:
        raise FileNotFoundError(f"Missing raw data {missing} in {raw_dir}: run `fr-power fetch`.")
    return {name: df for name, df in raw.items() if df is not None}


# --------------------------------------------------------------------------- ENTSO-E


def entsoe_client() -> Any:
    """Authenticated `EntsoePandasClient`, using ENTSOE_API_TOKEN from the env or `.env`."""
    from entsoe import EntsoePandasClient

    load_dotenv()
    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "ENTSOE_API_TOKEN is not set. Copy .env.example to .env and add your ENTSO-E "
            "Transparency Platform token (see README)."
        )
    return EntsoePandasClient(api_key=token)


def fetch_day_ahead_prices(client: Any, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Day-ahead prices at the resolution of the traded products.

    Hourly before `MTU_15MIN_START`, 15-minute from then on. The resolution must be asked
    for explicitly: after the switch, entsoe-py's default hourly query returns the price of
    each hour's first quarter, not the hourly average.
    """
    parts = []
    if start < MTU_15MIN_START:
        end_hourly = min(end, MTU_15MIN_START)
        hourly = client.query_day_ahead_prices(
            COUNTRY_CODE, start=start, end=end_hourly, resolution="60min"
        )
        parts.append(hourly[hourly.index < MTU_15MIN_START])
    if end > MTU_15MIN_START:
        parts.append(
            client.query_day_ahead_prices(
                COUNTRY_CODE, start=max(start, MTU_15MIN_START), end=end, resolution="15min"
            )
        )
    prices = pd.concat(parts)
    return prices[~prices.index.duplicated()].rename("day_ahead_price_eur_mwh").to_frame()


def fetch_entsoe(
    client: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch (or load from cache) the four ENTSO-E series used by the model."""
    queries: dict[str, Callable[[], pd.DataFrame]] = {
        DA_PRICES: lambda: fetch_day_ahead_prices(client, start, end),
        LOAD_FORECAST: lambda: client.query_load_forecast(COUNTRY_CODE, start=start, end=end),
        WIND_SOLAR_FORECAST: lambda: client.query_wind_and_solar_forecast(
            COUNTRY_CODE, start=start, end=end, process_type="A01"
        ),
        GENERATION: lambda: client.query_generation(COUNTRY_CODE, start=start, end=end, nett=True),
    }
    results = {}
    for name, query in queries.items():
        logger.info("Fetching %s", name)
        results[name] = cached_fetch(cache_dir, name, query, force_refresh=force_refresh)
    return results


# --------------------------------------------------------------------------- weather


def fetch_city_temperature(
    lat: float, lon: float, start_date: str, end_date: str, session: Any = None
) -> pd.Series:
    """Daily mean 2m temperature (Celsius) for one location, from Open-Meteo's archive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": TIMEZONE,
    }
    response = (session or requests).get(OPEN_METEO_URL, params=params, timeout=60)
    response.raise_for_status()
    daily = response.json()["daily"]
    return pd.Series(daily["temperature_2m_mean"], index=pd.to_datetime(daily["time"]))


def fetch_weather(
    config: WeatherConfig, start_date: str, end_date: str, session: Any = None
) -> pd.DataFrame:
    """Population-weighted national temperature, plus heating/cooling degree days."""
    temps = pd.DataFrame(
        {
            c.name: fetch_city_temperature(c.lat, c.lon, start_date, end_date, session)
            for c in config.cities
        }
    )
    weights = pd.Series({c.name: c.weight for c in config.cities})
    temperature = (temps * weights).sum(axis=1) / weights.sum()
    return pd.DataFrame(
        {
            "temperature_c": temperature,
            "hdd": (config.hdd_base_celsius - temperature).clip(lower=0),
            "cdd": (temperature - config.cdd_base_celsius).clip(lower=0),
        }
    )


# --------------------------------------------------------------------------- gas


def fetch_gas(ticker: str, start_date: str, end_date: str, session: Any = None) -> pd.DataFrame:
    """Daily closing price (EUR/MWh) of a gas future from Yahoo Finance, indexed by date."""
    params = {
        "period1": int(pd.Timestamp(start_date, tz="UTC").timestamp()),
        "period2": int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp()),
        "interval": "1d",
    }
    response = (session or requests).get(
        YAHOO_CHART_URL.format(ticker=ticker),
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s").normalize()
    close = pd.Series(result["indicators"]["quote"][0]["close"], index=dates, dtype=float)
    close = close[~close.index.duplicated(keep="last")].dropna()
    return close.rename("gas_close_eur_mwh").to_frame()
