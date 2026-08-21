"""Population-weighted French temperature and HDD/CDD via Open-Meteo.

Open-Meteo's historical archive API (https://archive-api.open-meteo.com) is
free and requires no API key. We pull daily mean temperature for a small
basket of major French cities, combine them into a national
population-weighted average, and derive heating/cooling degree days --
French electricity demand is strongly thermosensitive (electric heating),
so this is expected to be a meaningful demand-side feature (see Phase 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fr_power.config import WeatherConfig
from fr_power.data.cache import cached_fetch

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_city_daily_mean_temperature(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "Europe/Paris",
    session: Any = None,
) -> pd.Series:
    """Fetch daily mean 2m temperature (Celsius) for one location from Open-Meteo."""
    session = session or requests
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": timezone,
    }
    response = session.get(ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    dates = pd.to_datetime(payload["daily"]["time"])
    temps = payload["daily"]["temperature_2m_mean"]
    return pd.Series(temps, index=dates, name="temperature_c")


def fetch_population_weighted_temperature(
    config: WeatherConfig,
    start_date: str,
    end_date: str,
    timezone: str = "Europe/Paris",
    session: Any = None,
) -> pd.DataFrame:
    """Build a national population-weighted temperature series and HDD/CDD from it."""
    city_temps = {
        city.name: fetch_city_daily_mean_temperature(
            city.lat, city.lon, start_date, end_date, timezone=timezone, session=session
        )
        for city in config.cities
    }
    temps_df = pd.DataFrame(city_temps)
    weights = pd.Series({city.name: city.weight for city in config.cities})
    weighted_temp = (temps_df * weights).sum(axis=1) / weights.sum()

    result = pd.DataFrame({"temperature_c": weighted_temp})
    result["hdd"] = (config.hdd_base_celsius - result["temperature_c"]).clip(lower=0)
    result["cdd"] = (result["temperature_c"] - config.cdd_base_celsius).clip(lower=0)
    return result


def fetch_and_cache_weather(
    config: WeatherConfig,
    cache_dir: str | Path,
    start_date: str,
    end_date: str,
    timezone: str = "Europe/Paris",
    force_refresh: bool = False,
    session: Any = None,
) -> pd.DataFrame | None:
    """Fetch (or load from cache) the population-weighted FR temperature/HDD/CDD series.

    Returns None if weather enrichment is disabled in config.
    """
    if not config.enabled:
        logger.info("Weather enrichment disabled in config; skipping.")
        return None
    return cached_fetch(
        cache_dir,
        "weather_fr_population_weighted",
        lambda: fetch_population_weighted_temperature(
            config, start_date, end_date, timezone=timezone, session=session
        ),
        force_refresh=force_refresh,
    )
