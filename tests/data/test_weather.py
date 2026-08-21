"""Tests for the weather (Open-Meteo) ingestion. No live network access."""

from pathlib import Path
from typing import Any

from fr_power.config import WeatherCity, WeatherConfig
from fr_power.data.weather import (
    fetch_and_cache_weather,
    fetch_city_daily_mean_temperature,
    fetch_population_weighted_temperature,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    """Maps a (lat, lon) pair to a canned Open-Meteo daily-temperature payload."""

    def __init__(self, payloads_by_city: dict[str, dict[str, Any]]) -> None:
        self.payloads_by_city = payloads_by_city
        self.requested_urls: list[str] = []

    def get(self, url: str, params: dict[str, Any], timeout: int) -> FakeResponse:
        self.requested_urls.append(url)
        key = f"{params['latitude']},{params['longitude']}"
        return FakeResponse(self.payloads_by_city[key])


def _daily_payload(temps: list[float]) -> dict[str, Any]:
    dates = [f"2024-01-{i + 1:02d}" for i in range(len(temps))]
    return {"daily": {"time": dates, "temperature_2m_mean": temps}}


def test_fetch_city_daily_mean_temperature_parses_payload() -> None:
    session = FakeSession({"48.85,2.35": _daily_payload([5.0, 6.0, 7.0])})

    series = fetch_city_daily_mean_temperature(
        48.85, 2.35, "2024-01-01", "2024-01-03", session=session
    )

    assert series.tolist() == [5.0, 6.0, 7.0]
    assert len(series) == 3
    assert session.requested_urls == ["https://archive-api.open-meteo.com/v1/archive"]


def test_population_weighted_temperature_and_hdd_cdd() -> None:
    config = WeatherConfig(
        enabled=True,
        provider="open-meteo",
        cities=[
            WeatherCity(name="A", lat=1.0, lon=1.0, weight=0.5),
            WeatherCity(name="B", lat=2.0, lon=2.0, weight=0.5),
        ],
        hdd_base_celsius=15.5,
        cdd_base_celsius=18.0,
    )
    session = FakeSession(
        {
            "1.0,1.0": _daily_payload([0.0, 20.0]),
            "2.0,2.0": _daily_payload([10.0, 22.0]),
        }
    )

    result = fetch_population_weighted_temperature(
        config, "2024-01-01", "2024-01-02", session=session
    )

    # day 1: weighted temp = 0.5*0 + 0.5*10 = 5.0 -> hdd = 10.5, cdd = 0
    # day 2: weighted temp = 0.5*20 + 0.5*22 = 21.0 -> hdd = 0, cdd = 3.0
    assert result["temperature_c"].tolist() == [5.0, 21.0]
    assert result["hdd"].tolist() == [10.5, 0.0]
    assert result["cdd"].tolist() == [0.0, 3.0]


def test_fetch_and_cache_weather_skips_when_disabled(tmp_path: Path) -> None:
    config = WeatherConfig(
        enabled=False,
        provider="open-meteo",
        cities=[WeatherCity(name="A", lat=1.0, lon=1.0, weight=1.0)],
        hdd_base_celsius=15.5,
        cdd_base_celsius=18.0,
    )
    result = fetch_and_cache_weather(config, tmp_path, "2024-01-01", "2024-01-02")
    assert result is None


def test_fetch_and_cache_weather_caches_to_parquet(tmp_path: Path) -> None:
    config = WeatherConfig(
        enabled=True,
        provider="open-meteo",
        cities=[WeatherCity(name="A", lat=1.0, lon=1.0, weight=1.0)],
        hdd_base_celsius=15.5,
        cdd_base_celsius=18.0,
    )
    session = FakeSession({"1.0,1.0": _daily_payload([5.0, 6.0])})

    result = fetch_and_cache_weather(config, tmp_path, "2024-01-01", "2024-01-02", session=session)

    assert result is not None
    assert (tmp_path / "weather_fr_population_weighted.parquet").exists()
