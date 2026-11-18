"""Data ingestion and caching. Network access is always mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fr_power import data
from fr_power.config import City, WeatherConfig


class FakeEntsoeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.index = pd.date_range("2024-01-01", periods=3, freq="h", tz=data.TIMEZONE)

    def query_day_ahead_prices(self, country: str, start: Any, end: Any) -> pd.Series:
        self.calls.append("prices")
        return pd.Series([50.0, 60.0, 70.0], index=self.index)

    def query_load_forecast(self, country: str, start: Any, end: Any) -> pd.DataFrame:
        self.calls.append("load")
        return pd.DataFrame({"Forecasted Load": [1.0, 2.0, 3.0]}, index=self.index)

    def query_wind_and_solar_forecast(
        self, country: str, start: Any, end: Any, process_type: str
    ) -> pd.DataFrame:
        self.calls.append("wind_solar")
        return pd.DataFrame({"Solar": [0.0, 1.0, 2.0]}, index=self.index)

    def query_generation(self, country: str, start: Any, end: Any, nett: bool) -> pd.DataFrame:
        self.calls.append("generation")
        return pd.DataFrame({"Nuclear": [4.0, 5.0, 6.0]}, index=self.index)


def test_fetch_entsoe_caches_and_reuses(tmp_path: Path) -> None:
    client = FakeEntsoeClient()
    start, end = client.index[0], client.index[-1]

    first = data.fetch_entsoe(client, start, end, tmp_path)
    data.fetch_entsoe(client, start, end, tmp_path)

    assert set(first) == {
        data.DA_PRICES,
        data.LOAD_FORECAST,
        data.WIND_SOLAR_FORECAST,
        data.GENERATION,
    }
    assert first[data.DA_PRICES]["day_ahead_price_eur_mwh"].tolist() == [50.0, 60.0, 70.0]
    assert client.calls.count("prices") == 1  # second call served from cache


def test_force_refresh_bypasses_cache(tmp_path: Path) -> None:
    client = FakeEntsoeClient()
    data.fetch_entsoe(client, client.index[0], client.index[-1], tmp_path)
    data.fetch_entsoe(client, client.index[0], client.index[-1], tmp_path, force_refresh=True)
    assert client.calls.count("prices") == 2


def test_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTSOE_API_TOKEN", "")
    monkeypatch.setattr(data, "load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="ENTSOE_API_TOKEN"):
        data.entsoe_client()


def test_load_raw_reports_missing_datasets(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fr-power fetch"):
        data.load_raw(tmp_path)


class FakeSession:
    def __init__(self, temps_by_lat: dict[float, list[float]]) -> None:
        self.temps_by_lat = temps_by_lat

    def get(self, url: str, params: dict[str, Any], timeout: int) -> Any:
        temps = self.temps_by_lat[params["latitude"]]
        payload = {
            "daily": {
                "time": [f"2024-01-0{i + 1}" for i in range(len(temps))],
                "temperature_2m_mean": temps,
            }
        }
        return type(
            "Response", (), {"raise_for_status": lambda self: None, "json": lambda self: payload}
        )()


def test_weather_is_population_weighted_with_degree_days() -> None:
    config = WeatherConfig(
        cities=[
            City(name="A", lat=1.0, lon=0, weight=0.5),
            City(name="B", lat=2.0, lon=0, weight=0.5),
        ],
        hdd_base_celsius=15.5,
        cdd_base_celsius=18.0,
    )
    session = FakeSession({1.0: [0.0, 20.0], 2.0: [10.0, 22.0]})

    weather = data.fetch_weather(config, "2024-01-01", "2024-01-02", session=session)

    assert weather["temperature_c"].tolist() == [5.0, 21.0]
    assert weather["hdd"].tolist() == [10.5, 0.0]
    assert weather["cdd"].tolist() == [0.0, 3.0]
