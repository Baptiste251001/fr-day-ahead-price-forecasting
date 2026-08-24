"""Tests for ENTSO-E data ingestion. All network access is mocked -- these
tests must never hit the live ENTSO-E API (see CI config).
"""

from pathlib import Path

import pandas as pd
import pytest
from entsoe.exceptions import NoMatchingDataError

from fr_power.config import EntsoeSeriesConfig, Secrets
from fr_power.data import entsoe


def _index(n: int = 3) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")


class FakeClient:
    """Duck-typed stand-in for entsoe.EntsoePandasClient."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def query_day_ahead_prices(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        self.calls.append("query_day_ahead_prices")
        return pd.Series([50.0, 60.0, 70.0], index=_index(), name="Day-ahead Price")

    def query_load_forecast(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        self.calls.append("query_load_forecast")
        return pd.DataFrame({"Forecasted Load": [45000.0, 46000.0, 47000.0]}, index=_index())

    def query_wind_and_solar_forecast(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp, process_type: str
    ) -> pd.DataFrame:
        self.calls.append("query_wind_and_solar_forecast")
        return pd.DataFrame(
            {"Solar": [0.0, 100.0, 500.0], "Wind Onshore": [2000.0, 2100.0, 1800.0]}, index=_index()
        )

    def query_generation(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp, nett: bool
    ) -> pd.DataFrame:
        self.calls.append("query_generation")
        assert nett is True
        return pd.DataFrame({"Nuclear": [40000.0, 41000.0, 39000.0]}, index=_index())

    def query_net_position(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp, dayahead: bool
    ) -> pd.Series:
        self.calls.append("query_net_position")
        assert dayahead is True
        return pd.Series([-1000.0, -500.0, 200.0], index=_index(), name="Net Position")


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


def test_build_client_raises_without_token() -> None:
    secrets = Secrets(entsoe_api_token=None)
    with pytest.raises(RuntimeError, match="ENTSOE_API_TOKEN"):
        entsoe.build_client(secrets)


def test_fetch_day_ahead_prices_renames_column(fake_client: FakeClient) -> None:
    df = entsoe.fetch_day_ahead_prices(fake_client, _index()[0], _index()[-1])
    assert list(df.columns) == ["day_ahead_price_eur_mwh"]
    assert df["day_ahead_price_eur_mwh"].tolist() == [50.0, 60.0, 70.0]


def test_fetch_net_cross_border_position_renames_column(fake_client: FakeClient) -> None:
    df = entsoe.fetch_net_cross_border_position(fake_client, _index()[0], _index()[-1])
    assert list(df.columns) == ["net_position_mw"]


def test_fetch_all_respects_series_flags_and_caches(
    fake_client: FakeClient, tmp_path: Path
) -> None:
    flags = EntsoeSeriesConfig(
        day_ahead_prices=True,
        day_ahead_wind_solar_forecast=True,
        day_ahead_load_forecast=False,
        actual_generation_by_type=True,
        cross_border_flows=False,
    )
    results = entsoe.fetch_all(fake_client, _index()[0], _index()[-1], tmp_path, flags)

    assert set(results.keys()) == {
        "day_ahead_prices",
        "day_ahead_wind_solar_forecast",
        "actual_generation_by_type",
    }
    assert "query_load_forecast" not in fake_client.calls
    assert "query_net_position" not in fake_client.calls
    assert (tmp_path / "entsoe_day_ahead_prices.parquet").exists()


def test_fetch_all_uses_cache_on_second_call(fake_client: FakeClient, tmp_path: Path) -> None:
    flags = EntsoeSeriesConfig(
        day_ahead_prices=True,
        day_ahead_wind_solar_forecast=False,
        day_ahead_load_forecast=False,
        actual_generation_by_type=False,
        cross_border_flows=False,
    )
    entsoe.fetch_all(fake_client, _index()[0], _index()[-1], tmp_path, flags)
    entsoe.fetch_all(fake_client, _index()[0], _index()[-1], tmp_path, flags)

    assert fake_client.calls.count("query_day_ahead_prices") == 1


class _PartiallyUnavailableClient(FakeClient):
    """Like FakeClient, but net position is not published for this bidding zone/period --
    the real-world case that surfaced this (FR does not always publish it)."""

    def query_net_position(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp, dayahead: bool
    ) -> pd.Series:
        self.calls.append("query_net_position")
        raise NoMatchingDataError


def test_fetch_all_skips_a_series_that_has_no_matching_data(tmp_path: Path) -> None:
    client = _PartiallyUnavailableClient()
    flags = EntsoeSeriesConfig(
        day_ahead_prices=True,
        day_ahead_wind_solar_forecast=False,
        day_ahead_load_forecast=False,
        actual_generation_by_type=False,
        cross_border_flows=True,
    )

    results = entsoe.fetch_all(client, _index()[0], _index()[-1], tmp_path, flags)

    assert set(results.keys()) == {"day_ahead_prices"}
    assert "cross_border_flows" not in results
    assert not (tmp_path / "entsoe_net_cross_border_position.parquet").exists()
