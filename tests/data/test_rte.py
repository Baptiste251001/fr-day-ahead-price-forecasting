"""Tests for the optional RTE eCO2mix ingestion. No live network access."""

from pathlib import Path
from typing import Any

import pandas as pd

from fr_power.config import RteEco2mixConfig, Secrets
from fr_power.data.rte import (
    fetch_access_token,
    fetch_and_cache_eco2mix,
    fetch_eco2mix_national_real_time,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, headers: dict[str, Any], timeout: int) -> FakeResponse:
        self.posts.append((url, headers))
        return FakeResponse({"access_token": "fake-token-123"})

    def get(
        self, url: str, headers: dict[str, Any], params: dict[str, Any], timeout: int
    ) -> FakeResponse:
        self.gets.append((url, headers))
        return FakeResponse(
            {
                "national_real_time": [
                    {"date_heure": "2024-01-01T00:00:00+01:00", "nucleaire": 40000, "eolien": 2000},
                    {"date_heure": "2024-01-01T01:00:00+01:00", "nucleaire": 41000, "eolien": 2100},
                ]
            }
        )


def test_fetch_access_token_uses_basic_auth() -> None:
    session = FakeSession()
    token = fetch_access_token("client-id", "client-secret", session=session)

    assert token == "fake-token-123"
    assert len(session.posts) == 1
    assert session.posts[0][1]["Authorization"].startswith("Basic ")


def test_fetch_eco2mix_national_real_time_parses_records() -> None:
    session = FakeSession()
    df = fetch_eco2mix_national_real_time(
        "fake-token", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), session=session
    )

    assert len(df) == 2
    assert "nucleaire" in df.columns
    assert session.gets[0][1]["Authorization"] == "Bearer fake-token"


def test_fetch_and_cache_eco2mix_skips_when_disabled(tmp_path: Path) -> None:
    config = RteEco2mixConfig(enabled=False)
    secrets = Secrets(rte_client_id="id", rte_client_secret="secret")

    result = fetch_and_cache_eco2mix(
        config, secrets, tmp_path, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    )
    assert result is None


def test_fetch_and_cache_eco2mix_skips_when_credentials_missing(tmp_path: Path) -> None:
    config = RteEco2mixConfig(enabled=True)
    secrets = Secrets(rte_client_id=None, rte_client_secret=None)

    result = fetch_and_cache_eco2mix(
        config, secrets, tmp_path, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    )
    assert result is None


def test_fetch_and_cache_eco2mix_caches_to_parquet(tmp_path: Path) -> None:
    config = RteEco2mixConfig(enabled=True)
    secrets = Secrets(rte_client_id="id", rte_client_secret="secret")
    session = FakeSession()

    result = fetch_and_cache_eco2mix(
        config,
        secrets,
        tmp_path,
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        session=session,
    )

    assert result is not None
    assert (tmp_path / "rte_eco2mix_national_real_time.parquet").exists()
