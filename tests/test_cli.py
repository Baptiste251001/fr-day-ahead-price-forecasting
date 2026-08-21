"""Smoke tests for the CLI skeleton."""

import pandas as pd
import pytest
from typer.testing import CliRunner

from fr_power.cli import app
from fr_power.data import entsoe, weather

runner = CliRunner()


def test_cli_help_lists_all_stage_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["fetch", "features", "train", "calibrate", "backtest", "report"]:
        assert command in result.stdout


def test_fetch_without_api_token_exits_with_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENTSOE_API_TOKEN", raising=False)
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 1
    assert "ENTSOE_API_TOKEN" in result.stdout


def test_fetch_end_to_end_with_mocked_data_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises the full `fetch` orchestration wiring with all external calls mocked."""
    monkeypatch.setattr(entsoe, "build_client", lambda secrets: object())
    monkeypatch.setattr(
        entsoe,
        "fetch_all",
        lambda *args, **kwargs: {
            "day_ahead_prices": pd.DataFrame(
                {"day_ahead_price_eur_mwh": [42.0]},
                index=pd.date_range("2024-01-01", periods=1, freq="h", tz="Europe/Paris"),
            )
        },
    )
    monkeypatch.setattr(weather, "fetch_and_cache_weather", lambda *args, **kwargs: None)

    result = runner.invoke(app, ["fetch"])

    assert result.exit_code == 0
    assert "entsoe/day_ahead_prices: 1 rows" in result.stdout
    assert "intraday: not available" in result.stdout


def test_stub_commands_run_without_error() -> None:
    for command in ["features", "train", "calibrate", "backtest", "report"]:
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        assert "stub" in result.stdout
