"""Smoke tests for the CLI skeleton."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from fr_power.cli import app
from fr_power.config import DEFAULT_CONFIG_PATH
from fr_power.data import cache, entsoe, weather

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


def test_features_command_builds_matrix_from_cached_data(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    index = pd.date_range("2024-06-01", periods=72, freq="h", tz="Europe/Paris")

    cache.write_cache(
        pd.DataFrame({"day_ahead_price_eur_mwh": np.arange(len(index), dtype=float)}, index=index),
        cache.cache_path(raw_dir, "entsoe_day_ahead_prices"),
    )
    cache.write_cache(
        pd.DataFrame(
            {"Solar": np.full(len(index), 10.0), "Wind Onshore": np.full(len(index), 20.0)},
            index=index,
        ),
        cache.cache_path(raw_dir, "entsoe_day_ahead_wind_solar_forecast"),
    )
    cache.write_cache(
        pd.DataFrame({"Forecasted Load": np.full(len(index), 1000.0)}, index=index),
        cache.cache_path(raw_dir, "entsoe_day_ahead_load_forecast"),
    )
    cache.write_cache(
        pd.DataFrame({"Nuclear": np.full(len(index), 40000.0)}, index=index),
        cache.cache_path(raw_dir, "entsoe_actual_generation_by_type"),
    )
    cache.write_cache(
        pd.DataFrame({"net_position_mw": np.zeros(len(index))}, index=index),
        cache.cache_path(raw_dir, "entsoe_net_cross_border_position"),
    )

    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["raw_dir"] = str(raw_dir)
    config_dict["data"]["processed_dir"] = str(processed_dir)
    config_dict["data"]["intraday"]["source"] = "stub"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["features", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "target column: target_day_ahead_price_eur_mwh" in result.stdout
    assert (processed_dir / "feature_matrix.parquet").exists()
    assert (processed_dir / "feature_matrix.meta.json").exists()


def test_features_command_errors_when_fetch_not_run_yet(tmp_path: Path) -> None:
    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["raw_dir"] = str(tmp_path / "empty_raw")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["features", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "fr-power fetch" in result.stdout


def test_stub_commands_run_without_error() -> None:
    for command in ["train", "calibrate", "backtest", "report"]:
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        assert "stub" in result.stdout
