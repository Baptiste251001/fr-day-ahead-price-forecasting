"""Smoke tests for the CLI skeleton."""

import json
import os
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


def test_train_command_runs_full_forecast_suite(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    n = 300
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    rng = np.random.default_rng(0)
    target = 50.0 + 5.0 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(scale=1.0, size=n)
    matrix = pd.DataFrame(
        {
            "target_day_ahead_price_eur_mwh": target,
            "day_ahead_price_lag_24h": pd.Series(target, index=index).shift(24),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        },
        index=index,
    )
    cache.write_cache(matrix, processed_dir / "feature_matrix.parquet")
    (processed_dir / "feature_matrix.meta.json").write_text(
        json.dumps({"target_column": "target_day_ahead_price_eur_mwh"})
    )

    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(processed_dir)
    config_dict["models"]["naive"]["lag_hours"] = 24
    config_dict["models"]["lear"]["lasso_alpha_grid"] = [0.01, 0.1, 1.0]
    config_dict["models"]["lear"]["cv_folds"] = 2
    config_dict["models"]["lightgbm"]["n_estimators"] = 20
    config_dict["models"]["lightgbm"]["num_leaves"] = 7
    config_dict["models"]["lightgbm"]["min_child_samples"] = 2
    config_dict["models"]["lightgbm"]["early_stopping_rounds"] = 3
    config_dict["models"]["lstm"]["sequence_length_hours"] = 4
    config_dict["models"]["lstm"]["hidden_size"] = 4
    config_dict["models"]["lstm"]["num_layers"] = 1
    config_dict["models"]["lstm"]["batch_size"] = 8
    config_dict["models"]["lstm"]["max_epochs"] = 2
    config_dict["models"]["lstm"]["early_stopping_patience"] = 1
    config_dict["models"]["lstm"]["quantiles"] = [0.1, 0.5, 0.9]
    config_dict["backtest"]["walk_forward"]["initial_train_days"] = 3
    config_dict["backtest"]["walk_forward"]["retrain_every_days"] = 2
    config_dict["backtest"]["walk_forward"]["test_window_days"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    # cwd-relative "reports/" output -- isolate the test from the real repo's reports/ dir.
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["train", "--config", str(config_path)])
    finally:
        os.chdir(original_cwd)

    assert result.exit_code == 0
    for model_name in ["naive", "lear", "lightgbm", "lstm"]:
        assert model_name in result.stdout
    assert (tmp_path / "reports" / "forecast_metrics.csv").exists()
    assert (tmp_path / "reports" / "dm_tests.csv").exists()
    assert (tmp_path / "reports" / "figures" / "forecast_rmse.png").exists()


def test_train_command_errors_when_features_not_run_yet(tmp_path: Path) -> None:
    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(tmp_path / "empty_processed")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["train", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "fr-power features" in result.stdout


def _mean_reverting_series(n: int, kappa: float, mu: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.exp(-kappa)
    std_eps = sigma * np.sqrt((1 - a**2) / (2 * kappa))
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + a * (x[t - 1] - mu) + std_eps * rng.standard_normal()
    return x


def test_calibrate_command_runs_ou_calibration(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    n = 500
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    spread = _mean_reverting_series(n, kappa=0.1, mu=2.0, sigma=1.5, seed=0)
    matrix = pd.DataFrame({"target_spread_eur_mwh": spread}, index=index)
    cache.write_cache(matrix, processed_dir / "feature_matrix.parquet")
    (processed_dir / "feature_matrix.meta.json").write_text(
        json.dumps({"target_column": "target_spread_eur_mwh"})
    )

    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(processed_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["calibrate", "--config", str(config_path)])
    finally:
        os.chdir(original_cwd)

    assert result.exit_code == 0
    assert "half-life" in result.stdout
    assert (tmp_path / "reports" / "ou_calibration.json").exists()
    assert (processed_dir / "ou_positions.parquet").exists()

    summary = json.loads((tmp_path / "reports" / "ou_calibration.json").read_text())
    assert summary["mu"] == pytest.approx(2.0, abs=0.5)


def test_calibrate_command_skips_when_target_is_price_level(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    index = pd.date_range("2024-01-01", periods=50, freq="h", tz="Europe/Paris")
    matrix = pd.DataFrame({"target_day_ahead_price_eur_mwh": np.arange(50.0)}, index=index)
    cache.write_cache(matrix, processed_dir / "feature_matrix.parquet")
    (processed_dir / "feature_matrix.meta.json").write_text(
        json.dumps({"target_column": "target_day_ahead_price_eur_mwh"})
    )

    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(processed_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["calibrate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "skipped" in result.stdout


def test_calibrate_command_errors_when_features_not_run_yet(tmp_path: Path) -> None:
    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(tmp_path / "empty_processed")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["calibrate", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "fr-power features" in result.stdout


def test_backtest_command_runs_combined_strategy(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    n = 500
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    spread = pd.Series(_mean_reverting_series(n, kappa=0.1, mu=2.0, sigma=1.5, seed=0), index=index)
    spread_lag_24h = spread.shift(24)

    matrix = pd.DataFrame(
        {"target_spread_eur_mwh": spread, "spread_lag_24h": spread_lag_24h}, index=index
    )
    cache.write_cache(matrix, processed_dir / "feature_matrix.parquet")
    (processed_dir / "feature_matrix.meta.json").write_text(
        json.dumps({"target_column": "target_spread_eur_mwh"})
    )

    rng = np.random.default_rng(1)
    predictions = pd.DataFrame(
        {"naive": spread_lag_24h, "lightgbm": spread + rng.normal(scale=0.5, size=n)}, index=index
    )
    cache.write_cache(predictions, processed_dir / "forecast_predictions.parquet")

    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(processed_dir)
    config_dict["backtest"]["walk_forward"]["initial_train_days"] = 10
    config_dict["backtest"]["walk_forward"]["retrain_every_days"] = 5
    config_dict["backtest"]["walk_forward"]["test_window_days"] = 5
    config_dict["backtest"]["block_bootstrap"]["n_resamples"] = 50
    config_dict["backtest"]["cost_sensitivity_multipliers"] = [0.0, 1.0]
    config_dict["backtest"]["signal_model"] = "lightgbm"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["backtest", "--config", str(config_path)])
    finally:
        os.chdir(original_cwd)

    assert result.exit_code == 0, result.stdout
    assert "perfect_foresight" in result.stdout
    assert (tmp_path / "reports" / "backtest_metrics.csv").exists()
    assert (tmp_path / "reports" / "backtest_cost_sweep.csv").exists()
    assert (tmp_path / "reports" / "backtest_bootstrap.json").exists()
    assert (tmp_path / "reports" / "figures" / "backtest_cumulative_pnl.png").exists()
    assert (processed_dir / "backtest_positions.parquet").exists()


def test_backtest_command_skips_when_target_is_price_level(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    index = pd.date_range("2024-01-01", periods=50, freq="h", tz="Europe/Paris")
    matrix = pd.DataFrame({"target_day_ahead_price_eur_mwh": np.arange(50.0)}, index=index)
    cache.write_cache(matrix, processed_dir / "feature_matrix.parquet")
    (processed_dir / "feature_matrix.meta.json").write_text(
        json.dumps({"target_column": "target_day_ahead_price_eur_mwh"})
    )

    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(processed_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["backtest", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "skipped" in result.stdout


def test_backtest_command_errors_when_train_not_run_yet(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    index = pd.date_range("2024-01-01", periods=50, freq="h", tz="Europe/Paris")
    matrix = pd.DataFrame({"target_spread_eur_mwh": np.arange(50.0)}, index=index)
    cache.write_cache(matrix, processed_dir / "feature_matrix.parquet")
    (processed_dir / "feature_matrix.meta.json").write_text(
        json.dumps({"target_column": "target_spread_eur_mwh"})
    )

    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(processed_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["backtest", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "fr-power train" in result.stdout


def test_backtest_command_errors_when_features_not_run_yet(tmp_path: Path) -> None:
    with Path(DEFAULT_CONFIG_PATH).open() as f:
        config_dict = yaml.safe_load(f)
    config_dict["data"]["processed_dir"] = str(tmp_path / "empty_processed")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    result = runner.invoke(app, ["backtest", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "fr-power features" in result.stdout


def test_stub_commands_run_without_error() -> None:
    for command in ["report"]:
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        assert "stub" in result.stdout
