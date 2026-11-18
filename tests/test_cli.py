"""End-to-end pipeline run on synthetic raw data, plus the shipped default config."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from fr_power import data
from fr_power.cli import app
from fr_power.config import DEFAULT_CONFIG_PATH, Config, load_config

runner = CliRunner()


def _write_config(config: Config, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return path


def test_default_config_is_valid() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.data.resolved_end_date() > cfg.data.start_date


def test_features_then_train(
    raw_mtu: dict[str, pd.DataFrame], config: Config, tmp_path: Path
) -> None:
    for name, df in raw_mtu.items():
        data.write_cache(df, data.cache_path(config.data.raw_dir, name))
    config_path = _write_config(config, tmp_path)

    result = runner.invoke(app, ["features", "-c", str(config_path)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["train", "-c", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "Diebold-Mariano" in result.output

    predictions = pd.read_parquet(config.data.processed_dir / "predictions_hourly.parquet")
    assert list(predictions.columns) == [
        "naive",
        "lasso",
        "lightgbm",
        "extra_trees",
        "ensemble",
    ]
    quarters = pd.read_parquet(config.data.processed_dir / "predictions.parquet")
    assert pd.Series(quarters.index).diff().max() == pd.Timedelta("15min")
    shape = pd.read_csv(config.reports_dir / "intra_hour_shape.csv", index_col="forecast")["mae"]
    assert shape["shape model"] < shape["zero (hourly flat)"]  # the shape model adds skill
    reports = [
        "metrics.csv",
        "mae_by_year.csv",
        "dm_tests.csv",
        "intervals.csv",
        "metrics_15min.csv",
    ]
    figures = ["mae_by_year.png", "last_two_weeks.png", "quarter_hour_day.png"]
    for name in [*reports, *[f"figures/{f}" for f in figures]]:
        assert (config.reports_dir / name).exists(), name

    result = runner.invoke(app, ["bid", "-c", str(config_path)])
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(config.reports_dir / "bidding.csv", index_col="strategy")
    assert summary.loc["perfect_foresight", "capture_vs_perfect"] == 1.0
    assert (summary["capture_vs_perfect"] <= 1.0 + 1e-9).all()
    assert (config.reports_dir / "figures" / "bidding.png").exists()


def test_commands_fail_cleanly_without_inputs(config: Config, tmp_path: Path) -> None:
    config_path = _write_config(config, tmp_path)
    for command in ["features", "train", "bid"]:
        result = runner.invoke(app, [command, "-c", str(config_path)])
        assert result.exit_code == 1
        assert "ERROR" in result.output
