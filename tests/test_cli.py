"""Smoke tests for the CLI skeleton (Phase 0 scaffolding)."""

from typer.testing import CliRunner

from fr_power.cli import app

runner = CliRunner()


def test_cli_help_lists_all_stage_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ["fetch", "features", "train", "calibrate", "backtest", "report"]:
        assert command in result.stdout


def test_fetch_stub_runs_without_error() -> None:
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 0
    assert "stub" in result.stdout
