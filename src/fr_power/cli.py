"""Command-line entry point for the fr_power pipeline.

Each pipeline stage is its own subcommand so that a full run is:

    fr-power fetch --config configs/default.yaml
    fr-power features --config configs/default.yaml
    fr-power train --config configs/default.yaml
    fr-power calibrate --config configs/default.yaml
    fr-power backtest --config configs/default.yaml
    fr-power report --config configs/default.yaml

Every command is a stub in Phase 0; each is implemented in the phase that
owns it (see README build-plan section).
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from fr_power.config import DEFAULT_CONFIG_PATH, load_config, load_secrets

app = typer.Typer(
    name="fr-power",
    help="FR power intraday vs day-ahead spread forecasting and OU trading research pipeline.",
    no_args_is_help=True,
)

ConfigOption = typer.Option(
    DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to a YAML config file."
)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@app.command()
def fetch(config: Path = ConfigOption) -> None:
    """Fetch and cache raw data (ENTSO-E, intraday, weather, RTE) to parquet."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    load_secrets()
    typer.echo(
        f"[stub] fetch: would pull data for {cfg.market.bidding_zone} "
        f"from {cfg.data.start_date} to {cfg.data.resolved_end_date()}. "
        "Implemented in Phase 1."
    )


@app.command()
def features(config: Path = ConfigOption) -> None:
    """Build the point-in-time feature matrix from cached raw data."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    typer.echo(
        "[stub] features: would build the point-in-time feature matrix. Implemented in Phase 2."
    )


@app.command()
def train(config: Path = ConfigOption) -> None:
    """Train and walk-forward evaluate the forecasting models (naive, LEAR, LightGBM, LSTM)."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    typer.echo(
        "[stub] train: would run walk-forward forecast training and evaluation. Implemented in Phase 3."
    )


@app.command()
def calibrate(config: Path = ConfigOption) -> None:
    """Calibrate the OU spread model and derive optimal-stopping execution thresholds."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    typer.echo(
        "[stub] calibrate: would fit OU parameters and stopping thresholds. Implemented in Phase 4."
    )


@app.command()
def backtest(config: Path = ConfigOption) -> None:
    """Run the walk-forward, cost-aware backtest and save results."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    typer.echo("[stub] backtest: would run the walk-forward backtest. Implemented in Phase 5.")


@app.command()
def report(config: Path = ConfigOption) -> None:
    """Generate figures and the written conclusions for the French market analysis."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    typer.echo(
        "[stub] report: would generate figures and reports/conclusions.md. Implemented in Phase 6."
    )


if __name__ == "__main__":
    app()
