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

import pandas as pd
import typer

from fr_power.config import DEFAULT_CONFIG_PATH, Config, Secrets, load_config, load_secrets
from fr_power.data import entsoe, intraday, rte, weather

app = typer.Typer(
    name="fr-power",
    help="FR power intraday vs day-ahead spread forecasting and OU trading research pipeline.",
    no_args_is_help=True,
)

ConfigOption = typer.Option(
    DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to a YAML config file."
)
ForceRefreshOption = typer.Option(
    False, "--force-refresh", help="Re-fetch even if a local parquet cache already exists."
)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@app.command()
def fetch(config: Path = ConfigOption, force_refresh: bool = ForceRefreshOption) -> None:
    """Fetch and cache raw data (ENTSO-E, intraday, weather, RTE) to parquet."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    secrets = load_secrets()
    _run_fetch(cfg, secrets, force_refresh=force_refresh)


def _run_fetch(cfg: Config, secrets: Secrets, force_refresh: bool) -> None:
    log = logging.getLogger("fr_power.cli.fetch")
    start = pd.Timestamp(cfg.data.start_date, tz=cfg.project.timezone)
    end = pd.Timestamp(cfg.data.resolved_end_date(), tz=cfg.project.timezone)
    raw_dir = Path(cfg.data.raw_dir)

    try:
        client = entsoe.build_client(secrets)
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    log.info("Fetching ENTSO-E data for %s from %s to %s.", cfg.market.bidding_zone, start, end)
    entsoe_results = entsoe.fetch_all(
        client, start, end, raw_dir, cfg.data.entsoe, force_refresh=force_refresh
    )
    for name, df in entsoe_results.items():
        typer.echo(f"  entsoe/{name}: {len(df)} rows, {df.index.min()} -> {df.index.max()}")

    intraday_source = intraday.get_intraday_source(cfg.data.intraday)
    intraday_df = intraday_source.load()
    if intraday_df is None:
        typer.echo(
            "  intraday: not available -- pipeline will degrade to a "
            "day-ahead-price forecasting task (see warning above)."
        )
    else:
        typer.echo(f"  intraday: {len(intraday_df)} rows")

    weather_df = weather.fetch_and_cache_weather(
        cfg.data.weather,
        raw_dir,
        str(cfg.data.start_date),
        str(cfg.data.resolved_end_date()),
        timezone=cfg.project.timezone,
        force_refresh=force_refresh,
    )
    if weather_df is not None:
        typer.echo(f"  weather: {len(weather_df)} rows")

    eco2mix_df = rte.fetch_and_cache_eco2mix(
        cfg.data.rte_eco2mix, secrets, raw_dir, start, end, force_refresh=force_refresh
    )
    if eco2mix_df is not None:
        typer.echo(f"  rte_eco2mix: {len(eco2mix_df)} rows")


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
