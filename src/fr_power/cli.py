"""Command-line pipeline: `fr-power fetch`, then `fr-power features`, then `fr-power train`."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import typer

from fr_power import data
from fr_power.config import DEFAULT_CONFIG_PATH, Config, load_config
from fr_power.evaluation import (
    conformal_intervals,
    dm_table,
    interval_metrics,
    mae_by_year,
    point_metrics,
)
from fr_power.features import TARGET, build_feature_matrix, feature_columns
from fr_power.models import add_ensemble, build_models, walk_forward_predict

app = typer.Typer(help="French day-ahead electricity price forecasting.", no_args_is_help=True)
ConfigOption = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="YAML config file.")

FEATURES_FILE = "feature_matrix.parquet"
PREDICTIONS_FILE = "predictions.parquet"


def _load(config: Path) -> Config:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return load_config(config)


@app.command()
def fetch(
    config: Path = ConfigOption,
    force_refresh: bool = typer.Option(False, help="Re-download even if cached."),
) -> None:
    """Download ENTSO-E, Open-Meteo and gas-price data to data/raw/ (cached as parquet)."""
    cfg = _load(config)
    start, end = cfg.data.start_date, cfg.data.resolved_end_date()
    try:
        client = data.entsoe_client()
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    results = data.fetch_entsoe(
        client,
        pd.Timestamp(start, tz=data.TIMEZONE),
        pd.Timestamp(end, tz=data.TIMEZONE),
        cfg.data.raw_dir,
        force_refresh=force_refresh,
    )
    other_sources = {
        data.WEATHER: lambda: data.fetch_weather(cfg.data.weather, str(start), str(end)),
        data.GAS: lambda: data.fetch_gas(cfg.data.gas_ticker, str(start), str(end)),
    }
    for name, fetch_fn in other_sources.items():
        results[name] = data.cached_fetch(
            cfg.data.raw_dir, name, fetch_fn, force_refresh=force_refresh
        )
    for name, df in results.items():
        typer.echo(f"  {name}: {len(df):,} rows, {df.index.min()} -> {df.index.max()}")


@app.command()
def features(config: Path = ConfigOption) -> None:
    """Build the point-in-time feature matrix from the cached raw data."""
    cfg = _load(config)
    try:
        raw = data.load_raw(cfg.data.raw_dir)
    except FileNotFoundError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    matrix = build_feature_matrix(raw, cfg.features)
    out = cfg.data.processed_dir / FEATURES_FILE
    data.write_cache(matrix, out)
    typer.echo(f"  feature matrix: {matrix.shape[0]:,} rows x {matrix.shape[1]} cols -> {out}")


@app.command()
def train(config: Path = ConfigOption) -> None:
    """Walk-forward train and evaluate all models; write predictions and reports."""
    cfg = _load(config)
    matrix = data.read_cache(cfg.data.processed_dir / FEATURES_FILE)
    if matrix is None:
        typer.echo("ERROR: no feature matrix found. Run `fr-power features` first.", err=True)
        raise typer.Exit(1)
    matrix = matrix[matrix[TARGET].notna()]

    wf = cfg.walk_forward
    predictions = walk_forward_predict(
        matrix[feature_columns(matrix)],
        matrix[TARGET],
        build_models(cfg),
        initial_train_days=wf.initial_train_days,
        retrain_every_days=wf.retrain_every_days,
    )
    predictions = add_ensemble(predictions, cfg.models.ensemble)
    data.write_cache(predictions, cfg.data.processed_dir / PREDICTIONS_FILE)
    write_reports(matrix[TARGET], predictions, cfg.reports_dir)


def write_reports(actual: pd.Series, predictions: pd.DataFrame, reports_dir: Path) -> None:
    """Metric tables (CSV) and figures (PNG) under `reports_dir`."""
    matplotlib.use("Agg")
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    metrics = point_metrics(actual, predictions)
    yearly = mae_by_year(actual, predictions)
    dm = dm_table(actual, predictions)
    metrics.round(3).to_csv(reports_dir / "metrics.csv", index_label="model")
    yearly.round(3).to_csv(reports_dir / "mae_by_year.csv")
    dm.round(4).to_csv(reports_dir / "dm_tests.csv", index=False)
    intervals = {m: conformal_intervals(actual, predictions[m]) for m in predictions}
    interval_table = pd.DataFrame(
        {m: interval_metrics(actual, iv) for m, iv in intervals.items()}
    ).T
    interval_table.round(3).to_csv(reports_dir / "intervals.csv", index_label="model")

    typer.echo(f"\nOut-of-sample metrics ({metrics.attrs['n_hours']:,} hours, EUR/MWh):")
    typer.echo(metrics.round(2).to_string())
    typer.echo("\nMAE by year:")
    typer.echo(yearly.round(2).to_string())
    typer.echo("\nDiebold-Mariano tests (negative = model_a more accurate):")
    typer.echo(dm.round(4).to_string(index=False))
    typer.echo("\n90% rolling conformal intervals:")
    typer.echo(interval_table.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    yearly.plot.bar(ax=ax, rot=0)
    ax.set(ylabel="MAE (EUR/MWh)", xlabel="", title="Out-of-sample MAE by year")
    fig.tight_layout()
    fig.savefig(figures / "mae_by_year.png", dpi=120)
    plt.close(fig)

    best = "ensemble" if "ensemble" in predictions else predictions.columns[-1]
    last = predictions.dropna().index.max()
    week = slice(last - pd.Timedelta(days=14), last)
    band = intervals[best].loc[week]
    hours = band.index.tz_localize(None)  # plain matplotlib, local wall-clock time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hours, band["lower"], band["upper"], alpha=0.25, label="90% interval")
    ax.plot(hours, actual.reindex(band.index), color="black", linewidth=1.6, label="actual")
    ax.plot(hours, predictions[best].loc[week], linewidth=1.2, label=best)
    ax.plot(hours, predictions["naive"].loc[week], linewidth=0.8, alpha=0.6, label="naive")
    ax.set(ylabel="EUR/MWh", title="Day-ahead price: last two weeks out-of-sample")
    fig.autofmt_xdate()
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(figures / "last_two_weeks.png", dpi=120)
    plt.close(fig)
    typer.echo(f"\n  reports written to {reports_dir}/")


if __name__ == "__main__":
    app()
