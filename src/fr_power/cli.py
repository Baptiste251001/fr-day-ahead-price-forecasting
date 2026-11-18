"""Command-line pipeline: `fr-power fetch`, `features`, `train`, then `bid`."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from fr_power import bidding, data
from fr_power.config import DEFAULT_CONFIG_PATH, BiddingConfig, Config, load_config
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


@app.command()
def bid(config: Path = ConfigOption) -> None:
    """Backtest day-ahead bidding for a battery on the walk-forward forecasts."""
    cfg = _load(config)
    matrix = data.read_cache(cfg.data.processed_dir / FEATURES_FILE)
    predictions = data.read_cache(cfg.data.processed_dir / PREDICTIONS_FILE)
    if matrix is None or predictions is None:
        typer.echo("ERROR: no predictions found. Run `fr-power train` first.", err=True)
        raise typer.Exit(1)
    b = cfg.bidding
    battery = bidding.Battery(b.power_mw, b.energy_mwh, b.round_trip_efficiency)
    actual = matrix[TARGET]
    daily = bidding.backtest(
        actual,
        predictions,
        battery,
        schedule_models=b.schedule_models,
        curve_model=b.curve_model,
        window_days=b.scenario_window_days,
        min_days=b.scenario_min_days,
        imbalance_penalty=b.imbalance_penalty_eur_mwh,
    )
    if daily.empty:
        typer.echo("ERROR: no tradable day (forecasts or scenario history missing).", err=True)
        raise typer.Exit(1)
    summary = bidding.summarise(daily, battery)
    yearly = bidding.profit_by_year(daily, battery)
    reports = cfg.reports_dir
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(reports / "bidding.csv", index_label="strategy")
    yearly.round(0).to_csv(reports / "bidding_by_year.csv")

    typer.echo(f"\nBattery bidding backtest ({summary.attrs['n_days']:,} delivery days):")
    typer.echo(summary.round(3).to_string())
    typer.echo("\nProfit by year (EUR per MW):")
    typer.echo(yearly.round(0).to_string())
    _bidding_figure(daily, actual, predictions[b.curve_model], battery, b, reports)
    typer.echo(f"\n  reports written to {reports}/")


def _bidding_figure(
    daily: pd.DataFrame,
    actual: pd.Series,
    forecast: pd.Series,
    battery: bidding.Battery,
    b: BiddingConfig,
    reports_dir: Path,
) -> None:
    """Cumulative profit by strategy, and example bid curves from the last 30 traded days.

    The example is the day whose purchase and sale curves depend most on the price (on many
    days the optimal curves are almost flat: the battery trades the same volume anyway).
    """
    matplotlib.use("Agg")
    profit = daily.xs("profit_eur", axis=1, level="metric") / battery.power_mw / 1000
    errors = bidding.error_table(actual, forecast)

    def best_hours(quantities: np.ndarray) -> tuple[float, int, int]:
        spread, mean_q = np.ptp(quantities, axis=1), quantities.mean(axis=1)
        buy_spread, sell_spread = np.where(mean_q < 0, spread, 0), np.where(mean_q > 0, spread, 0)
        score = float(buy_spread.max() + sell_spread.max())
        return score, int(buy_spread.argmax()), int(sell_spread.argmax())

    candidates = []
    for ts in daily.index[-30:]:
        day = ts.date()
        day_forecast = forecast[forecast.index.date == day]
        scenarios = bidding.scenarios_for_day(
            day_forecast, errors, day, b.scenario_window_days, b.scenario_min_days
        )
        assert scenarios is not None  # the day was traded, so its scenarios exist
        curves = bidding.optimal_curves(scenarios, battery)
        candidates.append((best_hours(curves.quantities), day, day_forecast, curves))
    (_, buy, sell), day, day_forecast, curves = max(candidates, key=lambda c: c[0][0])
    realized = actual.reindex(day_forecast.index).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    for name in profit:
        axes[0].plot(profit.index.to_numpy(), profit[name].cumsum().to_numpy(), label=name)
    axes[0].set(ylabel="kEUR per MW", title="Cumulative day-ahead profit")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="grey", linewidth=0.6, zorder=0)
    for h in dict.fromkeys([buy, sell]):
        hour = day_forecast.index[h].hour
        (line,) = axes[1].plot(
            curves.prices[h], curves.quantities[h], marker=".", label=f"{hour:02d}:00"
        )
        axes[1].axvline(realized[h], color=line.get_color(), linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="price (EUR/MWh)",
        ylabel="net sale (MW), negative = purchase",
        title=f"Bid curves for {day} (dotted: cleared price)",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports_dir / "figures" / "bidding.png", dpi=120)
    plt.close(fig)


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
