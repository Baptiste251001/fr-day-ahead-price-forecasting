"""Command-line entry point for the fr_power pipeline.

Each pipeline stage is its own subcommand so that a full run is:

    fr-power fetch --config configs/default.yaml
    fr-power features --config configs/default.yaml
    fr-power train --config configs/default.yaml
    fr-power calibrate --config configs/default.yaml
    fr-power backtest --config configs/default.yaml
    fr-power report --config configs/default.yaml

`fetch`, `features`, `train`, `calibrate` and `backtest` are implemented;
`report` is a stub, implemented in the phase that owns it (see README
build-plan section).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import typer

from fr_power.backtest.engine import perfect_foresight_backtest, run_backtest
from fr_power.backtest.metrics import block_bootstrap_ci, compute_metrics
from fr_power.config import DEFAULT_CONFIG_PATH, Config, Secrets, load_config, load_secrets
from fr_power.data import cache, entsoe, intraday, rte, weather
from fr_power.features.build import build_feature_matrix
from fr_power.models.base import Forecaster
from fr_power.models.benchmarks import LearForecaster, NaivePersistenceForecaster, naive_lag_column
from fr_power.models.gbm import LightGBMForecaster
from fr_power.models.lstm import LSTMQuantileForecaster
from fr_power.models.metrics import diebold_mariano_test, evaluate_predictions
from fr_power.models.walk_forward import walk_forward_predict
from fr_power.ou.estimate import compute_z_score, estimate_ou
from fr_power.ou.optimal_stopping import generate_positions, thresholds_from_config

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


_REQUIRED_ENTSOE_CACHE_NAMES = {
    "day_ahead_prices": "entsoe_day_ahead_prices",
    "day_ahead_wind_solar_forecast": "entsoe_day_ahead_wind_solar_forecast",
    "day_ahead_load_forecast": "entsoe_day_ahead_load_forecast",
    "actual_generation_by_type": "entsoe_actual_generation_by_type",
}
# cross_border_flows is optional: a live run found ENTSO-E's FR net-position feed
# discontinued from 2025-09-30 onward, so `build_feature_matrix` treats it as optional
# and this is not required to exist even when `fr-power fetch` was run successfully.
_CROSS_BORDER_CACHE_NAME = "entsoe_net_cross_border_position"


@app.command()
def features(config: Path = ConfigOption) -> None:
    """Build the point-in-time feature matrix from cached raw data."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    _run_features(cfg)


def _run_features(cfg: Config) -> None:
    raw_dir = Path(cfg.data.raw_dir)

    missing = [
        name
        for name, cache_name in _REQUIRED_ENTSOE_CACHE_NAMES.items()
        if cache.read_cache(cache.cache_path(raw_dir, cache_name)) is None
    ]
    if missing:
        typer.echo(
            f"ERROR: missing cached ENTSO-E data for: {', '.join(missing)}. "
            "Run `fr-power fetch` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    raw = {
        name: cache.read_cache(cache.cache_path(raw_dir, cache_name))
        for name, cache_name in _REQUIRED_ENTSOE_CACHE_NAMES.items()
    }
    net_position_df = (
        cache.read_cache(cache.cache_path(raw_dir, _CROSS_BORDER_CACHE_NAME))
        if cfg.data.entsoe.cross_border_flows
        else None
    )
    weather_df = cache.read_cache(cache.cache_path(raw_dir, "weather_fr_population_weighted"))
    intraday_df = intraday.get_intraday_source(cfg.data.intraday).load()

    matrix = build_feature_matrix(
        day_ahead_prices=raw["day_ahead_prices"],
        wind_solar_forecast=raw["day_ahead_wind_solar_forecast"],
        load_forecast=raw["day_ahead_load_forecast"],
        actual_generation_by_type=raw["actual_generation_by_type"],
        config=cfg.features,
        net_position=net_position_df,
        weather=weather_df,
        intraday=intraday_df,
    )

    out_path = Path(cfg.data.processed_dir) / "feature_matrix.parquet"
    cache.write_cache(matrix, out_path)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"target_column": matrix.attrs["target_column"]}))

    typer.echo(f"  feature matrix: {matrix.shape[0]} rows x {matrix.shape[1]} cols -> {out_path}")
    typer.echo(f"  target column: {matrix.attrs['target_column']}")


_DM_COMPARISONS = [
    ("lear", "naive"),
    ("lightgbm", "naive"),
    ("lightgbm", "lear"),
    ("lstm", "naive"),
    ("lstm", "lear"),
]


@app.command()
def train(config: Path = ConfigOption) -> None:
    """Train and walk-forward evaluate the forecasting models (naive, LEAR, LightGBM, LSTM)."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    _run_train(cfg)


def _build_model_factories(cfg: Config, target_col: str) -> dict[str, Callable[[], Forecaster]]:
    naive_lag_col = naive_lag_column(target_col, cfg.models.naive.lag_hours)
    seed = cfg.project.random_seed
    return {
        "naive": lambda: NaivePersistenceForecaster(naive_lag_col),
        "lear": lambda: LearForecaster(
            cfg.models.lear.lasso_alpha_grid, cfg.models.lear.cv_folds, random_seed=seed
        ),
        "lightgbm": lambda: LightGBMForecaster(
            **cfg.models.lightgbm.model_dump(), random_seed=seed
        ),
        "lstm": lambda: LSTMQuantileForecaster(**cfg.models.lstm.model_dump(), random_seed=seed),
    }


def _run_train(cfg: Config) -> None:
    processed_path = Path(cfg.data.processed_dir) / "feature_matrix.parquet"
    meta_path = processed_path.with_suffix(".meta.json")
    if not processed_path.exists() or not meta_path.exists():
        typer.echo("ERROR: no feature matrix found. Run `fr-power features` first.", err=True)
        raise typer.Exit(code=1)

    matrix = cache.read_cache(processed_path)
    assert matrix is not None
    target_col = json.loads(meta_path.read_text())["target_column"]
    feature_cols = [c for c in matrix.columns if not c.startswith("target_")]

    wf = cfg.backtest.walk_forward
    predictions = walk_forward_predict(
        matrix,
        feature_cols,
        target_col,
        _build_model_factories(cfg, target_col),
        initial_train_hours=wf.initial_train_days * 24,
        retrain_every_hours=wf.retrain_every_days * 24,
        test_window_hours=wf.test_window_days * 24,
    )

    actual = matrix[target_col]
    metrics_df = pd.DataFrame(
        {name: evaluate_predictions(actual, preds) for name, preds in predictions.items()}
    ).T
    typer.echo(metrics_df.to_string())

    dm_rows = []
    for challenger, baseline in _DM_COMPARISONS:
        if challenger not in predictions or baseline not in predictions:
            continue
        errors_challenger = actual - predictions[challenger]
        errors_baseline = actual - predictions[baseline]
        aligned = pd.concat(
            [errors_challenger.rename("c"), errors_baseline.rename("b")], axis=1
        ).dropna()
        if len(aligned) < 2:
            continue
        dm_stat, p_value = diebold_mariano_test(aligned["c"], aligned["b"])
        dm_rows.append(
            {"challenger": challenger, "baseline": baseline, "dm_stat": dm_stat, "p_value": p_value}
        )
    dm_df = pd.DataFrame(dm_rows)
    if not dm_df.empty:
        typer.echo(dm_df.to_string(index=False))

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(reports_dir / "forecast_metrics.csv")
    dm_df.to_csv(reports_dir / "dm_tests.csv", index=False)
    _plot_rmse_comparison(metrics_df, figures_dir / "forecast_rmse.png")

    predictions_path = Path(cfg.data.processed_dir) / "forecast_predictions.parquet"
    cache.write_cache(pd.DataFrame(predictions), predictions_path)

    typer.echo(f"  saved: {reports_dir / 'forecast_metrics.csv'}")
    typer.echo(f"  saved: {reports_dir / 'dm_tests.csv'}")
    typer.echo(f"  saved: {figures_dir / 'forecast_rmse.png'}")
    typer.echo(f"  saved: {predictions_path}")


def _plot_rmse_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_df["rmse"].plot.bar(ax=ax)
    ax.set_ylabel("RMSE")
    ax.set_title("Out-of-sample forecast RMSE by model")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@app.command()
def calibrate(config: Path = ConfigOption) -> None:
    """Calibrate the OU spread model and derive optimal-stopping execution thresholds."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    _run_calibrate(cfg)


def _run_calibrate(cfg: Config) -> None:
    processed_path = Path(cfg.data.processed_dir) / "feature_matrix.parquet"
    meta_path = processed_path.with_suffix(".meta.json")
    if not processed_path.exists() or not meta_path.exists():
        typer.echo("ERROR: no feature matrix found. Run `fr-power features` first.", err=True)
        raise typer.Exit(code=1)

    matrix = cache.read_cache(processed_path)
    assert matrix is not None
    target_col = json.loads(meta_path.read_text())["target_column"]

    if target_col != "target_spread_eur_mwh":
        typer.echo(
            "OU calibration skipped: no intraday-vs-day-ahead spread series is available "
            "(the target degraded to the day-ahead price level -- see `fr-power fetch`/"
            "`features`). A price *level* is not economically mean-reverting in the sense "
            "this model assumes, so it is not calibrated.",
            err=True,
        )
        raise typer.Exit(code=1)

    spread = matrix[target_col]
    params = estimate_ou(
        spread, method=cfg.ou.estimation.method, min_observations=cfg.ou.estimation.min_observations
    )
    thresholds = thresholds_from_config(cfg.ou.optimal_stopping)

    typer.echo(f"  method: {params.method}")
    typer.echo(f"  kappa (mean-reversion speed): {params.kappa:.4f} / period")
    typer.echo(f"  mu (long-run mean): {params.mu:.4f} EUR/MWh")
    typer.echo(f"  sigma: {params.sigma:.4f}")
    typer.echo(f"  half-life: {params.half_life:.2f} periods")
    typer.echo(f"  stationary std: {params.stationary_std:.4f} EUR/MWh")
    typer.echo(f"  R^2: {params.r_squared:.4f}  (n={params.n_observations})")

    z_scores = compute_z_score(spread, params)
    positions = generate_positions(z_scores, thresholds)

    n_trades = int(positions["is_entry"].sum())
    hours_in_position = int((positions["position"] != "flat").sum())
    avg_holding_hours = hours_in_position / n_trades if n_trades else 0.0
    typer.echo(f"  trades: {n_trades}, avg holding: {avg_holding_hours:.1f}h")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": params.method,
        "kappa": params.kappa,
        "mu": params.mu,
        "sigma": params.sigma,
        "half_life_periods": params.half_life,
        "stationary_std": params.stationary_std,
        "r_squared": params.r_squared,
        "n_observations": params.n_observations,
        "thresholds": {
            "entry_z": thresholds.entry_z,
            "exit_z": thresholds.exit_z,
            "no_trade_z": thresholds.no_trade_z,
            "max_holding_hours": thresholds.max_holding_hours,
        },
        "n_trades": n_trades,
        "avg_holding_hours": avg_holding_hours,
    }
    summary_path = reports_dir / "ou_calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    positions_path = Path(cfg.data.processed_dir) / "ou_positions.parquet"
    cache.write_cache(positions.assign(z_score=z_scores), positions_path)

    typer.echo(f"  saved: {summary_path}")
    typer.echo(f"  saved: {positions_path}")


@app.command()
def backtest(config: Path = ConfigOption) -> None:
    """Run the walk-forward, cost-aware backtest and save results."""
    cfg = load_config(config)
    _setup_logging(cfg.logging.level)
    _run_backtest(cfg)


def _run_backtest(cfg: Config) -> None:
    processed_dir = Path(cfg.data.processed_dir)
    matrix_path = processed_dir / "feature_matrix.parquet"
    meta_path = matrix_path.with_suffix(".meta.json")
    predictions_path = processed_dir / "forecast_predictions.parquet"

    if not matrix_path.exists() or not meta_path.exists():
        typer.echo("ERROR: no feature matrix found. Run `fr-power features` first.", err=True)
        raise typer.Exit(code=1)
    target_col = json.loads(meta_path.read_text())["target_column"]
    if target_col != "target_spread_eur_mwh":
        typer.echo(
            "Backtest skipped: no intraday-vs-day-ahead spread series is available "
            "(the target degraded to the day-ahead price level). See `fr-power calibrate` "
            "for the same limitation.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not predictions_path.exists():
        typer.echo("ERROR: no forecast predictions found. Run `fr-power train` first.", err=True)
        raise typer.Exit(code=1)

    matrix = cache.read_cache(matrix_path)
    assert matrix is not None
    predictions_df = cache.read_cache(predictions_path)
    assert predictions_df is not None

    spread = matrix[target_col]
    signal_lag = min(cfg.features.spread_lags_hours)
    signal_ref_col = f"spread_lag_{signal_lag}h"
    if signal_ref_col not in matrix.columns:
        typer.echo(
            f"ERROR: expected lag feature '{signal_ref_col}' not found in feature matrix.", err=True
        )
        raise typer.Exit(code=1)
    signal_reference = matrix[signal_ref_col]

    initial_train_hours = cfg.backtest.walk_forward.initial_train_days * 24

    results: dict[str, pd.DataFrame] = {}
    for model_name in predictions_df.columns:
        results[model_name] = run_backtest(
            spread, signal_reference, predictions_df[model_name], cfg.costs, cfg.backtest, cfg.ou
        )
    results["perfect_foresight"] = perfect_foresight_backtest(
        spread, cfg.costs, cfg.backtest, start_index=initial_train_hours
    )

    metrics_df = pd.DataFrame(
        {name: compute_metrics(res).to_dict() for name, res in results.items()}
    ).T
    typer.echo(metrics_df.to_string())

    signal_model = cfg.backtest.signal_model
    if signal_model not in results:
        typer.echo(
            f"ERROR: configured backtest.signal_model={signal_model!r} has no predictions "
            f"in {predictions_path}. Available: {list(predictions_df.columns)}",
            err=True,
        )
        raise typer.Exit(code=1)

    bootstrap = block_bootstrap_ci(
        results[signal_model]["net_pnl_eur"],
        block_length_hours=cfg.backtest.block_bootstrap.block_length_days * 24,
        n_resamples=cfg.backtest.block_bootstrap.n_resamples,
        confidence_level=cfg.backtest.block_bootstrap.confidence_level,
        seed=cfg.project.random_seed,
    )
    typer.echo(f"  bootstrap CI ({signal_model}): {bootstrap}")

    sweep_rows = []
    for multiplier in cfg.backtest.cost_sensitivity_multipliers:
        swept = run_backtest(
            spread,
            signal_reference,
            predictions_df[signal_model],
            cfg.costs,
            cfg.backtest,
            cfg.ou,
            cost_multiplier=multiplier,
        )
        row = compute_metrics(swept).to_dict()
        row["cost_multiplier"] = multiplier
        sweep_rows.append(row)
    sweep_df = pd.DataFrame(sweep_rows)
    typer.echo(sweep_df.to_string(index=False))

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(reports_dir / "backtest_metrics.csv")
    sweep_df.to_csv(reports_dir / "backtest_cost_sweep.csv", index=False)
    (reports_dir / "backtest_bootstrap.json").write_text(
        json.dumps({"signal_model": signal_model, "confidence_intervals": bootstrap}, indent=2)
    )
    _plot_cumulative_pnl(results, signal_model, figures_dir / "backtest_cumulative_pnl.png")

    positions_path = processed_dir / "backtest_positions.parquet"
    cache.write_cache(results[signal_model], positions_path)

    typer.echo(f"  saved: {reports_dir / 'backtest_metrics.csv'}")
    typer.echo(f"  saved: {reports_dir / 'backtest_cost_sweep.csv'}")
    typer.echo(f"  saved: {reports_dir / 'backtest_bootstrap.json'}")
    typer.echo(f"  saved: {figures_dir / 'backtest_cumulative_pnl.png'}")
    typer.echo(f"  saved: {positions_path}")


def _plot_cumulative_pnl(results: dict[str, pd.DataFrame], signal_model: str, path: Path) -> None:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in [signal_model, "naive", "perfect_foresight"]:
        if name in results:
            results[name]["net_pnl_eur"].cumsum().plot(ax=ax, label=name)
    ax.set_ylabel("Cumulative net PnL (EUR)")
    ax.set_title("Backtest cumulative net PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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
