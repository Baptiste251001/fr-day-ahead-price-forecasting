"""Tests for the backtest engine: signal formula, gating rules, point-in-time
safety of the walk-forward loop, and the perfect-foresight comparator."""

import numpy as np
import pandas as pd
import pytest

from fr_power.backtest.engine import compute_fold_signal, perfect_foresight_backtest, run_backtest
from fr_power.config import (
    BacktestConfig,
    BlockBootstrapConfig,
    CostsConfig,
    OuConfig,
    OuEstimationConfig,
    OuOptimalStoppingConfig,
    WalkForwardConfig,
)


def _index(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="Europe/Paris")


def _configs(
    initial_train_days: int = 5,
    retrain_every_days: int = 5,
    test_window_days: int = 5,
    no_trade_band: float = 0.1,
    max_position: float = 1.0,
    fee: float = 0.05,
    half_spread: float = 0.05,
    entry_z: float = 1.0,
) -> tuple[CostsConfig, BacktestConfig, OuConfig]:
    costs = CostsConfig(fee_eur_per_mwh=fee, half_spread_eur_per_mwh=half_spread)
    backtest_cfg = BacktestConfig(
        walk_forward=WalkForwardConfig(
            initial_train_days=initial_train_days,
            retrain_every_days=retrain_every_days,
            test_window_days=test_window_days,
        ),
        no_trade_band_eur_per_mwh=no_trade_band,
        max_position_mwh=max_position,
        block_bootstrap=BlockBootstrapConfig(
            block_length_days=1, n_resamples=50, confidence_level=0.9
        ),
        cost_sensitivity_multipliers=[0.0, 1.0, 2.0],
        signal_model="lightgbm",
    )
    ou_cfg = OuConfig(
        estimation=OuEstimationConfig(method="ols", min_observations=20),
        optimal_stopping=OuOptimalStoppingConfig(
            entry_z_score=entry_z, exit_z_score=0.3, no_trade_z_score=0.1, max_holding_hours=6
        ),
    )
    return costs, backtest_cfg, ou_cfg


# ---------------------------------------------------------------------------
# compute_fold_signal: exact, hand-verifiable behavior given known OU params
# ---------------------------------------------------------------------------


def test_direction_from_z_score_threshold() -> None:
    index = _index(5)
    reference = pd.Series([2.0, 4.0, 0.0, 2.9, 2.0], index=index)  # mu=2, std=1 -> z=[0,2,-2,0.9,0]
    predictions = reference

    signal = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.0,
        cost_per_mwh=0.0,
        max_position_mwh=1.0,
    )

    assert signal["z_score"].tolist() == pytest.approx([0.0, 2.0, -2.0, 0.9, 0.0])
    assert signal["direction"].tolist() == [0.0, -1.0, 1.0, 0.0, 0.0]


def test_disagreeing_forecast_gates_trade_to_zero() -> None:
    index = _index(1)
    reference = pd.Series([4.0], index=index)  # z=2.0 -> direction=-1 (expect a fall)
    predictions = pd.Series([5.0], index=index)  # forecast_edge=+1.0 -> predicts a rise: disagrees

    signal = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.0,
        cost_per_mwh=0.0,
        max_position_mwh=1.0,
    )

    assert signal["direction"].iloc[0] == -1.0
    assert signal["confidence"].iloc[0] == 0.0
    assert signal["size_mwh"].iloc[0] == 0.0


def test_agreeing_forecast_sizes_by_confidence() -> None:
    index = _index(1)
    reference = pd.Series([4.0], index=index)  # z=2.0 -> direction=-1
    predictions = pd.Series([3.5], index=index)  # forecast_edge=-0.5 -> agrees

    signal = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.0,
        cost_per_mwh=0.0,
        max_position_mwh=2.0,
    )

    assert signal["confidence"].iloc[0] == pytest.approx(0.5)
    assert signal["size_mwh"].iloc[0] == pytest.approx(
        -1.0
    )  # direction(-1) * max(2.0) * confidence(0.5)


def test_confidence_capped_at_one() -> None:
    index = _index(1)
    reference = pd.Series([4.0], index=index)
    predictions = pd.Series([-10.0], index=index)  # huge agreeing edge

    signal = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.0,
        cost_per_mwh=0.0,
        max_position_mwh=1.0,
    )

    assert signal["confidence"].iloc[0] == 1.0
    assert signal["size_mwh"].iloc[0] == -1.0


def test_no_trade_band_gates_small_edges() -> None:
    index = _index(1)
    reference = pd.Series([4.0], index=index)  # direction=-1
    predictions = pd.Series([3.85], index=index)  # forecast_edge=-0.15, agrees but small

    gated = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.5,
        cost_per_mwh=0.0,
        max_position_mwh=1.0,
    )
    assert gated["size_mwh"].iloc[0] == 0.0

    not_gated = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.1,
        cost_per_mwh=0.0,
        max_position_mwh=1.0,
    )
    assert not_gated["size_mwh"].iloc[0] != 0.0


def test_higher_cost_gates_a_previously_tradeable_edge() -> None:
    index = _index(1)
    reference = pd.Series([4.0], index=index)
    predictions = pd.Series([3.7], index=index)  # forecast_edge=-0.3

    cheap = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.1,
        cost_per_mwh=0.05,
        max_position_mwh=1.0,
    )
    assert cheap["size_mwh"].iloc[0] != 0.0

    expensive = compute_fold_signal(
        reference,
        predictions,
        mu=2.0,
        stationary_std=1.0,
        entry_z=1.0,
        no_trade_band_eur_per_mwh=0.1,
        cost_per_mwh=0.5,
        max_position_mwh=1.0,
    )
    assert expensive["size_mwh"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# run_backtest: walk-forward point-in-time safety
# ---------------------------------------------------------------------------


def _simulate_ou_path(
    kappa: float, mu: float, sigma: float, n: int, x0: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.exp(-kappa)
    std_eps = sigma * np.sqrt((1 - a**2) / (2 * kappa))
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = mu + a * (x[t - 1] - mu) + std_eps * rng.standard_normal()
    return x


def test_run_backtest_decisions_do_not_depend_on_same_fold_realized_spread() -> None:
    n_train, n_test = 360, 120
    n = n_train + n_test
    index = _index(n)

    train_values = _simulate_ou_path(0.05, 2.0, 1.0, n_train, 2.0, seed=1)
    rng = np.random.default_rng(2)
    spread_a = pd.Series(
        np.concatenate([train_values, 2.0 + rng.normal(scale=1.0, size=n_test)]), index=index
    )
    spread_b = pd.Series(
        np.concatenate([train_values, 2.0 + rng.normal(scale=1.0, size=n_test)]), index=index
    )

    reference = pd.Series(2.0 + np.sin(np.arange(n) / 10.0), index=index)
    predictions = reference - 0.8

    costs, backtest_cfg, ou_cfg = _configs(
        initial_train_days=15, retrain_every_days=10, test_window_days=10
    )

    result_a = run_backtest(spread_a, reference, predictions, costs, backtest_cfg, ou_cfg)
    result_b = run_backtest(spread_b, reference, predictions, costs, backtest_cfg, ou_cfg)

    for col in ["z_score", "forecast_edge", "direction", "confidence", "size_mwh"]:
        pd.testing.assert_series_equal(result_a[col], result_b[col], check_names=False)

    assert not result_a["gross_pnl_eur"].equals(result_b["gross_pnl_eur"])


def test_run_backtest_produces_no_signal_before_initial_train_window() -> None:
    n = 300
    index = _index(n)
    values = _simulate_ou_path(0.05, 2.0, 1.0, n, 2.0, seed=3)
    spread = pd.Series(values, index=index)
    reference = spread.shift(1).bfill()
    predictions = reference - 1.0

    costs, backtest_cfg, ou_cfg = _configs(
        initial_train_days=10, retrain_every_days=5, test_window_days=5
    )
    result = run_backtest(spread, reference, predictions, costs, backtest_cfg, ou_cfg)

    initial_train_hours = 10 * 24
    assert result["z_score"].iloc[:initial_train_hours].isna().all()
    assert result["z_score"].iloc[initial_train_hours:].notna().any()


def test_run_backtest_returns_empty_when_no_test_period(caplog: pytest.LogCaptureFixture) -> None:
    n = 50
    index = _index(n)
    spread = pd.Series(np.full(n, 2.0), index=index)
    reference = spread.copy()
    predictions = spread.copy()

    costs, backtest_cfg, ou_cfg = _configs(
        initial_train_days=100, retrain_every_days=5, test_window_days=5
    )
    with caplog.at_level("WARNING"):
        result = run_backtest(spread, reference, predictions, costs, backtest_cfg, ou_cfg)

    assert (result["size_mwh"] == 0).all()
    assert result["z_score"].isna().all()
    assert "no backtest folds possible" in caplog.text


# ---------------------------------------------------------------------------
# perfect_foresight_backtest
# ---------------------------------------------------------------------------


def test_perfect_foresight_trades_correct_direction_when_above_band() -> None:
    index = _index(4)
    spread = pd.Series([2.0, -2.0, 0.05, -0.05], index=index)
    costs = CostsConfig(fee_eur_per_mwh=0.0, half_spread_eur_per_mwh=0.0)
    _, backtest_cfg, _ = _configs(no_trade_band=0.5, max_position=3.0)

    result = perfect_foresight_backtest(spread, costs, backtest_cfg)

    assert result["direction"].tolist() == [1.0, -1.0, 0.0, 0.0]
    assert result["size_mwh"].tolist() == [3.0, -3.0, 0.0, 0.0]
    assert result["gross_pnl_eur"].tolist() == pytest.approx([6.0, 6.0, 0.0, 0.0])
