"""Tests for backtest performance metrics."""

import numpy as np
import pandas as pd
import pytest

from fr_power.backtest.metrics import HOURS_PER_YEAR, block_bootstrap_ci, compute_metrics


def _result(
    size: list[float], gross: list[float], cost: list[float], net: list[float]
) -> pd.DataFrame:
    n = len(size)
    index = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Paris")
    return pd.DataFrame(
        {"size_mwh": size, "gross_pnl_eur": gross, "cost_eur": cost, "net_pnl_eur": net},
        index=index,
    )


def test_compute_metrics_basic_totals() -> None:
    result = _result(
        size=[1.0, -1.0, 0.0, 2.0],
        gross=[10.0, 5.0, 0.0, -4.0],
        cost=[1.0, 1.0, 0.0, 2.0],
        net=[9.0, 4.0, 0.0, -6.0],
    )
    metrics = compute_metrics(result)

    assert metrics.gross_pnl_eur == pytest.approx(11.0)
    assert metrics.net_pnl_eur == pytest.approx(7.0)
    assert metrics.total_cost_eur == pytest.approx(4.0)
    assert metrics.n_trades == 3
    assert metrics.traded_volume_mwh == pytest.approx(4.0)
    assert metrics.turnover_mwh == pytest.approx(4.0)
    assert metrics.pnl_per_mwh_eur == pytest.approx(7.0 / 4.0)
    assert metrics.hit_rate == pytest.approx(2 / 3)


def test_compute_metrics_hit_rate_nan_when_never_traded() -> None:
    result = _result(size=[0.0, 0.0], gross=[0.0, 0.0], cost=[0.0, 0.0], net=[0.0, 0.0])
    metrics = compute_metrics(result)
    assert metrics.n_trades == 0
    assert np.isnan(metrics.hit_rate)
    assert metrics.pnl_per_mwh_eur == 0.0


def test_compute_metrics_sharpe_uses_hourly_annualization() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    net = rng.normal(loc=0.1, scale=1.0, size=n)
    result = _result(size=list(np.ones(n)), gross=list(net), cost=list(np.zeros(n)), net=list(net))

    metrics = compute_metrics(result)
    # pandas .std() defaults to ddof=1 (sample std); match that convention here.
    expected = net.mean() / net.std(ddof=1) * np.sqrt(HOURS_PER_YEAR)
    assert metrics.sharpe_annualized == pytest.approx(expected, rel=1e-9)


def test_compute_metrics_max_drawdown_on_known_path() -> None:
    # cumulative pnl: 10, 15, 5, 8, 20 -> running max: 10,15,15,15,20 -> drawdown: 0,0,10,7,0 -> max=10
    net = [10.0, 5.0, -10.0, 3.0, 12.0]
    result = _result(size=list(np.ones(5)), gross=net, cost=list(np.zeros(5)), net=net)
    metrics = compute_metrics(result)
    assert metrics.max_drawdown_eur == pytest.approx(10.0)


def test_compute_metrics_zero_std_gives_zero_sharpe() -> None:
    net = [1.0, 1.0, 1.0]
    result = _result(size=list(np.ones(3)), gross=net, cost=list(np.zeros(3)), net=net)
    metrics = compute_metrics(result)
    assert metrics.sharpe_annualized == 0.0


def test_metrics_to_dict_has_expected_keys() -> None:
    result = _result(size=[1.0], gross=[1.0], cost=[0.1], net=[0.9])
    metrics = compute_metrics(result)
    assert set(metrics.to_dict().keys()) == {
        "gross_pnl_eur",
        "net_pnl_eur",
        "total_cost_eur",
        "pnl_per_mwh_eur",
        "traded_volume_mwh",
        "n_trades",
        "hit_rate",
        "sharpe_annualized",
        "max_drawdown_eur",
        "turnover_mwh",
    }


# ---------------------------------------------------------------------------
# block_bootstrap_ci
# ---------------------------------------------------------------------------


def test_block_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    net_pnl = pd.Series(rng.normal(loc=0.5, scale=2.0, size=n))

    ci = block_bootstrap_ci(net_pnl, block_length_hours=24, n_resamples=500, confidence_level=0.9)

    lower, upper = ci["net_pnl_eur"]
    assert lower < net_pnl.sum() < upper
    sharpe_lower, sharpe_upper = ci["sharpe_annualized"]
    assert sharpe_lower < sharpe_upper


def test_block_bootstrap_ci_is_deterministic_given_seed() -> None:
    net_pnl = pd.Series(np.random.default_rng(2).normal(size=500))
    ci_a = block_bootstrap_ci(
        net_pnl, block_length_hours=10, n_resamples=200, confidence_level=0.95, seed=7
    )
    ci_b = block_bootstrap_ci(
        net_pnl, block_length_hours=10, n_resamples=200, confidence_level=0.95, seed=7
    )
    assert ci_a == ci_b


def test_block_bootstrap_ci_handles_series_shorter_than_block() -> None:
    net_pnl = pd.Series([1.0, 2.0, 3.0])
    ci = block_bootstrap_ci(net_pnl, block_length_hours=100, n_resamples=50, confidence_level=0.9)
    assert "net_pnl_eur" in ci


def test_block_bootstrap_ci_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="block_length_hours"):
        block_bootstrap_ci(
            pd.Series([1.0, 2.0]), block_length_hours=0, n_resamples=10, confidence_level=0.9
        )
    with pytest.raises(ValueError, match="non-empty"):
        block_bootstrap_ci(
            pd.Series([], dtype=float), block_length_hours=5, n_resamples=10, confidence_level=0.9
        )
