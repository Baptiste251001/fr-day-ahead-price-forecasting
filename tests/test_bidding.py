"""Battery bidding: optimal schedules, bid curves, settlement and scenario timing."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from fr_power.bidding import (
    Battery,
    BidCurves,
    backtest,
    error_table,
    optimal_curves,
    scenarios_for_day,
    settle,
    summarise,
)

BATTERY = Battery(power_mw=1.0, energy_mwh=2.0, round_trip_efficiency=0.81)  # 0.9 each way


def _day_prices() -> np.ndarray:
    prices = np.full(24, 50.0)
    prices[[3, 4]] = [10.0, 12.0]
    prices[[19, 20]] = [100.0, 90.0]
    return prices


def test_perfect_foresight_matches_hand_computed_optimum() -> None:
    prices = _day_prices()
    result = settle(optimal_curves(prices[None, :], BATTERY), prices, BATTERY, 50.0)
    # Charge 1 MW at 03:00 and 04:00 and 0.22 MW in a 50 EUR hour to fill the 2 MWh, then
    # sell 1 MW at 19:00 and the remaining 0.8 MW at 20:00.
    expected = -10 - 12 - 50 * (0.2 / 0.9) + 100 + 90 * 0.8
    assert result["profit_eur"] == pytest.approx(expected, abs=1e-3)
    assert result["imbalance_mwh"] == pytest.approx(0, abs=1e-5)


def test_optimal_schedule_is_always_deliverable_even_with_negative_prices() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        prices = rng.normal(20, 40, 24)  # many negative hours: the optimiser burns energy
        result = settle(optimal_curves(prices[None, :], BATTERY), prices, BATTERY, 50.0)
        assert result["imbalance_mwh"] == pytest.approx(0, abs=1e-4)


def test_curves_are_monotone_and_reduce_to_schedule_without_uncertainty() -> None:
    prices = _day_prices()
    rng = np.random.default_rng(0)
    curves = optimal_curves(prices + rng.normal(0, 15, (30, 24)), BATTERY)
    assert (np.diff(curves.quantities, axis=1) >= -1e-9).all()
    assert (np.diff(curves.prices, axis=1) >= 0).all()
    assert np.abs(curves.quantities).max() <= BATTERY.power_mw + 1e-9

    certain = optimal_curves(np.repeat(prices[None, :], 5, axis=0), BATTERY)
    perfect = optimal_curves(prices[None, :], BATTERY)
    assert settle(certain, prices, BATTERY, 50.0)["profit_eur"] == pytest.approx(
        settle(perfect, prices, BATTERY, 50.0)["profit_eur"], abs=1e-3
    )


def test_settlement_cannot_sell_energy_the_battery_does_not_hold() -> None:
    curves = BidCurves(prices=np.zeros((3, 1)), quantities=np.ones((3, 1)))  # always sell 1 MW
    prices = np.array([40.0, 50.0, 60.0])
    result = settle(curves, prices, BATTERY, imbalance_penalty=50.0)
    assert result["discharged_mwh"] == 0  # the battery starts the day empty
    assert result["imbalance_mwh"] == pytest.approx(3.0)
    assert result["profit_eur"] == pytest.approx(-150.0)


def test_scenarios_only_use_errors_known_before_the_delivery_day() -> None:
    index = pd.date_range("2024-03-01", periods=20 * 24, freq="h", tz="Europe/Paris")
    rng = np.random.default_rng(1)
    actual = pd.Series(50 + rng.normal(0, 10, len(index)), index=index)
    forecast = pd.Series(50.0, index=index)
    day = dt.date(2024, 3, 15)
    day_forecast = forecast[forecast.index.date == day]

    before = scenarios_for_day(day_forecast, error_table(actual, forecast), day, 10, 5)
    perturbed = actual.copy()
    perturbed[perturbed.index.date >= day] += 1_000  # prices of day D and later
    after = scenarios_for_day(day_forecast, error_table(perturbed, forecast), day, 10, 5)
    assert before is not None and after is not None
    assert before.shape == (10, 24)
    np.testing.assert_array_equal(before, after)
    assert scenarios_for_day(day_forecast, error_table(actual, forecast), day, 10, 30) is None


def test_backtest_ranks_perfect_foresight_first() -> None:
    index = pd.date_range("2024-03-01", periods=40 * 24, freq="h", tz="Europe/Paris")
    rng = np.random.default_rng(2)
    shape = 50 + 30 * np.sin((index.hour.to_numpy() - 13) / 24 * 2 * np.pi)
    actual = pd.Series(shape + rng.normal(0, 10, len(index)), index=index)
    predictions = pd.DataFrame({"good": shape, "naive": actual.shift(24).to_numpy()}, index=index)
    daily = backtest(actual, predictions, BATTERY, ["naive", "good"], "good", 10, 5, 50.0)
    summary = summarise(daily, BATTERY)
    assert summary.loc["perfect_foresight", "capture_vs_perfect"] == pytest.approx(1.0)
    assert (summary["capture_vs_perfect"] < 1.0 + 1e-9).all()
    assert summary.loc["schedule_good", "capture_vs_perfect"] > 0.5
    assert summary.loc["schedule_good", "imbalance_mwh_per_day"] == pytest.approx(0, abs=1e-4)
