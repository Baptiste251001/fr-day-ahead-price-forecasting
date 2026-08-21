"""Tests for the synthetic demo-data generator: shapes, plausibility, and
reproducibility. Not a test of "real" French market statistics -- there
are none here, by design (see module docstring)."""

import numpy as np
import pandas as pd

from fr_power.demo.synthetic import generate_synthetic_market_data
from fr_power.ou.estimate import estimate_ou


def _small_dataset(
    start: str = "2020-01-01", end: str = "2020-04-01", seed: int = 0
) -> dict[str, pd.DataFrame]:
    return generate_synthetic_market_data(
        start=start, end=end, seed=seed, crisis_start="2020-02-15", crisis_end="2020-03-01"
    )


def test_generate_returns_expected_keys_and_shapes() -> None:
    data = _small_dataset()
    expected_keys = {
        "day_ahead_prices",
        "day_ahead_wind_solar_forecast",
        "day_ahead_load_forecast",
        "actual_generation_by_type",
        "cross_border_flows",
        "weather",
        "intraday",
    }
    assert set(data.keys()) == expected_keys

    n_hours = len(data["day_ahead_prices"])
    for key in expected_keys - {"weather"}:
        assert len(data[key]) == n_hours
        assert data[key].index.tz is not None

    assert data["weather"].index.tz is None  # daily, as produced by fr_power.data.weather


def test_generation_series_are_non_negative() -> None:
    data = _small_dataset()
    assert (data["day_ahead_wind_solar_forecast"]["Wind Onshore"] >= 0).all()
    assert (data["day_ahead_wind_solar_forecast"]["Solar"] >= 0).all()
    assert (data["actual_generation_by_type"]["Nuclear"] > 0).all()


def test_weather_daily_index_covers_full_hourly_span_without_gaps() -> None:
    data = _small_dataset()
    hourly_dates = data["day_ahead_prices"].index.tz_localize(None).normalize()
    reindexed = data["weather"]["hdd"].reindex(hourly_dates)
    assert reindexed.notna().all()


def test_crisis_regime_shifts_the_spread_mean() -> None:
    data = _small_dataset(start="2020-01-01", end="2020-04-01")
    spread = (
        data["intraday"]["intraday_price_eur_mwh"]
        - data["day_ahead_prices"]["day_ahead_price_eur_mwh"]
    )
    index = spread.index
    is_crisis = (index >= pd.Timestamp("2020-02-15", tz="Europe/Paris")) & (
        index <= pd.Timestamp("2020-03-01", tz="Europe/Paris")
    )
    assert spread[is_crisis].mean() > spread[~is_crisis].mean() + 1.0


def test_generator_is_reproducible_given_seed() -> None:
    a = _small_dataset(seed=7)
    b = _small_dataset(seed=7)
    pd.testing.assert_frame_equal(a["day_ahead_prices"], b["day_ahead_prices"])
    pd.testing.assert_frame_equal(a["intraday"], b["intraday"])


def test_generator_seeds_produce_different_output() -> None:
    a = _small_dataset(seed=1)
    b = _small_dataset(seed=2)
    assert not a["day_ahead_prices"]["day_ahead_price_eur_mwh"].equals(
        b["day_ahead_prices"]["day_ahead_price_eur_mwh"]
    )


def test_synthetic_spread_is_genuinely_ou_like() -> None:
    """Cross-checks the generator against our own OU estimator: a long, single-regime
    (no crisis) synthetic spread series should calibrate to a positive, finite half-life."""
    data = generate_synthetic_market_data(
        start="2020-01-01",
        end="2020-07-01",
        seed=3,
        crisis_start="2099-01-01",
        crisis_end="2099-01-02",
    )
    spread = (
        data["intraday"]["intraday_price_eur_mwh"]
        - data["day_ahead_prices"]["day_ahead_price_eur_mwh"]
    )

    params = estimate_ou(spread, method="ols")

    assert params.kappa > 0
    assert np.isfinite(params.half_life)
    assert params.half_life > 0
