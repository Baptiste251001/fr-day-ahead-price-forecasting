"""Shared fixtures: a small synthetic raw dataset shaped like the real ENTSO-E pulls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fr_power import data
from fr_power.config import Config


def make_raw(days: int = 60, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Hourly synthetic raw data; the last third of the period is at 15-minute resolution
    for load and generation, as in the real ENTSO-E feed after the 15-minute MTU switch."""
    rng = np.random.default_rng(seed)
    hourly = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz=data.TIMEZONE)
    n = len(hourly)
    hour = hourly.hour.to_numpy()
    load = 50_000 + 8_000 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 500, n)
    wind = 5_000 + rng.normal(0, 1_000, n)
    solar = np.clip(4_000 * np.sin((hour - 6) / 12 * np.pi), 0, None)
    price = 20 + (load - wind - solar) / 1_000 + rng.normal(0, 3, n)

    quarter = pd.date_range(hourly[2 * n // 3], hourly[-1], freq="15min")
    load_df = pd.DataFrame({"Forecasted Load": load}, index=hourly)
    load_df = pd.concat(
        [load_df.iloc[: 2 * n // 3], load_df.reindex(quarter, method="ffill")]
    )  # mixed hourly / 15-minute resolution
    generation = pd.DataFrame(
        {"Nuclear": 40_000 + rng.normal(0, 1_000, n), "Wind Offshore": np.nan}, index=hourly
    )
    generation.iloc[n // 2 :, 1] = 500.0  # a technology that starts reporting halfway

    days_index = pd.date_range("2024-01-01", periods=days, freq="D")
    temperature = 8 + rng.normal(0, 3, days)
    return {
        data.DA_PRICES: pd.DataFrame({"day_ahead_price_eur_mwh": price}, index=hourly),
        data.LOAD_FORECAST: load_df,
        data.WIND_SOLAR_FORECAST: pd.DataFrame(
            {"Solar": solar, "Wind Onshore": wind}, index=hourly
        ),
        data.GENERATION: generation,
        data.WEATHER: pd.DataFrame(
            {"temperature_c": temperature, "hdd": 15.5 - temperature, "cdd": 0.0},
            index=days_index,
        ),
    }


@pytest.fixture
def raw() -> dict[str, pd.DataFrame]:
    return make_raw()


@pytest.fixture
def config(tmp_path) -> Config:  # type: ignore[no-untyped-def]
    return Config.model_validate(
        {
            "reports_dir": str(tmp_path / "reports"),
            "data": {
                "start_date": "2024-01-01",
                "end_date": "2024-02-29",
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "weather": {
                    "cities": [{"name": "Paris", "lat": 48.9, "lon": 2.4, "weight": 1.0}],
                    "hdd_base_celsius": 15.5,
                    "cdd_base_celsius": 18.0,
                },
            },
            "walk_forward": {"initial_train_days": 30, "retrain_every_days": 10},
            "models": {
                "lasso": {"alpha_grid": [0.01, 0.1, 1.0], "cv_folds": 3},
                "lightgbm": {
                    "n_estimators": 50,
                    "learning_rate": 0.1,
                    "num_leaves": 7,
                    "min_child_samples": 10,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "early_stopping_rounds": 10,
                },
            },
        }
    )
