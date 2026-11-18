"""Typed configuration, loaded from a single YAML file (`configs/default.yaml`)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class City(BaseModel):
    name: str
    lat: float
    lon: float
    weight: float


class WeatherConfig(BaseModel):
    cities: list[City]
    hdd_base_celsius: float
    cdd_base_celsius: float

    @field_validator("cities")
    @classmethod
    def weights_sum_to_one(cls, cities: list[City]) -> list[City]:
        total = sum(c.weight for c in cities)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"City weights must sum to ~1.0, got {total}")
        return cities


class DataConfig(BaseModel):
    start_date: dt.date
    # ISO date, or "today" (resolved to today minus `end_date_margin_days`: ENTSO-E 400s
    # on a range past the last cleared auction and Open-Meteo's archive lags ~2 days).
    end_date: str = "today"
    end_date_margin_days: int = Field(default=2, ge=0)
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    weather: WeatherConfig
    gas_ticker: str = "TTF=F"  # Yahoo Finance: Dutch TTF natural gas, front month, EUR/MWh

    def resolved_end_date(self) -> dt.date:
        if self.end_date == "today":
            return dt.date.today() - dt.timedelta(days=self.end_date_margin_days)
        return dt.date.fromisoformat(self.end_date)


class FeaturesConfig(BaseModel):
    price_lags_hours: list[int] = [24, 48, 168]
    price_rolling_windows_hours: list[int] = [24, 168]
    realized_lag_hours: int = Field(default=48, ge=48)
    gas_plant_efficiency: float = Field(default=0.52, gt=0, le=1)

    @field_validator("price_lags_hours", "price_rolling_windows_hours")
    @classmethod
    def at_least_one_day(cls, values: list[int]) -> list[int]:
        # Prices of delivery day D all come out of the same auction, so nothing from day D
        # itself may be used: lags must reach back at least 24h.
        if any(v < 24 for v in values):
            raise ValueError(f"Price lags/windows must be >= 24h, got {values}")
        return values


class WalkForwardConfig(BaseModel):
    initial_train_days: int = Field(gt=0)
    retrain_every_days: int = Field(gt=0)


class LassoConfig(BaseModel):
    alpha_grid: list[float]
    cv_folds: int = Field(gt=1)


class LightGBMConfig(BaseModel):
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    early_stopping_rounds: int


class ExtraTreesConfig(BaseModel):
    n_estimators: int
    min_samples_leaf: int
    max_features: float = Field(gt=0, le=1)


class ModelsConfig(BaseModel):
    lasso: LassoConfig
    lightgbm: LightGBMConfig
    extra_trees: ExtraTreesConfig
    ensemble: list[str] = ["lasso", "lightgbm", "extra_trees"]


class Config(BaseModel):
    random_seed: int = 42
    reports_dir: Path = Path("reports")
    data: DataConfig
    features: FeaturesConfig = FeaturesConfig()
    walk_forward: WalkForwardConfig
    models: ModelsConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with Path(path).open(encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
