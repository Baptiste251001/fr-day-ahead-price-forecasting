"""Typed configuration for the fr_power pipeline, loaded from YAML.

All pipeline stages take a `Config` instance rather than reading files or
environment variables directly, so a run is fully reproducible from a single
YAML file plus a `.env`.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class ProjectConfig(BaseModel):
    name: str
    random_seed: int
    timezone: str


class MarketConfig(BaseModel):
    bidding_zone: str
    eic_code: str
    currency: str


class EntsoeSeriesConfig(BaseModel):
    day_ahead_prices: bool
    day_ahead_wind_solar_forecast: bool
    day_ahead_load_forecast: bool
    actual_generation_by_type: bool
    cross_border_flows: bool


class IntradaySourceConfig(BaseModel):
    source: Literal["csv", "stub"]
    path: str


class WeatherCity(BaseModel):
    name: str
    lat: float
    lon: float
    weight: float


class WeatherConfig(BaseModel):
    enabled: bool
    provider: str
    cities: list[WeatherCity]
    hdd_base_celsius: float
    cdd_base_celsius: float

    @field_validator("cities")
    @classmethod
    def weights_sum_to_one(cls, cities: list[WeatherCity]) -> list[WeatherCity]:
        total = sum(c.weight for c in cities)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"City weights must sum to ~1.0, got {total}")
        return cities


class RteEco2mixConfig(BaseModel):
    enabled: bool


class DataConfig(BaseModel):
    start_date: dt.date
    end_date: str  # "today" resolved lazily; see `resolved_end_date`
    raw_dir: str
    interim_dir: str
    processed_dir: str
    cache_format: Literal["parquet"]
    entsoe: EntsoeSeriesConfig
    intraday: IntradaySourceConfig
    weather: WeatherConfig
    rte_eco2mix: RteEco2mixConfig

    def resolved_end_date(self) -> dt.date:
        if self.end_date == "today":
            return dt.datetime.now(dt.UTC).date()
        return dt.date.fromisoformat(self.end_date)


class CostsConfig(BaseModel):
    fee_eur_per_mwh: float = Field(ge=0)
    half_spread_eur_per_mwh: float = Field(ge=0)


class WalkForwardConfig(BaseModel):
    initial_train_days: int = Field(gt=0)
    retrain_every_days: int = Field(gt=0)
    test_window_days: int = Field(gt=0)


class BlockBootstrapConfig(BaseModel):
    block_length_days: int = Field(gt=0)
    n_resamples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)


class BacktestConfig(BaseModel):
    walk_forward: WalkForwardConfig
    no_trade_band_eur_per_mwh: float = Field(ge=0)
    max_position_mwh: float = Field(gt=0)
    block_bootstrap: BlockBootstrapConfig
    cost_sensitivity_multipliers: list[float]


class NaiveModelConfig(BaseModel):
    lag_hours: int = Field(gt=0)


class LearModelConfig(BaseModel):
    lags_days: list[int]
    lasso_alpha_grid: list[float]
    cv_folds: int = Field(gt=1)


class LightgbmModelConfig(BaseModel):
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    early_stopping_rounds: int


class LstmModelConfig(BaseModel):
    sequence_length_hours: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    quantiles: list[float]


class ModelsConfig(BaseModel):
    naive: NaiveModelConfig
    lear: LearModelConfig
    lightgbm: LightgbmModelConfig
    lstm: LstmModelConfig


class FeaturesConfig(BaseModel):
    price_lags_hours: list[int]
    price_rolling_windows_hours: list[int]
    spread_lags_hours: list[int]
    generation_availability_lag_hours: int = Field(gt=0)
    weather_availability_lag_hours: int = Field(gt=0)


class OuEstimationConfig(BaseModel):
    method: Literal["ols", "mle"]
    min_observations: int = Field(gt=1)


class OuOptimalStoppingConfig(BaseModel):
    entry_z_score: float = Field(gt=0)
    exit_z_score: float = Field(ge=0)
    no_trade_z_score: float = Field(ge=0)
    max_holding_hours: int = Field(gt=0)

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> OuOptimalStoppingConfig:
        if not (self.no_trade_z_score <= self.exit_z_score <= self.entry_z_score):
            raise ValueError(
                "OU optimal-stopping thresholds must satisfy "
                "no_trade_z_score <= exit_z_score <= entry_z_score, got "
                f"{self.no_trade_z_score} <= {self.exit_z_score} <= {self.entry_z_score}"
            )
        return self


class OuConfig(BaseModel):
    estimation: OuEstimationConfig
    optimal_stopping: OuOptimalStoppingConfig


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseModel):
    """Root configuration object for a pipeline run."""

    project: ProjectConfig
    market: MarketConfig
    data: DataConfig
    costs: CostsConfig
    backtest: BacktestConfig
    models: ModelsConfig
    features: FeaturesConfig
    ou: OuConfig
    logging: LoggingConfig


class Secrets(BaseModel):
    """API credentials, loaded from environment variables (via `.env`)."""

    entsoe_api_token: str | None = None
    rte_client_id: str | None = None
    rte_client_secret: str | None = None
    weather_api_key: str | None = None

    @classmethod
    def from_env(cls) -> Secrets:
        return cls(
            entsoe_api_token=os.environ.get("ENTSOE_API_TOKEN") or None,
            rte_client_id=os.environ.get("RTE_CLIENT_ID") or None,
            rte_client_secret=os.environ.get("RTE_CLIENT_SECRET") or None,
            weather_api_key=os.environ.get("WEATHER_API_KEY") or None,
        )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML config file into a `Config` object."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def load_secrets(env_file: str | Path | None = None) -> Secrets:
    """Load API credentials from a `.env` file (if present) and the environment."""
    load_dotenv(dotenv_path=env_file, override=False)
    return Secrets.from_env()
