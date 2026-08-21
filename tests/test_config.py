"""Tests for configuration loading (Phase 0 scaffolding)."""

import datetime as dt

from fr_power.config import DEFAULT_CONFIG_PATH, Config, load_config


def test_default_config_loads_and_validates() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.market.bidding_zone == "FR"


def test_resolved_end_date_defaults_to_today_when_configured() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.data.end_date == "today"
    assert cfg.data.resolved_end_date() == dt.datetime.now(dt.UTC).date()


def test_weather_city_weights_sum_to_one() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    total = sum(c.weight for c in cfg.data.weather.cities)
    assert 0.99 <= total <= 1.01


def test_costs_are_non_negative() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.costs.fee_eur_per_mwh >= 0
    assert cfg.costs.half_spread_eur_per_mwh >= 0
