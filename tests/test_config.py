"""Tests for configuration loading (Phase 0 scaffolding)."""

import datetime as dt

from fr_power.config import DEFAULT_CONFIG_PATH, Config, load_config


def test_default_config_loads_and_validates() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.market.bidding_zone == "FR"


def test_resolved_end_date_applies_safety_margin_when_today_configured() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.data.end_date == "today"
    expected = dt.datetime.now(dt.UTC).date() - dt.timedelta(
        days=cfg.data.end_date_safety_margin_days
    )
    assert cfg.data.resolved_end_date() == expected


def test_resolved_end_date_respects_explicit_date_exactly() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"end_date": "2024-06-15"})})
    assert cfg.data.resolved_end_date() == dt.date(2024, 6, 15)


def test_weather_city_weights_sum_to_one() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    total = sum(c.weight for c in cfg.data.weather.cities)
    assert 0.99 <= total <= 1.01


def test_costs_are_non_negative() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.costs.fee_eur_per_mwh >= 0
    assert cfg.costs.half_spread_eur_per_mwh >= 0
