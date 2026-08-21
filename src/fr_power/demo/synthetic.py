"""Synthetic, illustrative FR-like market data.

**Not real market data.** This generator exists solely so the pipeline
and the exploration notebooks under `notebooks/` have something to run
end-to-end against without a live ENTSO-E API token -- see
`reports/conclusions.md` for the actual, honest discussion of the real
French power market (drawn from public domain knowledge, not this
generator) and for why no real quantitative results are reported without
one.

The generator is deliberately hand-designed with plausible *structural*
features of the French market -- nuclear baseload with periodic
maintenance dips and an illustrative crisis-period shortfall, solar/wind
seasonality, thermosensitive load, occasional negative price hours, and a
genuinely Ornstein-Uhlenbeck-distributed intraday-vs-day-ahead spread
with a regime shift -- so that every downstream stage (feature
engineering, forecasting, OU calibration, backtesting) has something
structurally reasonable, and not just meaningless, to work with. None of
the specific numbers it produces should be read as a claim about real
French prices, generation, or demand.

Output shapes exactly match what `fr_power.data.entsoe.fetch_all` /
`fr_power.data.weather.fetch_and_cache_weather` /
`fr_power.data.intraday.CsvIntradaySource` would produce, so the rest of
the pipeline (`fr_power.features.build.build_feature_matrix` onward) runs
completely unmodified against this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _synthetic_weather(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    # tz-naive daily index, matching fr_power.data.weather's real output shape (Open-Meteo
    # returns plain "YYYY-MM-DD" dates with no tz attached).
    start_date = index[0].replace(tzinfo=None).normalize()
    end_date = index[-1].replace(tzinfo=None).normalize()
    daily_index = pd.date_range(start_date, end_date, freq="D")
    doy = daily_index.dayofyear.to_numpy()
    seasonal_temp = 11.0 - 9.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    temp = seasonal_temp + rng.normal(scale=2.5, size=len(daily_index))
    temp = pd.Series(temp, index=daily_index).rolling(5, min_periods=1).mean().to_numpy()
    hdd = np.clip(15.5 - temp, 0, None)
    cdd = np.clip(temp - 18.0, 0, None)
    return pd.DataFrame({"temperature_c": temp, "hdd": hdd, "cdd": cdd}, index=daily_index)


def _synthetic_generation(
    index: pd.DatetimeIndex, is_crisis: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (wind_mw, solar_mw, nuclear_mw)."""
    n = len(index)
    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()

    wind_base = 6000.0 + 3000.0 * (-np.cos(2 * np.pi * (doy - 15) / 365.25))
    wind_noise = (
        pd.Series(rng.normal(scale=1500.0, size=n), index=index).ewm(span=12).mean().to_numpy()
    )
    wind = np.clip(wind_base + wind_noise, 0, None)

    solar_elevation = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    solar_amplitude = 2000.0 + 6000.0 * np.clip(np.sin(2 * np.pi * (doy - 80) / 365.25), 0, None)
    solar = np.clip(solar_amplitude * solar_elevation + rng.normal(scale=200.0, size=n), 0, None)

    nuclear = np.full(n, 54000.0)
    n_days = (index[-1] - index[0]).days + 1
    n_maintenance_events = max(1, n_days // 25)
    maintenance_start_days = rng.integers(low=0, high=n_days, size=n_maintenance_events)
    for start_day in maintenance_start_days:
        dip_len_hours = int(rng.integers(7 * 24, 35 * 24))
        dip_amount = rng.uniform(3000.0, 9000.0)
        dip_start = start_day * 24
        dip_end = min(n, dip_start + dip_len_hours)
        if dip_start < n:
            nuclear[dip_start:dip_end] -= dip_amount
    nuclear = np.where(is_crisis, nuclear - rng.uniform(6000.0, 10000.0, size=n), nuclear)
    nuclear = np.clip(nuclear + rng.normal(scale=400.0, size=n), 20000.0, None)

    return wind, solar, nuclear


def _synthetic_load(
    index: pd.DatetimeIndex,
    hdd_hourly: np.ndarray,
    cdd_hourly: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    hour = index.hour.to_numpy()
    is_weekend = index.dayofweek.to_numpy() >= 5
    hour_shape = 0.85 + 0.30 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2
    weekend_factor = np.where(is_weekend, 0.90, 1.0)
    load = (48000.0 + 1400.0 * hdd_hourly + 250.0 * cdd_hourly) * hour_shape * weekend_factor
    return load + rng.normal(scale=800.0, size=len(index))


def _synthetic_price_and_spread(
    index: pd.DatetimeIndex,
    residual_load: np.ndarray,
    nuclear: np.ndarray,
    solar: np.ndarray,
    solar_amplitude_mean: float,
    is_crisis: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (day_ahead_price, intraday_price)."""
    n = len(index)
    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()

    base_price = 15.0 + 0.0016 * residual_load + 0.0009 * (54000.0 - nuclear)
    seasonal_price = 5.0 * np.sin(2 * np.pi * (doy - 15) / 365.25 + np.pi)
    price = base_price + seasonal_price
    price = np.where(is_crisis, price * 3.2 + rng.normal(scale=40.0, size=n), price)
    price += rng.normal(scale=6.0, size=n)

    negative_mask = (
        (solar > solar_amplitude_mean * 0.7)
        & (hour >= 11)
        & (hour <= 15)
        & (~is_crisis)
        & (rng.random(n) < 0.15)
    )
    price = np.where(negative_mask, rng.uniform(-40.0, -1.0, size=n), price)

    # A genuinely OU-distributed intraday-vs-day-ahead spread, with a regime shift in its
    # long-run mean during the crisis window, plus a weak dependence on the residual-load
    # anomaly so forecasting models have a real, if modest, learnable signal beyond pure
    # autocorrelation.
    kappa, sigma = 0.06, 3.0
    mu_normal, mu_crisis = 0.5, 4.0
    mu_t = np.where(is_crisis, mu_crisis, mu_normal)
    residual_load_anomaly = (residual_load - residual_load.mean()) / (residual_load.std() + 1e-9)
    a = np.exp(-kappa)
    innovation_std = sigma * np.sqrt((1 - a**2) / (2 * kappa))
    spread = np.empty(n)
    spread[0] = mu_t[0]
    for t in range(1, n):
        spread[t] = (
            mu_t[t]
            + a * (spread[t - 1] - mu_t[t])
            + innovation_std * rng.standard_normal()
            + 0.4 * residual_load_anomaly[t]
        )

    return price, price + spread


def generate_synthetic_market_data(
    start: str = "2019-01-01",
    end: str = "2022-01-01",
    seed: int = 42,
    crisis_start: str = "2021-06-01",
    crisis_end: str = "2021-12-31",
) -> dict[str, pd.DataFrame]:
    """Generate a synthetic, illustrative FR-like dataset spanning `start` (inclusive) to
    `end` (exclusive), with an illustrative "crisis" regime shift from `crisis_start` to
    `crisis_end`. See module docstring: not real market data.

    Returns a dict with the same keys/shapes as `fr_power.data.entsoe.fetch_all`'s result,
    plus "weather" (daily) and "intraday" (hourly, `intraday_price_eur_mwh`).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="h", tz="Europe/Paris", inclusive="left")
    is_crisis = (index >= pd.Timestamp(crisis_start, tz="Europe/Paris")) & (
        index <= pd.Timestamp(crisis_end, tz="Europe/Paris")
    )

    weather_daily = _synthetic_weather(index, rng)
    hourly_dates = index.tz_localize(None).normalize()
    hdd_hourly = weather_daily["hdd"].reindex(hourly_dates).to_numpy()
    cdd_hourly = weather_daily["cdd"].reindex(hourly_dates).to_numpy()

    wind, solar, nuclear = _synthetic_generation(index, is_crisis, rng)
    load = _synthetic_load(index, hdd_hourly, cdd_hourly, rng)
    residual_load = load - wind - solar

    solar_amplitude_mean = float(
        (
            2000.0
            + 6000.0
            * np.clip(np.sin(2 * np.pi * (index.dayofyear.to_numpy() - 80) / 365.25), 0, None)
        ).mean()
    )
    day_ahead_price, intraday_price = _synthetic_price_and_spread(
        index, residual_load, nuclear, solar, solar_amplitude_mean, is_crisis, rng
    )

    net_position = rng.normal(
        loc=np.where(is_crisis, 2000.0, -3000.0), scale=2500.0, size=len(index)
    )

    return {
        "day_ahead_prices": pd.DataFrame({"day_ahead_price_eur_mwh": day_ahead_price}, index=index),
        "day_ahead_wind_solar_forecast": pd.DataFrame(
            {"Wind Onshore": wind, "Solar": solar}, index=index
        ),
        "day_ahead_load_forecast": pd.DataFrame({"Forecasted Load": load}, index=index),
        "actual_generation_by_type": pd.DataFrame(
            {
                "Nuclear": nuclear,
                "Hydro Run-of-river and poundage": 5000.0
                + rng.normal(scale=500.0, size=len(index)),
            },
            index=index,
        ),
        "cross_border_flows": pd.DataFrame({"net_position_mw": net_position}, index=index),
        "weather": weather_daily,
        "intraday": pd.DataFrame({"intraday_price_eur_mwh": intraday_price}, index=index),
    }
