"""Feature matrix: above all, no information from after the D-1 decision time may leak in."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fr_power import data
from fr_power.config import FeaturesConfig
from fr_power.features import TARGET, build_feature_matrix, feature_columns, fill_leading_gaps

CONFIG = FeaturesConfig()
DAY = pd.Timestamp("2024-02-10", tz=data.TIMEZONE)  # an arbitrary delivery day D


def _day_rows(matrix: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return matrix.loc[day : day + pd.Timedelta(hours=23), feature_columns(matrix)]


def _perturb(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:  # daily weather, indexed by naive local dates
        start, end = start.tz_localize(None), end.tz_localize(None)
    out.loc[(idx >= start) & (idx < end)] += 1_000.0
    return out


def test_features_ignore_prices_of_the_delivery_day(raw: dict[str, pd.DataFrame]) -> None:
    before = build_feature_matrix(raw, CONFIG)
    raw[data.DA_PRICES] = _perturb(raw[data.DA_PRICES], DAY, DAY + pd.Timedelta(days=1))
    after = build_feature_matrix(raw, CONFIG)

    pd.testing.assert_frame_equal(_day_rows(before, DAY), _day_rows(after, DAY))
    assert (after.loc[DAY, TARGET] - before.loc[DAY, TARGET]) == pytest.approx(1_000.0)


@pytest.mark.parametrize("dataset", [data.GENERATION, data.WEATHER])
def test_realized_data_from_d_minus_1_is_not_used(
    raw: dict[str, pd.DataFrame], dataset: str
) -> None:
    """Generation and weather of day D-1 are not complete at decision time (D-1 morning)."""
    before = build_feature_matrix(raw, CONFIG)
    raw[dataset] = _perturb(raw[dataset], DAY - pd.Timedelta(days=1), DAY + pd.Timedelta(days=1))
    after = build_feature_matrix(raw, CONFIG)

    pd.testing.assert_frame_equal(_day_rows(before, DAY), _day_rows(after, DAY))


def test_lag_and_rolling_values(raw: dict[str, pd.DataFrame]) -> None:
    matrix = build_feature_matrix(raw, CONFIG)
    price = raw[data.DA_PRICES]["day_ahead_price_eur_mwh"]
    t = DAY + pd.Timedelta(hours=15)
    h = pd.Timedelta(hours=1)

    assert matrix.loc[t, "price_lag_24h"] == price[t - 24 * h]
    assert matrix.loc[t, "price_lag_168h"] == price[t - 168 * h]
    assert matrix.loc[t, "price_mean_24h"] == pytest.approx(price[t - 47 * h : t - 24 * h].mean())
    gen = raw[data.GENERATION]["Nuclear"]
    assert matrix.loc[t, "gen_nuclear_lag48h"] == gen[t - 48 * h]


def test_mixed_resolution_inputs_are_averaged_to_hourly(raw: dict[str, pd.DataFrame]) -> None:
    matrix = build_feature_matrix(raw, CONFIG)
    assert matrix.index.freq == "h" or (np.diff(matrix.index.asi8) == 3_600 * 10**9).all()
    last_hour = matrix.index[-2]
    quarter_hours = raw[data.LOAD_FORECAST].loc[last_hour : last_hour + pd.Timedelta(minutes=45)]
    assert len(quarter_hours) == 4
    assert matrix.loc[last_hour, "load_forecast_mw"] == pytest.approx(
        quarter_hours.iloc[:, 0].mean()
    )


def test_fill_leading_gaps_only_fills_before_first_report() -> None:
    df = pd.DataFrame({"x": [np.nan, np.nan, 5.0, np.nan, 7.0], "never": [np.nan] * 5})
    out = fill_leading_gaps(df)
    assert out["x"].tolist()[:3] == [0.0, 0.0, 5.0]
    assert np.isnan(out["x"].iloc[3])  # genuine reporting gap is kept as NaN
    assert out["never"].isna().all()


def test_lags_shorter_than_a_day_are_rejected() -> None:
    with pytest.raises(ValidationError):
        FeaturesConfig(price_lags_hours=[1, 24])
    with pytest.raises(ValidationError):
        FeaturesConfig(realized_lag_hours=24)
