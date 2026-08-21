"""Tests for calendar feature construction."""

import pandas as pd
import pytest

from fr_power.features.calendar import add_calendar_features


def test_add_calendar_features_basic_columns() -> None:
    index = pd.date_range("2024-06-03", periods=48, freq="h", tz="Europe/Paris")  # a Monday
    df = add_calendar_features(index)

    assert list(df.columns) == [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_french_holiday",
        "is_dst_transition_day",
    ]
    assert df["hour"].tolist()[:3] == [0, 1, 2]
    assert df["day_of_week"].iloc[0] == 0  # Monday
    assert df["month"].iloc[0] == 6


def test_weekend_flag() -> None:
    # 2024-06-08 is a Saturday, 2024-06-10 is a Monday
    index = pd.to_datetime(["2024-06-08 12:00", "2024-06-10 12:00"]).tz_localize("Europe/Paris")
    df = add_calendar_features(index)
    assert df["is_weekend"].tolist() == [True, False]


def test_french_holiday_flag_bastille_day() -> None:
    index = pd.to_datetime(["2024-07-14 10:00", "2024-07-15 10:00"]).tz_localize("Europe/Paris")
    df = add_calendar_features(index)
    assert df["is_french_holiday"].tolist() == [True, False]


def test_dst_transition_day_flagged() -> None:
    # 2024-03-31: spring-forward in France, 23 local hours instead of 24.
    normal_day = pd.date_range("2024-03-01", periods=24, freq="h", tz="Europe/Paris")
    transition_day = pd.date_range("2024-03-31", periods=23, freq="h", tz="Europe/Paris")
    index = normal_day.append(transition_day)

    df = add_calendar_features(index)

    assert not df.loc[
        df.index.normalize() == pd.Timestamp("2024-03-01", tz="Europe/Paris"),
        "is_dst_transition_day",
    ].any()
    assert df.loc[
        df.index.normalize() == pd.Timestamp("2024-03-31", tz="Europe/Paris"),
        "is_dst_transition_day",
    ].all()


def test_raises_on_naive_index() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    with pytest.raises(ValueError, match="tz-aware"):
        add_calendar_features(index)
