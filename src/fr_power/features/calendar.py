"""Calendar features for the FR hourly delivery grid.

All timestamps flowing through this project are tz-aware `Europe/Paris`
(CET/UTC+1 in winter, CEST/UTC+2 in summer). Extracting `.hour`,
`.dayofweek`, `.month` directly from a tz-aware `DatetimeIndex` already
yields correct local wall-clock values across the DST transition -- no
manual offset handling is needed for those. The one thing that does need
explicit handling is that a DST transition day has 23 or 25 hourly
delivery periods instead of 24, which matters for any code that assumes a
fixed 24-hours-per-day grid (e.g. day-of-week lag features); this module
flags those days so downstream code can account for it.
"""

from __future__ import annotations

import pandas as pd
from holidays import France


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build calendar features for an hourly, tz-aware `Europe/Paris` index.

    Columns: hour, day_of_week (0=Monday), month, is_weekend,
    is_french_holiday, is_dst_transition_day.
    """
    if index.tz is None:
        raise ValueError("index must be tz-aware (Europe/Paris)")

    fr_holidays = France(years=sorted({ts.year for ts in index}))
    local_dates = index.tz_convert("Europe/Paris").normalize().date

    df = pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": index.dayofweek >= 5,
            "is_french_holiday": [d in fr_holidays for d in local_dates],
        },
        index=index,
    )
    df["is_dst_transition_day"] = _dst_transition_flag(index, local_dates)
    return df


def _dst_transition_flag(index: pd.DatetimeIndex, local_dates: pd.Index) -> pd.Series:
    """True for every row whose local calendar date does not have exactly 24 hourly rows."""
    hours_per_date = pd.Series(1, index=local_dates).groupby(level=0).sum()
    is_transition = hours_per_date != 24
    return pd.Series(local_dates, index=index).map(is_transition)
