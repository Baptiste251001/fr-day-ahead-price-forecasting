"""Tests for the OU threshold-band execution rule, traced step by step against
hand-computed expected state sequences."""

import numpy as np
import pandas as pd
import pytest

from fr_power.config import OuOptimalStoppingConfig
from fr_power.ou.optimal_stopping import (
    StoppingThresholds,
    generate_positions,
    thresholds_from_config,
)


def _z_series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="Europe/Paris")
    return pd.Series(values, index=index)


def test_thresholds_reject_invalid_ordering() -> None:
    with pytest.raises(ValueError, match="must satisfy"):
        StoppingThresholds(entry_z=1.0, exit_z=1.5, no_trade_z=0.1, max_holding_hours=6)


def test_thresholds_reject_non_positive_max_holding() -> None:
    with pytest.raises(ValueError, match="max_holding_hours"):
        StoppingThresholds(entry_z=1.0, exit_z=0.3, no_trade_z=0.1, max_holding_hours=0)


def test_thresholds_from_config() -> None:
    cfg = OuOptimalStoppingConfig(
        entry_z_score=1.0, exit_z_score=0.3, no_trade_z_score=0.1, max_holding_hours=6
    )
    thresholds = thresholds_from_config(cfg)
    assert thresholds == StoppingThresholds(
        entry_z=1.0, exit_z=0.3, no_trade_z=0.1, max_holding_hours=6
    )


def test_short_entry_and_exit_on_reversion() -> None:
    thresholds = StoppingThresholds(entry_z=1.0, exit_z=0.3, no_trade_z=0.1, max_holding_hours=100)
    z = _z_series([0.0, 0.5, 1.5, 1.2, 0.2, 0.0])

    result = generate_positions(z, thresholds)

    assert result["position"].tolist() == ["flat", "flat", "short", "short", "short", "flat"]
    assert result["hours_held"].tolist() == [0, 0, 1, 2, 3, 0]
    assert result["is_entry"].tolist() == [False, False, True, False, False, False]
    assert result["is_exit"].tolist() == [False, False, False, False, True, False]


def test_long_entry_on_negative_deviation() -> None:
    thresholds = StoppingThresholds(entry_z=1.0, exit_z=0.3, no_trade_z=0.1, max_holding_hours=100)
    z = _z_series([-1.5, -1.5, -0.2])

    result = generate_positions(z, thresholds)

    assert result["position"].tolist() == ["long", "long", "long"]
    assert result["is_entry"].tolist() == [True, False, False]
    assert result["is_exit"].tolist() == [False, False, True]
    assert result["hours_held"].tolist() == [1, 2, 3]


def test_max_holding_hours_forces_exit_without_reversion() -> None:
    thresholds = StoppingThresholds(entry_z=1.0, exit_z=0.1, no_trade_z=0.05, max_holding_hours=2)
    z = _z_series([1.5, 1.5, 1.5, 1.5])

    result = generate_positions(z, thresholds)

    assert result["position"].tolist() == ["short", "short", "flat", "flat"]
    assert result["hours_held"].tolist() == [1, 2, 0, 0]
    assert result["is_entry"].tolist() == [True, False, False, False]
    assert result["is_exit"].tolist() == [False, True, False, False]


def test_rearming_blocks_immediate_whipsaw_reentry() -> None:
    thresholds = StoppingThresholds(entry_z=1.0, exit_z=0.3, no_trade_z=0.1, max_holding_hours=100)
    # enter short, exit on reversion, immediately re-cross entry (blocked: not yet re-armed),
    # dip through the no-trade zone (re-arms), then re-cross entry (now allowed).
    z = _z_series([1.5, 0.2, 1.5, 0.05, 1.5])

    result = generate_positions(z, thresholds)

    assert result["position"].tolist() == ["short", "short", "flat", "flat", "short"]
    assert result["is_entry"].tolist() == [True, False, False, False, True]
    assert result["is_exit"].tolist() == [False, True, False, False, False]
    assert result["armed"].tolist() == [True, True, False, True, True]
    assert result["hours_held"].tolist() == [1, 2, 0, 0, 1]


def test_nan_forces_flat_and_does_not_count_as_exit() -> None:
    thresholds = StoppingThresholds(entry_z=1.0, exit_z=0.3, no_trade_z=0.1, max_holding_hours=100)
    z = _z_series([1.5, np.nan, 1.5])

    result = generate_positions(z, thresholds)

    assert result["position"].tolist() == ["short", "flat", "short"]
    assert result["is_entry"].tolist() == [True, False, True]
    assert result["is_exit"].tolist() == [False, False, False]
    assert result["hours_held"].tolist() == [1, 0, 1]


def test_never_entering_stays_flat_throughout() -> None:
    thresholds = StoppingThresholds(entry_z=1.0, exit_z=0.3, no_trade_z=0.1, max_holding_hours=100)
    z = _z_series([0.1, -0.2, 0.5, -0.6, 0.9])

    result = generate_positions(z, thresholds)

    assert (result["position"] == "flat").all()
    assert not result["is_entry"].any()
    assert not result["is_exit"].any()
