"""Optimal-stopping execution rule for an OU-modelled spread.

Implements the practical threshold-band version of the optimal-stopping /
optimal-liquidation problem for an Ornstein-Uhlenbeck process (see Leung &
Li, "Optimal Mean Reversion Trading", for the free-boundary theory this
approximates in closed form): an entry/exit hysteresis band around the OU
long-run mean `mu`, plus a re-arming rule to stop the strategy from
whipsawing in and out near the entry boundary. All thresholds are
expressed in units of the process's stationary standard deviation, so
they adapt automatically to how volatile the spread currently is.

Let z = (X_t - mu) / stationary_std. Thresholds are ordered
`no_trade_z_score <= exit_z_score <= entry_z_score`:

  * |z| >= entry_z_score: a flat, *armed* trader opens a position betting
    on reversion toward `mu` -- short the spread if z > 0 (spread too
    high), long the spread if z < 0 (spread too low).
  * An open position is closed once |z| reverts into the exit band
    (|z| <= exit_z_score), or once it has been held for
    `max_holding_hours` (pure OU mean reversion gives no guaranteed
    reversion time, so exposure must be bounded).
  * Closing a position **disarms** the strategy: it will not open a new
    one, even if |z| immediately re-crosses `entry_z_score`, until the
    spread has first revisited the tighter `no_trade_z_score` band close
    to `mu`. Without this, a spread oscillating between the exit and
    entry bands would generate a costly round-trip on every oscillation;
    this is the "no-trade zone" the prompt calls for -- not a dead region
    that's simply unreachable (an early draft of this module had
    `no_trade_z_score` nested inside `exit_z_score`, making it
    mathematically redundant; re-arming was the fix).

This module produces the *timing* signal only (when to be in a position,
and in which direction). Combining it with the Phase 3 forecast's
directional edge/magnitude and with cost-aware position sizing is the
backtest engine's job (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from fr_power.config import OuOptimalStoppingConfig


class Position(str, Enum):
    FLAT = "flat"
    LONG = "long"  # betting the spread reverts upward toward mu
    SHORT = "short"  # betting the spread reverts downward toward mu


@dataclass(frozen=True)
class StoppingThresholds:
    entry_z: float
    exit_z: float
    no_trade_z: float
    max_holding_hours: int

    def __post_init__(self) -> None:
        if not (0 <= self.no_trade_z <= self.exit_z <= self.entry_z):
            raise ValueError(
                "Thresholds must satisfy 0 <= no_trade_z <= exit_z <= entry_z, got "
                f"{self.no_trade_z} <= {self.exit_z} <= {self.entry_z}"
            )
        if self.max_holding_hours <= 0:
            raise ValueError(f"max_holding_hours must be positive, got {self.max_holding_hours}")


def thresholds_from_config(config: OuOptimalStoppingConfig) -> StoppingThresholds:
    return StoppingThresholds(
        entry_z=config.entry_z_score,
        exit_z=config.exit_z_score,
        no_trade_z=config.no_trade_z_score,
        max_holding_hours=config.max_holding_hours,
    )


def generate_positions(z_scores: pd.Series, thresholds: StoppingThresholds) -> pd.DataFrame:
    """Simulate the threshold-band execution rule over a z-score series.

    Uses only `z_scores` up to and including the current period at each
    step -- no lookahead. Returns a DataFrame indexed like `z_scores` with:

      - position: "flat" / "long" / "short", the exposure *held during*
        that period (usable directly for PnL attribution downstream).
      - hours_held: periods the current (non-flat) position has been open,
        as of that period; 0 while flat.
      - is_entry / is_exit: True on the period a position is opened / on
        the last period it is held before closing.
      - armed: True if a new entry is currently permitted (see module
        docstring for the re-arming rule).

    A NaN in `z_scores` (e.g. insufficient history to compute the target)
    forces flat for that period and leaves `armed` unchanged.
    """
    position = Position.FLAT
    hours_held = 0
    armed = True

    positions_out: list[str] = []
    hours_held_out: list[int] = []
    is_entry_out: list[bool] = []
    is_exit_out: list[bool] = []
    armed_out: list[bool] = []

    for z in z_scores.to_numpy(dtype=float):
        is_entry = False
        is_exit = False

        if np.isnan(z):
            position = Position.FLAT
            hours_held = 0
            positions_out.append(position.value)
            hours_held_out.append(0)
            is_entry_out.append(False)
            is_exit_out.append(False)
            armed_out.append(armed)
            continue

        abs_z = abs(z)

        if not armed and abs_z < thresholds.no_trade_z:
            armed = True

        if position == Position.FLAT and armed and abs_z >= thresholds.entry_z:
            position = Position.SHORT if z > 0 else Position.LONG
            hours_held = 1
            is_entry = True

        exposure = position

        if position != Position.FLAT:
            should_exit = abs_z <= thresholds.exit_z or hours_held >= thresholds.max_holding_hours
            if should_exit:
                is_exit = True

        positions_out.append(exposure.value)
        hours_held_out.append(hours_held if exposure != Position.FLAT else 0)
        is_entry_out.append(is_entry)
        is_exit_out.append(is_exit)
        armed_out.append(armed)

        if position != Position.FLAT:
            if is_exit:
                position = Position.FLAT
                hours_held = 0
                armed = False
            else:
                hours_held += 1

    return pd.DataFrame(
        {
            "position": positions_out,
            "hours_held": hours_held_out,
            "is_entry": is_entry_out,
            "is_exit": is_exit_out,
            "armed": armed_out,
        },
        index=z_scores.index,
    )
