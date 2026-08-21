"""Walk-forward, cost-aware backtest combining the Phase 4 OU timing signal
with the Phase 3 forecast's directional/magnitude confirmation.

Trade structure
----------------
Each row of the feature matrix is one hourly delivery product's
intraday-vs-day-ahead spread (`target_spread_eur_mwh`). Given this
repository's default data (a per-product *settled* spread series, not a
granular intraday tick path within one product's trading window -- see
`fr_power.data.intraday`), each hourly product is a **one-shot** trade:
a position is decided once, at the D-1 information cutoff, and it settles
automatically against the realized spread -- there is no multi-hour
holding of the *same* instrument to voluntarily exit from.

This is why the engine below computes an OU-based z-score directly
(the same `(x - mu) / stationary_std` as
`fr_power.ou.estimate.compute_z_score`) but does not use
`fr_power.ou.optimal_stopping.generate_positions`: that state machine's
entry/exit/holding-period logic is designed for
holding a continuously-tradeable position *within* one product's trading
window, which is exactly what it will be useful for once a granular
intraday feed is plugged in. For now, `entry_z_score` (from
`ou.optimal_stopping`) is reused as the OU conviction threshold gating
whether a given hourly product is traded at all; `exit_z_score`,
`no_trade_z_score` and `max_holding_hours` do not apply here.

Point-in-time safety
---------------------
The OU process is recalibrated walk-forward (expanding window, same
cadence as `backtest.walk_forward`), using only realized spread history
strictly before the test fold. Within a fold, the *current* deviation from
the OU mean is read off a lagged, already-realized spread feature (the
shortest configured `features.spread_lags_hours`, e.g. "yesterday same
hour") -- never the current row's own `target_spread_eur_mwh`, which is
the very thing being predicted/traded and only becomes known after the
fact. The forecast (`predictions`) is assumed to already be a
point-in-time-safe, walk-forward out-of-sample prediction (see
`fr_power.models.walk_forward`).

Signal combination
-------------------
  * OU direction: -1 (short the spread, expect it to fall) if the lagged
    z-score exceeds +entry_z_score; +1 (long) if it is below
    -entry_z_score; 0 (no OU conviction) otherwise.
  * Forecast confirmation: the forecast's implied edge (predicted spread
    minus the same lagged reference point) must agree in sign with the OU
    direction, or the trade is skipped entirely -- this is the "forecast
    gives the directional edge" half of the combination.
  * Confidence-weighted sizing: position size scales with the forecast
    edge's magnitude relative to the OU process's stationary std, capped
    at `backtest.max_position_mwh`.
  * Cost-aware no-trade band: a trade is skipped unless the forecast
    edge's magnitude clears `backtest.no_trade_band_eur_per_mwh` *plus*
    the round-trip-equivalent transaction cost -- a small, real edge that
    costs more to trade than it's worth is not a trade.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fr_power.config import BacktestConfig, CostsConfig, OuConfig
from fr_power.ou.estimate import estimate_ou

logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "z_score",
    "forecast_edge",
    "direction",
    "confidence",
    "size_mwh",
    "gross_pnl_eur",
    "cost_eur",
    "net_pnl_eur",
]


def compute_fold_signal(
    reference: pd.Series,
    predictions: pd.Series,
    mu: float,
    stationary_std: float,
    entry_z: float,
    no_trade_band_eur_per_mwh: float,
    cost_per_mwh: float,
    max_position_mwh: float,
) -> pd.DataFrame:
    """Pure signal computation for one fold, given already-calibrated OU parameters.

    Kept separate from OU estimation (and from `run_backtest`'s walk-forward
    loop) so the sizing/gating rules can be tested exactly against
    hand-picked parameters, without OLS/MLE estimation noise in the mix.
    Returns a DataFrame indexed like `reference` with columns z_score,
    forecast_edge, direction, confidence, size_mwh.
    """
    z = (reference - mu) / stationary_std
    forecast_edge = predictions.reindex(reference.index) - reference

    direction = pd.Series(0.0, index=reference.index)
    direction[z > entry_z] = -1.0
    direction[z < -entry_z] = 1.0

    agrees = np.sign(forecast_edge) == direction
    confidence = (forecast_edge.abs() / stationary_std).clip(upper=1.0)
    confidence = confidence.where(agrees & (direction != 0), 0.0)

    min_edge = no_trade_band_eur_per_mwh + cost_per_mwh
    confidence = confidence.where(forecast_edge.abs() >= min_edge, 0.0)

    size = (direction * max_position_mwh * confidence).clip(
        lower=-max_position_mwh, upper=max_position_mwh
    )

    return pd.DataFrame(
        {
            "z_score": z,
            "forecast_edge": forecast_edge,
            "direction": direction,
            "confidence": confidence,
            "size_mwh": size,
        },
        index=reference.index,
    )


def run_backtest(
    spread: pd.Series,
    signal_reference: pd.Series,
    predictions: pd.Series,
    costs: CostsConfig,
    backtest_cfg: BacktestConfig,
    ou_cfg: OuConfig,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Run the walk-forward combined OU + forecast backtest.

    Parameters
    ----------
    spread : realized `target_spread_eur_mwh`, used both to calibrate OU
        (training folds only) and to settle PnL (test folds).
    signal_reference : the point-in-time-safe lagged spread feature used
        to compute "where does the process currently stand" (e.g.
        `spread_lag_24h`).
    predictions : a walk-forward out-of-sample forecast of `spread`,
        aligned to the same index (NaN outside its own out-of-sample
        period is fine -- those rows simply never trade here).
    cost_multiplier : scales `costs.fee_eur_per_mwh` and
        `costs.half_spread_eur_per_mwh`, for the cost-sensitivity sweep.

    Returns
    -------
    A DataFrame indexed like `spread` with the columns in `RESULT_COLUMNS`
    (all zero/NaN outside any out-of-sample fold).
    """
    n = len(spread)
    index = spread.index
    out = pd.DataFrame(0.0, index=index, columns=RESULT_COLUMNS)
    out["z_score"] = np.nan
    out["forecast_edge"] = np.nan

    wf = backtest_cfg.walk_forward
    initial_train_hours = wf.initial_train_days * 24
    retrain_every_hours = wf.retrain_every_days * 24
    entry_z = ou_cfg.optimal_stopping.entry_z_score
    cost_per_mwh = (costs.fee_eur_per_mwh + costs.half_spread_eur_per_mwh) * cost_multiplier

    if initial_train_hours >= n:
        logger.warning(
            "initial_train_hours (%d) >= number of rows (%d); no backtest folds possible.",
            initial_train_hours,
            n,
        )
        return out

    train_end = initial_train_hours
    n_folds = 0
    n_folds_skipped = 0
    while train_end < n:
        test_end = min(train_end + retrain_every_hours, n)
        train_spread = spread.iloc[:train_end]

        try:
            params = estimate_ou(
                train_spread,
                method=ou_cfg.estimation.method,
                min_observations=ou_cfg.estimation.min_observations,
            )
        except ValueError:
            n_folds_skipped += 1
            train_end = test_end
            continue

        fold_slice = slice(train_end, test_end)
        fold_index = index[fold_slice]
        ref = signal_reference.iloc[fold_slice]
        realized = spread.iloc[fold_slice]

        signal = compute_fold_signal(
            ref,
            predictions,
            params.mu,
            params.stationary_std,
            entry_z,
            backtest_cfg.no_trade_band_eur_per_mwh,
            cost_per_mwh,
            backtest_cfg.max_position_mwh,
        )
        gross_pnl = signal["size_mwh"] * realized.to_numpy()
        cost = signal["size_mwh"].abs() * cost_per_mwh
        net_pnl = gross_pnl - cost

        out.loc[fold_index, ["z_score", "forecast_edge", "direction", "confidence", "size_mwh"]] = (
            signal.to_numpy()
        )
        out.loc[fold_index, "gross_pnl_eur"] = gross_pnl.to_numpy()
        out.loc[fold_index, "cost_eur"] = cost.to_numpy()
        out.loc[fold_index, "net_pnl_eur"] = net_pnl.to_numpy()

        n_folds += 1
        train_end = test_end

    logger.info(
        "Backtest completed: %d fold(s) traded, %d skipped (OU calibration failed).",
        n_folds,
        n_folds_skipped,
    )
    return out


def perfect_foresight_backtest(
    spread: pd.Series,
    costs: CostsConfig,
    backtest_cfg: BacktestConfig,
    cost_multiplier: float = 1.0,
    start_index: int = 0,
) -> pd.DataFrame:
    """Upper-bound comparator: full size in the *correct* direction whenever the
    realized spread's magnitude clears costs plus the no-trade band, from `start_index`
    onward (matching the first period a real strategy could have started trading)."""
    index = spread.index
    out = pd.DataFrame(0.0, index=index, columns=RESULT_COLUMNS)
    out["z_score"] = np.nan
    out["forecast_edge"] = np.nan

    cost_per_mwh = (costs.fee_eur_per_mwh + costs.half_spread_eur_per_mwh) * cost_multiplier
    min_edge = backtest_cfg.no_trade_band_eur_per_mwh + cost_per_mwh

    tradable = spread.iloc[start_index:]
    qualifies = tradable.abs() >= min_edge
    direction = np.sign(tradable).where(qualifies, 0.0)
    size = (direction * backtest_cfg.max_position_mwh).where(qualifies, 0.0)
    gross_pnl = size * tradable
    cost = size.abs() * cost_per_mwh
    net_pnl = gross_pnl - cost

    out.loc[tradable.index, "direction"] = direction
    out.loc[tradable.index, "size_mwh"] = size
    out.loc[tradable.index, "gross_pnl_eur"] = gross_pnl
    out.loc[tradable.index, "cost_eur"] = cost
    out.loc[tradable.index, "net_pnl_eur"] = net_pnl
    return out
