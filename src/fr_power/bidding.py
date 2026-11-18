"""Day-ahead bidding for a small, price-taking battery, driven by the price forecasts.

A forecast only has value once it becomes orders. This module turns the walk-forward
forecasts into SDAC-style orders for a battery, settles them against the cleared prices and
measures the result in euros.

Strategies compared on every delivery day:

- `perfect_foresight`: the optimal schedule if day D's prices were known (upper bound).
- `schedule_<model>`: optimise the battery on the model's point forecast and submit the
  result as price-independent quantities (the simplest desk practice).
- `curves_<model>`: build one price-quantity bid curve per period from forecast scenarios
  (the model's forecast plus whole-day error vectors from the previous weeks), by
  maximising the expected profit over the scenarios. This is the classic scenario-based
  bidding formulation (Fleten & Pettersen, 2005), adapted to storage.

Each day is traded at the resolution of its products: 24 hourly orders until 30 September
2025, 96 quarter-hour orders from 1 October 2025. Quantities are powers (MW); a period of
`dt` hours moves `power * dt` MWh.

Settlement. The executed quantity of each period is read on its curve at the cleared price
(linear interpolation, as with EPEX linear piecewise orders). A curve set may commit the
battery to sell energy it does not hold if the cleared prices fall outside the scenarios.
The battery then delivers the feasible dispatch closest to its commitments, and the gap is
settled as an imbalance: at the day-ahead price plus a penalty (a simple proxy for the
imbalance price).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from fr_power.data import MTU_15MIN_START

# Small cost per MWh charged or discharged: it breaks ties so the optimiser never cycles
# the battery for nothing. It is not counted in the reported profit.
_THROUGHPUT_TIE_BREAK = 1e-3


@dataclass(frozen=True)
class Battery:
    power_mw: float
    energy_mwh: float
    round_trip_efficiency: float

    @property
    def one_way_efficiency(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


@dataclass(frozen=True)
class BidCurves:
    """One curve per delivery period: increasing prices and net sell quantities (MW).

    Positive quantities are sales (discharge), negative ones purchases (charge). Between
    points the curve is linear, beyond the end points it is flat.
    """

    prices: np.ndarray  # shape (periods, points)
    quantities: np.ndarray  # shape (periods, points), non-decreasing along points

    def committed(self, clearing_prices: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.interp(p, xp, fp)
                for p, xp, fp in zip(clearing_prices, self.prices, self.quantities, strict=True)
            ]
        )


def optimal_curves(scenarios: np.ndarray, battery: Battery, dt: float = 1.0) -> BidCurves:
    """Bid curves maximising the mean profit over price scenarios (shape scenarios x periods).

    `dt` is the period length in hours. The bid points of each period are that period's
    scenario prices, so in scenario s the battery executes x[h, rank of s in period h]. Curves must be non-decreasing in price (sell more
    when the price is higher), and the resulting dispatch must be feasible for the battery in
    every scenario: power limits, state of charge between 0 and capacity, empty at the start
    of the day. With a single scenario this is the deterministic optimal schedule.
    """
    n_s, n_h = scenarios.shape
    eta = battery.one_way_efficiency
    order = np.argsort(scenarios, axis=0, kind="stable")  # (S, H): scenario index by rank
    sorted_prices = np.take_along_axis(scenarios, order, axis=0)  # (S, H)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n_s)[:, None].repeat(n_h, axis=1), axis=0)

    n = n_h * n_s
    x0, c0, d0, e0 = 0, n, 2 * n, 3 * n  # offsets: curve points, charge, discharge, energy

    def x_idx(h: np.ndarray, k: np.ndarray) -> np.ndarray:
        return x0 + h * n_s + k

    s_grid, h_grid = np.meshgrid(np.arange(n_s), np.arange(n_h), indexing="ij")
    s_flat, h_flat = s_grid.ravel(), h_grid.ravel()
    sh = s_flat * n_h + h_flat  # index of (s, h) within the per-scenario blocks

    # Objective: minimise minus the mean revenue, sum_s sum_h price[s,h] * x[h, rank[s,h]].
    cost = np.zeros(4 * n)
    cost[x_idx(h_grid.T.ravel(), s_grid.T.ravel())] = -sorted_prices.T.ravel() * dt / n_s
    cost[c0 : c0 + n] = cost[d0 : d0 + n] = _THROUGHPUT_TIE_BREAK * dt / n_s

    # Dispatch: discharge - charge - x[h, rank] = 0, for each scenario and hour.
    rows = np.concatenate([np.arange(n)] * 3)
    cols = np.concatenate([d0 + sh, c0 + sh, x_idx(h_flat, rank[s_flat, h_flat])])
    vals = np.concatenate([np.ones(n), -np.ones(n), -np.ones(n)])
    # Energy balance: e[h] - e[h-1] - dt * (eta * charge - discharge / eta) = 0, e[-1] = 0.
    prev = sh[h_flat > 0] - 1
    rows = np.concatenate(
        [rows, n + np.arange(n), n + np.arange(n), n + np.arange(n), n + np.flatnonzero(h_flat > 0)]
    )
    cols = np.concatenate([cols, e0 + sh, c0 + sh, d0 + sh, e0 + prev])
    vals = np.concatenate(
        [vals, np.ones(n), -eta * dt * np.ones(n), dt * np.ones(n) / eta, -np.ones(len(prev))]
    )
    a_eq = sparse.csr_array((vals, (rows, cols)), shape=(2 * n, 4 * n))

    # Monotone curves: x[h, k] - x[h, k+1] <= 0.
    h_m, k_m = np.meshgrid(np.arange(n_h), np.arange(n_s - 1), indexing="ij")
    m = h_m.size
    a_ub = sparse.csr_array(
        (
            np.concatenate([np.ones(m), -np.ones(m)]),
            (
                np.concatenate([np.arange(m)] * 2),
                np.concatenate(
                    [x_idx(h_m.ravel(), k_m.ravel()), x_idx(h_m.ravel(), k_m.ravel() + 1)]
                ),
            ),
        ),
        shape=(m, 4 * n),
    )

    p, e = battery.power_mw, battery.energy_mwh
    bounds = [(-p, p)] * n + [(0, p)] * (2 * n) + [(0, e)] * n
    result = linprog(
        cost,
        A_ub=a_ub if m else None,
        b_ub=np.zeros(m) if m else None,
        A_eq=a_eq,
        b_eq=np.zeros(2 * n),
        bounds=bounds,
        method="highs-ipm",  # ~10x faster than simplex on the 96-period problems
    )
    if not result.success:
        raise RuntimeError(f"Bidding LP failed: {result.message}")
    quantities = result.x[x0 : x0 + n].reshape(n_h, n_s)
    return BidCurves(prices=sorted_prices.T, quantities=np.round(quantities, 6))


def settle(
    curves: BidCurves,
    clearing_prices: np.ndarray,
    battery: Battery,
    imbalance_penalty: float,
    dt: float = 1.0,
) -> dict[str, float]:
    """Execute the curves at the cleared prices, deliver what is feasible, settle the gap.

    Profit = dt * [sum(price * delivered) - penalty * sum(|delivered - committed|)]: the committed
    volume is paid at the day-ahead price and any shortfall or excess is traded back at the
    same price plus the penalty.

    The whole day's position is known once the auction clears (D-1, ~12:45), so the operator
    plans the physical dispatch for the day at once: the feasible dispatch closest to the
    commitments (least total imbalance). Prices play no part in that choice, so the imbalance
    can never be used to re-trade after the fact.
    """
    committed = curves.committed(clearing_prices)
    delivered = _closest_feasible_dispatch(committed, battery, dt)
    imbalance = np.abs(delivered - committed).sum() * dt
    return {
        "profit_eur": float(clearing_prices @ delivered * dt - imbalance_penalty * imbalance),
        "imbalance_mwh": float(imbalance),
        "discharged_mwh": float(delivered.clip(min=0).sum() * dt),
    }


def _closest_feasible_dispatch(
    committed: np.ndarray, battery: Battery, dt: float = 1.0
) -> np.ndarray:
    """Net sales (MW) that minimise sum |delivered - committed| under the battery limits.

    Variables per period: charge, discharge, energy, positive and negative deviation. Like
    the bidding optimiser, the battery may charge and discharge in the same period, which burns
    energy as conversion losses (useful to absorb a purchase larger than the free capacity).
    """
    n, eta = len(committed), battery.one_way_efficiency
    c, d, e, up, down = (np.arange(n) + k * n for k in range(5))
    cost = np.zeros(5 * n)
    cost[up] = cost[down] = 1.0
    cost[c] = cost[d] = _THROUGHPUT_TIE_BREAK
    a_eq = np.zeros((2 * n, 5 * n))
    rows = np.arange(n)
    # discharge - charge - up + down = committed  (delivered = committed + up - down)
    a_eq[rows, d], a_eq[rows, c], a_eq[rows, up], a_eq[rows, down] = 1, -1, -1, 1
    # e[h] - e[h-1] - dt * (eta * charge - discharge / eta) = 0, with e[-1] = 0
    a_eq[n + rows, e], a_eq[n + rows, c], a_eq[n + rows, d] = 1, -eta * dt, dt / eta
    a_eq[n + rows[1:], e[:-1]] = -1
    p, cap = battery.power_mw, battery.energy_mwh
    bounds = [(0, p)] * (2 * n) + [(0, cap)] * n + [(0, None)] * (2 * n)
    result = linprog(cost, A_eq=a_eq, b_eq=np.concatenate([committed, np.zeros(n)]), bounds=bounds)
    if not result.success:  # pragma: no cover - always feasible (idle battery)
        raise RuntimeError(f"Dispatch LP failed: {result.message}")
    return np.asarray(result.x[d] - result.x[c])


def error_table(actual: pd.Series, forecast: pd.Series) -> pd.DataFrame:
    """Signed forecast errors (actual - forecast), one row per day, one column per period.

    Columns are minutes since midnight in local time (0, 15, ..., 1425).
    """
    errors = (actual - forecast).dropna()
    slot = errors.index.hour * 60 + errors.index.minute
    table = errors.groupby([errors.index.date, slot]).mean().unstack()
    return table.ffill(axis=1).bfill(axis=1)  # a DST day lacks one hour of labels


def scenarios_for_day(
    forecast_day: pd.Series, errors: pd.DataFrame, day: dt.date, window_days: int, min_days: int
) -> np.ndarray | None:
    """Price scenarios for a delivery day: its forecast plus each past day's error vector.

    Only errors of the `window_days` days before `day` are used; they are all known at
    decision time on D-1 (D-1's prices clear on D-2). Whole-day vectors keep the correlation
    between periods, which is what matters for a battery. Returns None if too few days.
    """
    start = day - dt.timedelta(days=window_days)
    past = errors.loc[(errors.index < day) & (errors.index >= start)]
    past = past.dropna()
    if len(past) < min_days:
        return None
    slots = forecast_day.index.hour * 60 + forecast_day.index.minute
    return forecast_day.to_numpy()[None, :] + past[slots].to_numpy()


def traded_products(day_frame: pd.DataFrame, day: dt.date) -> tuple[pd.DataFrame, float]:
    """A day's 15-minute data at the resolution of its products, and the period in hours.

    Before the switch the auction traded hourly products: quarters are averaged to hours.
    """
    if day >= MTU_15MIN_START.date():
        return day_frame, 0.25
    return day_frame.resample("1h").mean(), 1.0


def backtest(
    actual: pd.Series,
    predictions: pd.DataFrame,
    battery: Battery,
    schedule_models: list[str],
    curve_model: str,
    window_days: int = 60,
    min_days: int = 20,
    imbalance_penalty: float = 50.0,
) -> pd.DataFrame:
    """Daily profit of every strategy, on days where all forecasts and scenarios exist.

    `actual` and `predictions` are 15-minute series. Days before the switch to 15-minute
    products are traded as 24 hourly orders (quarters averaged to hours), later days as 96
    quarter-hour orders. Returns one row per delivery day and a (strategy, metric) column
    MultiIndex with metrics `profit_eur`, `imbalance_mwh` and `discharged_mwh`.
    """
    models = list(dict.fromkeys([*schedule_models, curve_model]))
    df = pd.concat([actual.rename("actual"), predictions[models]], axis=1)
    errors = error_table(actual, predictions[curve_model])
    rows = {}
    for day, quarters in df.groupby(df.index.date):
        day_df, dt_hours = traded_products(quarters, day)
        if day_df.isna().any().any() or len(day_df) < 23 / dt_hours:
            continue
        scenarios = scenarios_for_day(day_df[curve_model], errors, day, window_days, min_days)
        if scenarios is None:
            continue
        prices = day_df["actual"].to_numpy()
        strategies = {"perfect_foresight": prices[None, :]}
        strategies |= {f"schedule_{m}": day_df[m].to_numpy()[None, :] for m in schedule_models}
        strategies[f"curves_{curve_model}"] = scenarios
        rows[day] = {
            (name, metric): value
            for name, scen in strategies.items()
            for metric, value in settle(
                optimal_curves(scen, battery, dt_hours),
                prices,
                battery,
                imbalance_penalty,
                dt_hours,
            ).items()
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["strategy", "metric"])
    out.index = pd.to_datetime(out.index).rename("date")
    return out


def summarise(daily: pd.DataFrame, battery: Battery) -> pd.DataFrame:
    """Profit per MW per year, share of perfect-foresight profit, imbalance and cycles."""
    profit = daily.xs("profit_eur", axis=1, level="metric")
    years = len(daily) / 365.25
    out = pd.DataFrame(
        {
            "profit_eur_per_mw_year": profit.sum() / battery.power_mw / years,
            "capture_vs_perfect": profit.sum() / profit["perfect_foresight"].sum(),
            "imbalance_mwh_per_day": daily.xs("imbalance_mwh", axis=1, level="metric").mean(),
            "cycles_per_day": daily.xs("discharged_mwh", axis=1, level="metric").mean()
            / battery.energy_mwh,
        }
    )
    out.attrs["n_days"] = len(daily)
    return out


def profit_by_year(daily: pd.DataFrame, battery: Battery) -> pd.DataFrame:
    """Profit per MW for each calendar year (EUR/MW)."""
    profit = daily.xs("profit_eur", axis=1, level="metric")
    return (profit.groupby(profit.index.year).sum() / battery.power_mw).rename_axis("year")
