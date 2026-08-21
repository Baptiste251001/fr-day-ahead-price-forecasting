"""Backtest performance metrics: PnL, Sharpe, hit rate, drawdown, turnover,
and block-bootstrap confidence intervals.

Every metric is computed over the *full* out-of-sample period (including
hours the strategy chose not to trade, at zero PnL), matching how a real
book's realized return stream would be measured -- not just the subset of
hours it happened to be active.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class BacktestMetrics:
    gross_pnl_eur: float
    net_pnl_eur: float
    total_cost_eur: float
    pnl_per_mwh_eur: float
    traded_volume_mwh: float
    n_trades: int
    hit_rate: float
    sharpe_annualized: float
    max_drawdown_eur: float
    turnover_mwh: float

    def to_dict(self) -> dict[str, float]:
        return {
            "gross_pnl_eur": self.gross_pnl_eur,
            "net_pnl_eur": self.net_pnl_eur,
            "total_cost_eur": self.total_cost_eur,
            "pnl_per_mwh_eur": self.pnl_per_mwh_eur,
            "traded_volume_mwh": self.traded_volume_mwh,
            "n_trades": self.n_trades,
            "hit_rate": self.hit_rate,
            "sharpe_annualized": self.sharpe_annualized,
            "max_drawdown_eur": self.max_drawdown_eur,
            "turnover_mwh": self.turnover_mwh,
        }


def compute_metrics(result: pd.DataFrame) -> BacktestMetrics:
    """Compute headline metrics from a `backtest.engine.run_backtest` result."""
    net_pnl = result["net_pnl_eur"]
    gross_pnl = result["gross_pnl_eur"]
    cost = result["cost_eur"]
    size = result["size_mwh"]

    traded = size != 0
    n_trades = int(traded.sum())
    traded_volume = float(size.abs().sum())

    gross_total = float(gross_pnl.sum())
    net_total = float(net_pnl.sum())
    cost_total = float(cost.sum())
    pnl_per_mwh = net_total / traded_volume if traded_volume > 0 else 0.0
    hit_rate = float((net_pnl[traded] > 0).mean()) if n_trades > 0 else float("nan")

    mean_pnl = net_pnl.mean()
    std_pnl = net_pnl.std()
    sharpe = float(mean_pnl / std_pnl * np.sqrt(HOURS_PER_YEAR)) if std_pnl > 0 else 0.0

    cumulative = net_pnl.cumsum()
    drawdown = cumulative.cummax() - cumulative
    max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0

    return BacktestMetrics(
        gross_pnl_eur=gross_total,
        net_pnl_eur=net_total,
        total_cost_eur=cost_total,
        pnl_per_mwh_eur=pnl_per_mwh,
        traded_volume_mwh=traded_volume,
        n_trades=n_trades,
        hit_rate=hit_rate,
        sharpe_annualized=sharpe,
        max_drawdown_eur=max_drawdown,
        # Each hourly product is a single fresh one-shot trade (see engine.py
        # docstring) rather than a position with deltas to track, so turnover
        # reduces to total traded volume.
        turnover_mwh=traded_volume,
    )


def block_bootstrap_ci(
    net_pnl: pd.Series,
    block_length_hours: int,
    n_resamples: int,
    confidence_level: float,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Block-bootstrap confidence intervals for total net PnL and annualized Sharpe.

    Resamples contiguous blocks of `net_pnl` with replacement (preserving
    short-range autocorrelation) rather than resampling individual hours iid.
    """
    if block_length_hours <= 0:
        raise ValueError(f"block_length_hours must be positive, got {block_length_hours}")
    values = net_pnl.to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("net_pnl must be non-empty")

    rng = np.random.default_rng(seed)
    block_length = min(block_length_hours, n)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length

    pnl_samples = np.empty(n_resamples)
    sharpe_samples = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([values[s : s + block_length] for s in starts])[:n]
        pnl_samples[i] = sample.sum()
        std = sample.std()
        sharpe_samples[i] = float(sample.mean() / std * np.sqrt(HOURS_PER_YEAR)) if std > 0 else 0.0

    alpha = 1 - confidence_level
    lower_q, upper_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    return {
        "net_pnl_eur": (
            float(np.percentile(pnl_samples, lower_q)),
            float(np.percentile(pnl_samples, upper_q)),
        ),
        "sharpe_annualized": (
            float(np.percentile(sharpe_samples, lower_q)),
            float(np.percentile(sharpe_samples, upper_q)),
        ),
    }
