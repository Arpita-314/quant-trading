"""Vectorized backtest engine.

This is the single place execution lag and transaction costs are applied,
so every strategy is scored under the same rules. See `strategies/base.py`
for the causality contract strategies must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategies.base import Strategy
from ..utils.metrics import compute_all_metrics


@dataclass
class BacktestResult:
    returns: pd.Series
    positions: pd.DataFrame
    equity_curve: pd.Series
    turnover: pd.Series
    metrics: dict


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    cost_bps: float = 5.0,
    initial_capital: float = 1.0,
    max_gross_exposure: float | None = 1.0,
) -> BacktestResult:
    """Backtest `strategy` on `prices`.

    `cost_bps` is a round-trip-agnostic per-unit-turnover cost in basis
    points, charged on every change in position size (a simple linear
    market-impact + spread proxy -- not a fill simulator).

    Strategies emit a per-asset conviction weight in [-1, 1] independently;
    on a day when several assets in the universe fire at once, summed
    across the book that can exceed 100% notional. `max_gross_exposure` is a
    risk overlay applied here (not inside each strategy) that scales every
    day's book down proportionally so sum(|position|) never exceeds the cap
    -- the same "no leverage beyond X" constraint a real book would run
    under. Pass None to disable and see the raw, uncapped conviction sizing.
    """
    signals = strategy.generate_signals(prices).reindex(columns=prices.columns).fillna(0.0)
    asset_returns = prices.pct_change().fillna(0.0)

    executed = signals.shift(1).fillna(0.0)  # the one place execution lag is applied

    if max_gross_exposure is not None:
        gross = executed.abs().sum(axis=1)
        scale = (max_gross_exposure / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        executed = executed.mul(scale, axis=0)

    gross_returns = (executed * asset_returns).sum(axis=1)

    turnover = executed.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 10_000.0)

    net_returns = gross_returns - costs
    equity_curve = initial_capital * (1.0 + net_returns).cumprod()

    return BacktestResult(
        returns=net_returns,
        positions=executed,
        equity_curve=equity_curve,
        turnover=turnover,
        metrics=compute_all_metrics(net_returns),
    )


def run_many(
    prices: pd.DataFrame,
    strategies: dict[str, Strategy],
    cost_bps: float = 5.0,
    initial_capital: float = 1.0,
    max_gross_exposure: float | None = 1.0,
) -> dict[str, BacktestResult]:
    return {
        name: run_backtest(
            prices,
            strat,
            cost_bps=cost_bps,
            initial_capital=initial_capital,
            max_gross_exposure=max_gross_exposure,
        )
        for name, strat in strategies.items()
    }
