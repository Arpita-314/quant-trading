"""Adaptive capital allocator across a basket of strategies.

This is the "agent" in the portfolio-construction sense used by
multi-strategy quant funds: no single signal is trusted heavily. Capital is
continuously reweighted toward strategies with strong trailing risk-adjusted
performance and away from whatever looks dead, on a fixed rebalance cadence.
It is a deterministic, fully explainable rule -- rolling Sharpe -> softmax-free
normalized weights -- not an LLM making trade decisions. That is a deliberate
choice: a fund cares whether an allocation rule is auditable and backtestable
with a stable random seed, not whether it sounds like AI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdaptiveEnsembleAgent:
    def __init__(
        self,
        lookback: int = 60,
        rebalance_every: int = 21,
        min_weight: float = 0.0,
        max_weight: float = 0.6,
    ):
        self.lookback = lookback
        self.rebalance_every = rebalance_every
        self.min_weight = min_weight
        self.max_weight = max_weight

    def allocate(self, strategy_returns: pd.DataFrame) -> pd.DataFrame:
        """Return per-strategy capital weights, decided causally at each
        rebalance date using only data through the prior close."""
        cols = strategy_returns.columns
        n_strats = len(cols)
        equal_weight = pd.Series(1.0 / n_strats, index=cols)

        rolling_sharpe = (
            strategy_returns.rolling(self.lookback).mean()
            / strategy_returns.rolling(self.lookback).std()
            * np.sqrt(252)
        )

        weights = pd.DataFrame(index=strategy_returns.index, columns=cols, dtype=float)
        current = equal_weight
        for i, date in enumerate(strategy_returns.index):
            if i > 0 and i % self.rebalance_every == 0:
                scores = rolling_sharpe.iloc[i - 1].fillna(0.0).clip(lower=0.0)
                if scores.sum() <= 0:
                    current = equal_weight
                else:
                    raw = scores / scores.sum()
                    capped = raw.clip(lower=self.min_weight, upper=self.max_weight)
                    current = capped / capped.sum()
            weights.loc[date] = current
        return weights

    def combine(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """Blend strategy return streams into a single ensemble return series.

        Weights decided at date t are applied to date t+1's return, consistent
        with the backtest engine's one-bar execution-lag convention.
        """
        weights = self.allocate(strategy_returns)
        applied = weights.shift(1).fillna(1.0 / weights.shape[1])
        return (applied * strategy_returns).sum(axis=1)
