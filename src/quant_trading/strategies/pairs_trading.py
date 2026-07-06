"""Cointegration-flavored statistical arbitrage between two assets.

Use `select_pair` (in this module) offline to screen candidate pairs for
cointegration before wiring one up here -- this class assumes the pair is
already a reasonable stat-arb candidate and only handles signal generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def engle_granger_pvalue(a: pd.Series, b: pd.Series) -> float:
    """Engle-Granger cointegration test p-value (low = likely cointegrated)."""
    from statsmodels.tsa.stattools import coint

    _, pvalue, _ = coint(a, b)
    return float(pvalue)


class PairsTradingStrategy(Strategy):
    name = "pairs_trading"

    def __init__(
        self,
        asset_a: str,
        asset_b: str,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def _hedge_ratio(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Rolling beta_t = Cov(a, b) / Var(b) over `lookback`, causal at each t."""
        cov = a.rolling(self.lookback).cov(b)
        var = b.rolling(self.lookback).var()
        return cov / var

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        a = prices[self.asset_a]
        b = prices[self.asset_b]

        beta = self._hedge_ratio(a, b)
        spread = a - beta * b
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        z = (spread - mean) / std

        # Entry/exit thresholds need explicit state (stay short/long until the
        # spread reverts), which doesn't vectorize cleanly -- a plain loop over
        # a few thousand daily bars is fast enough and much easier to verify
        # than a vectorized state-machine trick.
        z_vals = z.to_numpy()
        position = np.zeros(len(z_vals))
        state = 0.0
        for i, zi in enumerate(z_vals):
            if np.isnan(zi):
                state = 0.0
            elif state == 0.0:
                if zi > self.entry_z:
                    state = -1.0
                elif zi < -self.entry_z:
                    state = 1.0
            elif abs(zi) < self.exit_z:
                state = 0.0
            position[i] = state

        pos_a = pd.Series(position, index=prices.index)
        beta_filled = beta.fillna(0.0)

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[self.asset_a] = pos_a
        signals[self.asset_b] = -pos_a * beta_filled

        gross = signals.abs().sum(axis=1).replace(0.0, np.nan)
        scale = (1.0 / gross).clip(upper=1.0)
        return signals.mul(scale, axis=0).fillna(0.0)
