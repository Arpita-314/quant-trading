"""Insider-buying signal from real SEC Form 4 filings.

Takes a pre-fetched daily net insider dollar-flow matrix (see
`data.sec_edgar.build_daily_insider_flow`) rather than fetching inside
`generate_signals` -- fetching is async I/O and belongs outside the
synchronous strategy interface; this class only turns already-fetched flow
into a position.

Insider *selling* is largely non-discretionary noise in practice (10b5-1
pre-scheduled plans, tax diversification, option-exercise liquidity needs),
while insider *buying* is almost always a voluntary, information-bearing
decision. `long_only=True` (the default) reflects that asymmetry: the
strategy only takes long positions on a buying signal and otherwise sits
flat, rather than shorting on selling.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class InsiderTradingStrategy(Strategy):
    name = "insider_trading"

    def __init__(
        self,
        daily_flow: pd.DataFrame,
        lookback: int = 90,
        z_lookback: int = 252,
        entry_z: float = 0.5,
        long_only: bool = True,
    ):
        self.daily_flow = daily_flow
        self.lookback = lookback
        self.z_lookback = z_lookback
        self.entry_z = entry_z
        self.long_only = long_only

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        flow = self.daily_flow.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)

        rolling_flow = flow.rolling(self.lookback).sum()
        mean = rolling_flow.rolling(self.z_lookback).mean()
        std = rolling_flow.rolling(self.z_lookback).std()
        z = (rolling_flow - mean) / std

        if self.long_only:
            direction = (z > self.entry_z).astype(float)
        else:
            direction = pd.DataFrame(0.0, index=z.index, columns=z.columns)
            direction[z > self.entry_z] = 1.0
            direction[z < -self.entry_z] = -1.0

        strength = (z.abs().clip(upper=3.0) / 3.0).fillna(0.0)
        signal = direction * strength

        return signal.fillna(0.0).clip(-1.0, 1.0)
