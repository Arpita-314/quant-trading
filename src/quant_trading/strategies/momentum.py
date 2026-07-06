"""Time-series momentum (trend following) with inverse-volatility sizing.

Follows the classic academic convention of skipping the most recent `skip`
days of the lookback window to avoid short-term reversal contaminating the
momentum signal (e.g. "12-1 month" momentum).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self, lookback: int = 90, vol_lookback: int = 20, skip: int = 5):
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.skip = skip

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        past_return = prices.shift(self.skip) / prices.shift(self.skip + self.lookback) - 1.0
        direction = np.sign(past_return)

        vol = prices.pct_change().rolling(self.vol_lookback).std()
        inv_vol = 1.0 / vol.replace(0.0, np.nan)

        sized = direction * inv_vol
        gross = sized.abs().sum(axis=1).replace(0.0, np.nan)
        normalized = sized.div(gross, axis=0)

        return normalized.fillna(0.0).clip(-1.0, 1.0)
