"""Per-asset time-series mean reversion on a rolling z-score (Bollinger-style)."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 1.0,
        exit_z: float = 0.25,
        max_z: float = 3.0,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_z = max_z

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        rolling_mean = prices.rolling(self.lookback).mean()
        rolling_std = prices.rolling(self.lookback).std()
        z = (prices - rolling_mean) / rolling_std

        direction = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        direction[z > self.entry_z] = -1.0  # overbought -> short
        direction[z < -self.entry_z] = 1.0  # oversold -> long

        strength = (z.abs().clip(upper=self.max_z) / self.max_z).fillna(0.0)
        signal = direction * strength

        flat_mask = z.abs() < self.exit_z
        signal = signal.mask(flat_mask, 0.0)

        return signal.fillna(0.0).clip(-1.0, 1.0)
