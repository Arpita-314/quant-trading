"""Filing-language-drift signal ("Lazy Prices", Cohen/Malloy/Nguyen 2020).

Takes a pre-fetched sparse filing-similarity series (see
`data.filing_text.fetch_filing_drift`) rather than fetching inside
`generate_signals` -- fetching is async I/O and belongs outside the
synchronous strategy interface; this class only turns already-fetched
similarity scores into a position.

An unusually large language change (low similarity to the prior same-type
filing, relative to that company's own filing-history baseline) predicts
negative subsequent returns -- short on entry, hold for `holding_days`
trading days, then flat until the next qualifying event. The company's own
historical baseline is used (rather than a cross-sectional rank across the
whole universe) because a 7-ticker universe is far too small for decile
sorting to mean anything; comparing a company's filing to its own history
is a smaller but still meaningful test of "is this unusually different."
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class FilingDriftStrategy(Strategy):
    name = "filing_drift"

    def __init__(
        self,
        drift_events: pd.DataFrame,
        z_lookback: int = 8,
        entry_z: float = 1.0,
        holding_days: int = 60,
    ):
        self.drift_events = drift_events
        self.z_lookback = z_lookback
        self.entry_z = entry_z
        self.holding_days = holding_days

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for ticker in prices.columns:
            if ticker not in self.drift_events.columns:
                continue
            similarity = self.drift_events[ticker].dropna()
            if len(similarity) < 2:
                continue

            change = 1.0 - similarity
            # Compare today's change to the PRIOR z_lookback filings only
            # (shift(1) excludes today's own value) -- measuring surprise
            # relative to an established baseline, not a self-inclusive one.
            rolling_mean = change.shift(1).rolling(self.z_lookback).mean()
            rolling_std = change.shift(1).rolling(self.z_lookback).std()
            z = (change - rolling_mean) / rolling_std

            entry_dates = z[z > self.entry_z].index
            col_idx = signals.columns.get_loc(ticker)
            for event_date in entry_dates:
                candidates = prices.index[prices.index >= event_date]
                if len(candidates) == 0:
                    continue  # event postdates the available price history
                start_loc = prices.index.get_loc(candidates[0])
                end_loc = min(start_loc + self.holding_days, len(prices.index))
                signals.iloc[start_loc:end_loc, col_idx] = -1.0

        return signals.clip(-1.0, 1.0)
