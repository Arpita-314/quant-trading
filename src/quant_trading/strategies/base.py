"""Shared interface all strategies implement.

Execution-lag convention (applies to every strategy in this package):
`generate_signals(prices)` returns the target position decided using
information available at each row's close. The backtest engine
(`backtest.engine.run_backtest`) is the ONLY place that shifts signals
forward by one bar before multiplying by returns. Individual strategies
must not add their own extra lag on top of this -- doing so would silently
double-lag execution and understate a strategy's real responsiveness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target positions in [-1, 1], indexed/columned like `prices`.

        Must be causal: signal.loc[date] may only depend on prices.loc[:date].
        """
        raise NotImplementedError
