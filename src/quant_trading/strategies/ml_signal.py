"""ML-driven directional signal with walk-forward (expanding-window) retraining.

The model is refit every `retrain_every` bars using only data strictly
before the current bar, then used to score the current bar. This is the
walk-forward validation loop and the live signal-generation loop fused into
one pass -- there is no separate "fit once on everything" step, which is
the most common source of lookahead bias in ML trading signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import Strategy


def _build_features(prices: pd.Series) -> pd.DataFrame:
    ret = prices.pct_change()
    feats = pd.DataFrame(index=prices.index)
    for lag in (1, 2, 3, 5, 10, 20):
        feats[f"ret_sum_{lag}"] = ret.rolling(lag).sum()
    feats["vol_20"] = ret.rolling(20).std()
    feats["momentum_60"] = prices / prices.shift(60) - 1.0
    feats["sma_gap"] = prices.rolling(10).mean() / prices.rolling(50).mean() - 1.0
    return feats


class MLSignalStrategy(Strategy):
    name = "ml_signal"

    def __init__(
        self,
        tickers: list[str],
        train_window: int = 252,
        retrain_every: int = 21,
        min_train_obs: int = 120,
        confidence_threshold: float = 0.55,
        random_state: int = 42,
    ):
        self.tickers = tickers
        self.train_window = train_window
        self.retrain_every = retrain_every
        self.min_train_obs = min_train_obs
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

    def _signals_for_ticker(self, prices: pd.Series) -> pd.Series:
        feats = _build_features(prices)
        # Label for day d: was the return from d to d+1 positive? Only ever
        # used as a training target for d < current bar, i.e. already realized.
        fwd_return = prices.pct_change().shift(-1)
        label = (fwd_return > 0).astype(float)

        signal = pd.Series(0.0, index=prices.index)
        model = None
        last_train_i = -10**9

        for i, date in enumerate(prices.index):
            row = feats.loc[date]
            if row.isna().any():
                continue

            if model is None or i - last_train_i >= self.retrain_every:
                train_start = max(0, i - self.train_window)
                train_idx = prices.index[train_start:i]  # strictly before today
                x_train = feats.loc[train_idx].dropna()
                y_train = label.loc[x_train.index].dropna()
                common = x_train.index.intersection(y_train.index)
                if len(common) >= self.min_train_obs and y_train.loc[common].nunique() >= 2:
                    model = HistGradientBoostingClassifier(
                        max_depth=3, random_state=self.random_state
                    )
                    model.fit(x_train.loc[common].to_numpy(), y_train.loc[common].to_numpy())
                    last_train_i = i
                elif model is None:
                    continue

            proba_up = model.predict_proba(row.to_numpy().reshape(1, -1))[0, 1]
            if proba_up > self.confidence_threshold or proba_up < 1 - self.confidence_threshold:
                signal.loc[date] = (proba_up - 0.5) * 2.0

        return signal.clip(-1.0, 1.0)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        out = {t: self._signals_for_ticker(prices[t]) for t in self.tickers if t in prices.columns}
        return pd.DataFrame(out, index=prices.index).fillna(0.0)
