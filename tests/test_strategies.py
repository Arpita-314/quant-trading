import pandas as pd
import pytest

from quant_trading.strategies import (
    MeanReversionStrategy,
    MLSignalStrategy,
    MomentumStrategy,
    PairsTradingStrategy,
)

NON_ML_STRATEGIES = [
    MeanReversionStrategy(lookback=10, entry_z=0.5, exit_z=0.1),
    MomentumStrategy(lookback=30, vol_lookback=10, skip=2),
    PairsTradingStrategy("AAA", "BBB", lookback=20, entry_z=1.0, exit_z=0.25),
]


@pytest.mark.parametrize("strategy", NON_ML_STRATEGIES, ids=lambda s: s.name)
def test_signal_shape_and_bounds(strategy, synthetic_prices):
    signals = strategy.generate_signals(synthetic_prices)
    assert list(signals.columns) == list(synthetic_prices.columns)
    assert signals.index.equals(synthetic_prices.index)
    assert (signals.to_numpy() >= -1.0 - 1e-9).all()
    assert (signals.to_numpy() <= 1.0 + 1e-9).all()
    assert not signals.isna().any().any()


@pytest.mark.parametrize("strategy", NON_ML_STRATEGIES, ids=lambda s: s.name)
def test_no_lookahead_bias(strategy, synthetic_prices):
    """Signal at date t must not change if we hide all data after t.

    This is the core correctness property every strategy in this repo must
    satisfy: generate_signals(prices) may only use prices.loc[:t] to decide
    the value at t. We verify it directly rather than trusting the
    implementation, by truncating the input and checking the last row
    of signals is unchanged.
    """
    cutoff = 150
    truncated = synthetic_prices.iloc[: cutoff + 1]

    full_signals = strategy.generate_signals(synthetic_prices)
    truncated_signals = strategy.generate_signals(truncated)

    pd.testing.assert_series_equal(
        full_signals.iloc[cutoff],
        truncated_signals.iloc[cutoff],
        check_names=False,
    )


def test_ml_signal_no_lookahead_bias(synthetic_prices):
    strategy = MLSignalStrategy(
        tickers=list(synthetic_prices.columns),
        train_window=100,
        retrain_every=10,
        min_train_obs=30,
    )
    cutoff = 250
    truncated = synthetic_prices.iloc[: cutoff + 1]

    full_signals = strategy.generate_signals(synthetic_prices)
    truncated_signals = strategy.generate_signals(truncated)

    pd.testing.assert_series_equal(
        full_signals.iloc[cutoff],
        truncated_signals.iloc[cutoff],
        check_names=False,
    )


def test_ml_signal_shape_and_bounds(synthetic_prices):
    strategy = MLSignalStrategy(
        tickers=list(synthetic_prices.columns),
        train_window=100,
        retrain_every=10,
        min_train_obs=30,
    )
    signals = strategy.generate_signals(synthetic_prices)
    assert list(signals.columns) == list(synthetic_prices.columns)
    assert (signals.to_numpy() >= -1.0 - 1e-9).all()
    assert (signals.to_numpy() <= 1.0 + 1e-9).all()
