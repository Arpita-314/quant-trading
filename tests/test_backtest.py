import numpy as np
import pandas as pd

from quant_trading.agents import AdaptiveEnsembleAgent
from quant_trading.backtest.engine import run_backtest, run_many
from quant_trading.strategies.base import Strategy


class _AlwaysLong(Strategy):
    name = "always_long"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


class _PerfectForesight(Strategy):
    """Sizes today's position using TODAY's own return -- this is a lookahead
    bug if the engine didn't lag execution. Used to prove the engine's lag
    is actually applied."""

    name = "perfect_foresight"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        todays_return = prices.pct_change()
        return np.sign(todays_return).fillna(0.0)


def test_run_backtest_result_shapes(synthetic_prices):
    result = run_backtest(synthetic_prices, _AlwaysLong(), cost_bps=0.0)
    assert len(result.returns) == len(synthetic_prices)
    assert len(result.equity_curve) == len(synthetic_prices)
    assert set(result.metrics.keys()) >= {"sharpe", "cagr", "max_drawdown"}


def test_always_long_matches_summed_asset_returns(synthetic_prices):
    """The engine sums per-asset position*return contributions -- signals are
    per-asset conviction weights, not portfolio-normalized weights, so 1.0 on
    every asset means 100% notional in each. Uses max_gross_exposure=None to
    check the raw, uncapped mechanics (the exposure cap is tested separately)."""
    result = run_backtest(synthetic_prices, _AlwaysLong(), cost_bps=0.0, max_gross_exposure=None)
    asset_returns = synthetic_prices.pct_change().fillna(0.0)
    expected = asset_returns.sum(axis=1)
    expected.iloc[0] = 0.0  # first bar has no executed position yet (signal not lagged in)
    pd.testing.assert_series_equal(result.returns, expected, check_names=False)


def test_max_gross_exposure_caps_book_leverage(synthetic_prices):
    result = run_backtest(synthetic_prices, _AlwaysLong(), cost_bps=0.0, max_gross_exposure=1.0)
    gross = result.positions.abs().sum(axis=1)
    assert (gross <= 1.0 + 1e-9).all()


def test_execution_lag_prevents_perfect_foresight_profit(synthetic_prices):
    """If a strategy could trade on today's own return with zero lag, it
    would post an implausibly large, guaranteed profit. The engine's one-bar
    lag must neutralize that -- this is the regression test for the most
    dangerous class of backtest bug (lookahead in the engine itself)."""
    result = run_backtest(synthetic_prices, _PerfectForesight(), cost_bps=0.0, max_gross_exposure=None)
    naive_no_lag_returns = (
        np.sign(synthetic_prices.pct_change()).fillna(0.0) * synthetic_prices.pct_change().fillna(0.0)
    ).sum(axis=1)
    # With the lag applied, actual returns must NOT equal the (impossible) no-lag returns.
    assert not np.allclose(result.returns.to_numpy(), naive_no_lag_returns.to_numpy())


def test_transaction_costs_reduce_returns_for_high_turnover(synthetic_prices):
    class _Flippy(Strategy):
        name = "flippy"

        def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
            n = len(prices)
            alt = np.tile([1.0, -1.0], n // 2 + 1)[:n]
            return pd.DataFrame(
                {col: alt for col in prices.columns}, index=prices.index
            )

    no_cost = run_backtest(synthetic_prices, _Flippy(), cost_bps=0.0, max_gross_exposure=None)
    with_cost = run_backtest(synthetic_prices, _Flippy(), cost_bps=50.0, max_gross_exposure=None)
    assert with_cost.returns.sum() < no_cost.returns.sum()


def test_run_many_returns_one_result_per_strategy(synthetic_prices):
    strategies = {"long": _AlwaysLong(), "foresight": _PerfectForesight()}
    results = run_many(synthetic_prices, strategies, cost_bps=1.0)
    assert set(results.keys()) == set(strategies.keys())


def test_ensemble_agent_weights_sum_to_one(synthetic_prices):
    strategies = {"long": _AlwaysLong(), "foresight": _PerfectForesight()}
    results = run_many(synthetic_prices, strategies, cost_bps=1.0)
    strategy_returns = pd.DataFrame({name: r.returns for name, r in results.items()})

    agent = AdaptiveEnsembleAgent(lookback=20, rebalance_every=10)
    weights = agent.allocate(strategy_returns)
    row_sums = weights.sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), 1.0, atol=1e-9)


def test_ensemble_agent_combine_produces_series(synthetic_prices):
    strategies = {"long": _AlwaysLong(), "foresight": _PerfectForesight()}
    results = run_many(synthetic_prices, strategies, cost_bps=1.0)
    strategy_returns = pd.DataFrame({name: r.returns for name, r in results.items()})

    agent = AdaptiveEnsembleAgent(lookback=20, rebalance_every=10)
    combined = agent.combine(strategy_returns)
    assert len(combined) == len(strategy_returns)
    assert not combined.isna().any()
