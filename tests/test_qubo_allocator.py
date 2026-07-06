import numpy as np
import pandas as pd

from quant_trading.agents import QuboEnsembleAgent


def _synthetic_strategy_returns(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    good = rng.normal(0.001, 0.01, n)
    bad = rng.normal(-0.002, 0.02, n)
    redundant_good = 0.9 * good + 0.1 * rng.normal(0.001, 0.01, n)
    return pd.DataFrame({"good": good, "bad": bad, "redundant_good": redundant_good}, index=dates)


def test_allocate_weights_sum_to_one_every_row():
    returns = _synthetic_strategy_returns()
    agent = QuboEnsembleAgent(lookback=60, rebalance_every=30, num_reads=50, seed=1)
    weights = agent.allocate(returns)
    row_sums = weights.sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_allocate_weights_are_nonnegative():
    """The QUBO's budget/one-hot constraints are soft (Lagrange penalties),
    not hard -- the annealer isn't guaranteed to land exactly on a
    raw-grid combination that already sums to 1, so weights are
    renormalized afterward and won't generally still sit on the original
    discrete grid. Non-negativity is the property that must always hold."""
    returns = _synthetic_strategy_returns()
    agent = QuboEnsembleAgent(lookback=60, rebalance_every=30, num_reads=50, seed=1)
    weights = agent.allocate(returns)
    assert (weights.to_numpy() >= 0.0).all()


def test_allocate_avoids_the_negative_mean_strategy():
    """The 'bad' strategy has a strongly negative trailing mean return --
    a sane allocator should put little to no weight on it most of the time."""
    returns = _synthetic_strategy_returns()
    agent = QuboEnsembleAgent(lookback=60, rebalance_every=30, num_reads=100, seed=2)
    weights = agent.allocate(returns)
    rebalance_rows = weights.iloc[30::30]
    assert (rebalance_rows["bad"] <= 0.25).mean() >= 0.8


def test_combine_produces_a_return_series():
    returns = _synthetic_strategy_returns()
    agent = QuboEnsembleAgent(lookback=60, rebalance_every=30, num_reads=50, seed=3)
    combined = agent.combine(returns)
    assert len(combined) == len(returns)
    assert not combined.isna().any()


def test_allocate_is_causal_no_lookahead():
    """Weights decided at a rebalance date must depend only on the trailing
    window strictly before it -- verified by truncating the return series
    after the last rebalance point and checking the decision is unchanged."""
    returns = _synthetic_strategy_returns(n=200)
    agent = QuboEnsembleAgent(lookback=60, rebalance_every=30, num_reads=50, seed=4)

    full_weights = agent.allocate(returns)
    cutoff = 150  # a rebalance date (multiple of 30)
    truncated_weights = agent.allocate(returns.iloc[: cutoff + 5])

    pd.testing.assert_series_equal(
        full_weights.iloc[cutoff],
        truncated_weights.iloc[cutoff],
        check_names=False,
    )
