import numpy as np
import pandas as pd
import pytest

from quant_trading.utils.metrics import (
    cagr,
    compute_all_metrics,
    max_drawdown,
    sharpe_ratio,
)


def test_sharpe_zero_vol_returns_nan():
    returns = pd.Series([0.0] * 50)
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_positive_for_steady_gains():
    returns = pd.Series([0.001] * 252)
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_is_negative_or_zero():
    returns = pd.Series([0.01, -0.05, 0.02, -0.03, 0.01])
    mdd = max_drawdown(returns)
    assert mdd <= 0


def test_max_drawdown_known_value():
    # equity path: 1 -> 1.1 -> 0.99 -> drawdown from peak 1.1 to 0.99 is -10%
    returns = pd.Series([0.10, -0.10])
    assert max_drawdown(returns) == pytest.approx(-0.10)


def test_cagr_flat_returns_is_zero():
    returns = pd.Series([0.0] * 252)
    assert abs(cagr(returns)) < 1e-9


def test_compute_all_metrics_has_expected_keys():
    returns = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 100))
    metrics = compute_all_metrics(returns)
    expected_keys = {
        "cagr",
        "annualized_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "profit_factor",
    }
    assert expected_keys == set(metrics.keys())
