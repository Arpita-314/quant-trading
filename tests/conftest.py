import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Deterministic synthetic GBM price paths for 3 tickers, ~300 trading days."""
    rng = np.random.default_rng(seed=7)
    n_days = 300
    tickers = ["AAA", "BBB", "CCC"]
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    data = {}
    for i, ticker in enumerate(tickers):
        drift = 0.0002 * (i + 1)
        vol = 0.01 + 0.002 * i
        shocks = rng.normal(loc=drift, scale=vol, size=n_days)
        prices = 100.0 * np.exp(np.cumsum(shocks))
        data[ticker] = prices

    return pd.DataFrame(data, index=dates)
