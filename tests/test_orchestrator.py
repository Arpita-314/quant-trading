import pandas as pd
import pytest

from quant_trading.agents import orchestrator


@pytest.mark.asyncio
async def test_load_prices_best_effort_drops_bad_ticker(monkeypatch):
    """A single unresolved ticker from the IPO scanner (e.g. priced but not
    yet trading on the data source) must not sink the whole batch -- this
    locks in the fallback-to-per-ticker-fetch behavior."""
    good = pd.DataFrame({"GOOD": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2026-01-01", periods=3))

    def fake_load_prices(tickers, price_start, cache_dir=None):
        if tickers == ["GOOD", "BAD"]:
            raise ValueError("No price data returned for: ['BAD']")
        if tickers == ["GOOD"]:
            return good
        if tickers == ["BAD"]:
            raise ValueError("No price data returned for: ['BAD']")
        raise AssertionError(f"unexpected call: {tickers}")

    monkeypatch.setattr(orchestrator, "load_prices", fake_load_prices)

    result = await orchestrator._load_prices_best_effort(["GOOD", "BAD"], "2026-01-01")

    assert list(result.columns) == ["GOOD"]
    assert len(result) == 3


@pytest.mark.asyncio
async def test_load_prices_best_effort_all_bad_returns_none(monkeypatch):
    def fake_load_prices(tickers, price_start, cache_dir=None):
        raise ValueError("No price data returned")

    monkeypatch.setattr(orchestrator, "load_prices", fake_load_prices)

    result = await orchestrator._load_prices_best_effort(["BAD1", "BAD2"], "2026-01-01")

    assert result is None


@pytest.mark.asyncio
async def test_load_prices_best_effort_empty_input_returns_none():
    result = await orchestrator._load_prices_best_effort([], "2026-01-01")
    assert result is None


@pytest.mark.asyncio
async def test_discover_recent_ipo_tickers_excludes_and_filters_none(monkeypatch):
    class _Filing:
        def __init__(self, ticker):
            self.ticker = ticker

    async def fake_fetch_recent_ipos(start, end):
        return [_Filing("NEWCO"), _Filing(None), _Filing("ALREADYIN")]

    monkeypatch.setattr(orchestrator, "fetch_recent_ipos", fake_fetch_recent_ipos)

    result = await orchestrator._discover_recent_ipo_tickers(180, exclude={"ALREADYIN"})

    assert result == ["NEWCO"]


@pytest.mark.asyncio
async def test_discover_recent_ipo_tickers_degrades_to_empty_on_error(monkeypatch):
    async def fake_fetch_recent_ipos(start, end):
        raise RuntimeError("SEC full text search is down")

    monkeypatch.setattr(orchestrator, "fetch_recent_ipos", fake_fetch_recent_ipos)

    result = await orchestrator._discover_recent_ipo_tickers(180, exclude=set())

    assert result == []
