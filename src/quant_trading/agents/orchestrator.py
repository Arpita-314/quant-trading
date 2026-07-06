"""Async multi-agent live signal orchestrator.

Mirrors the pattern behind async coding agents (Cursor-style background
agents): dispatch several independent workers concurrently, gather their
results, then synthesize -- rather than one linear script that scrapes news,
*then* waits, *then* fetches filings, *then* waits, *then* computes signals.
Here a news-scraping agent, a SEC insider-filing agent, and price loading all
run concurrently via `asyncio.gather`. Quant-strategy scoring is CPU-bound
and synchronous, so it runs in a thread pool via `asyncio.to_thread`
alongside the I/O-bound agents instead of blocking them.

This produces a live, as-of-today report -- it is NOT a backtest. See
`data/news_scraper.py` for why headline data can't be backtested for free.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pandas as pd

from ..data.loaders import load_prices
from ..data.news_scraper import fetch_headlines
from ..data.sec_edgar import build_daily_insider_flow, fetch_insider_transactions
from ..strategies import InsiderTradingStrategy, MeanReversionStrategy, MomentumStrategy
from .sentiment import SentimentAnalyzer, default_sentiment_analyzer


@dataclass
class TickerReport:
    ticker: str
    sentiment_score: float
    sentiment_rationale: str
    recent_insider_net_flow: float
    quant_signals: dict
    headline_count: int


async def _sentiment_agent(
    headlines_by_ticker: dict, analyzer: SentimentAnalyzer
) -> dict[str, tuple[float, str]]:
    tickers = list(headlines_by_ticker.keys())
    results = await asyncio.gather(
        *[analyzer.score([h.title for h in headlines_by_ticker[t]]) for t in tickers]
    )
    return dict(zip(tickers, results))


def _quant_signal_agent(prices: pd.DataFrame, daily_flow: pd.DataFrame, tickers: list[str]) -> dict:
    """Synchronous, CPU-bound: today's signal from each strategy that can
    run standalone (pairs trading and the ML signal need extra config, so
    the live report sticks to the strategies that work out of the box)."""
    strategies = {
        "mean_reversion": MeanReversionStrategy(),
        "momentum": MomentumStrategy(),
        "insider_trading": InsiderTradingStrategy(daily_flow=daily_flow),
    }
    out: dict[str, dict] = {t: {} for t in tickers}
    for name, strat in strategies.items():
        signals = strat.generate_signals(prices)
        latest = signals.iloc[-1]
        for t in tickers:
            if t in latest.index:
                out[t][name] = float(latest[t])
    return out


async def generate_live_report(
    tickers: list[str],
    price_start: str = "2019-01-01",
    analyzer: SentimentAnalyzer | None = None,
) -> list[TickerReport]:
    analyzer = analyzer or default_sentiment_analyzer()

    prices, headlines_by_ticker, insider_txns = await asyncio.gather(
        asyncio.to_thread(load_prices, tickers, price_start),
        fetch_headlines(tickers),
        fetch_insider_transactions(tickers, start_date=price_start),
    )

    daily_flow = build_daily_insider_flow(insider_txns, prices.index)

    sentiment_by_ticker, quant_signals = await asyncio.gather(
        _sentiment_agent(headlines_by_ticker, analyzer),
        asyncio.to_thread(_quant_signal_agent, prices, daily_flow, tickers),
    )

    cutoff = prices.index[-1] - pd.Timedelta(days=90)
    reports = []
    for t in tickers:
        recent_flow = (
            daily_flow.loc[daily_flow.index >= cutoff, t].sum() if t in daily_flow.columns else 0.0
        )
        score, rationale = sentiment_by_ticker.get(t, (0.0, "n/a"))
        reports.append(
            TickerReport(
                ticker=t,
                sentiment_score=score,
                sentiment_rationale=rationale,
                recent_insider_net_flow=float(recent_flow),
                quant_signals=quant_signals.get(t, {}),
                headline_count=len(headlines_by_ticker.get(t, [])),
            )
        )
    return reports
