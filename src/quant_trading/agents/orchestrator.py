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
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..data.ipo_scanner import fetch_recent_ipos
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
    is_recent_ipo: bool = False


async def _discover_recent_ipo_tickers(lookback_days: int, exclude: set[str]) -> list[str]:
    """Best-effort: a full text search hiccup here should degrade to "no
    newly discovered tickers", not take down the whole live report."""
    try:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=lookback_days)
        filings = await fetch_recent_ipos(start.isoformat(), end.isoformat())
    except Exception:
        return []
    return [f.ticker for f in filings if f.ticker and f.ticker not in exclude]


async def _load_prices_best_effort(tickers: list[str], price_start: str) -> pd.DataFrame | None:
    """Resolve newly-discovered tickers one at a time on failure, so a single
    bad/unmapped symbol from the IPO scanner can't sink the whole batch --
    unlike the core universe, these are unverified and expected to sometimes
    not resolve yet (e.g. priced but not yet trading on the data source)."""
    if not tickers:
        return None
    try:
        return await asyncio.to_thread(load_prices, tickers, price_start, cache_dir=None)
    except ValueError:
        pass

    frames = []
    for t in tickers:
        try:
            frames.append(await asyncio.to_thread(load_prices, [t], price_start, cache_dir=None))
        except ValueError:
            continue
    return pd.concat(frames, axis=1) if frames else None


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
    include_recent_ipos: bool = True,
    ipo_lookback_days: int = 180,
) -> list[TickerReport]:
    """Generate the live report for `tickers`, plus (by default) any company
    that completed a US IPO in the last `ipo_lookback_days` days and isn't
    already in `tickers` -- see data/ipo_scanner.py. Newly-public names will
    have too little history for the lookback-heavy strategies to say
    anything (they correctly report 0.0, not an error), but still get
    sentiment and insider-filing coverage.
    """
    analyzer = analyzer or default_sentiment_analyzer()
    base_tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order

    discovered: list[str] = []
    if include_recent_ipos:
        discovered = await _discover_recent_ipo_tickers(ipo_lookback_days, exclude=set(base_tickers))

    all_tickers = base_tickers + discovered

    base_prices, discovered_prices, headlines_by_ticker, insider_txns = await asyncio.gather(
        asyncio.to_thread(load_prices, base_tickers, price_start),
        _load_prices_best_effort(discovered, price_start),
        fetch_headlines(all_tickers),
        fetch_insider_transactions(all_tickers, start_date=price_start),
    )

    prices = (
        pd.concat([base_prices, discovered_prices], axis=1).sort_index()
        if discovered_prices is not None
        else base_prices
    )
    resolved_tickers = [t for t in all_tickers if t in prices.columns]
    unresolved = set(all_tickers) - set(resolved_tickers)

    daily_flow = build_daily_insider_flow(insider_txns, prices.index)

    sentiment_by_ticker, quant_signals = await asyncio.gather(
        _sentiment_agent(headlines_by_ticker, analyzer),
        asyncio.to_thread(_quant_signal_agent, prices, daily_flow, resolved_tickers),
    )

    cutoff = prices.index[-1] - pd.Timedelta(days=90)
    min_history_for_signal = 60  # below this, lookback-heavy strategies have nothing to say yet
    reports = []
    for t in resolved_tickers:
        recent_flow = (
            daily_flow.loc[daily_flow.index >= cutoff, t].sum() if t in daily_flow.columns else 0.0
        )
        score, rationale = sentiment_by_ticker.get(t, (0.0, "n/a"))
        days_of_history = int(prices[t].notna().sum())
        reports.append(
            TickerReport(
                ticker=t,
                sentiment_score=score,
                sentiment_rationale=rationale,
                recent_insider_net_flow=float(recent_flow),
                quant_signals=quant_signals.get(t, {}),
                headline_count=len(headlines_by_ticker.get(t, [])),
                is_recent_ipo=days_of_history < min_history_for_signal,
            )
        )
    if unresolved:
        print(f"[orchestrator] skipped {len(unresolved)} ticker(s) with no resolvable price data: {sorted(unresolved)}")
    return reports
