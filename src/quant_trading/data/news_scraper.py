"""Async concurrent headline scraper.

Yahoo Finance's per-ticker RSS feed only ever returns the current set of
recent headlines -- there is no free, point-in-time historical archive of
"what the news said on 2021-03-04" to backtest against. That's a real data
limitation, not a bug: this module is for the *live* signal-generation path
(`scripts/run_live_agents.py`), not for historical backtesting. Fetches for
every ticker run concurrently via asyncio -- this is the same "many
independent workers, gathered at the end" shape as an async coding-agent
dispatcher.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import aiohttp

RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
HEADERS = {"User-Agent": "Mozilla/5.0 (quant-trading-research)"}


@dataclass
class Headline:
    ticker: str
    title: str
    published: str
    link: str


async def _fetch_one(session: aiohttp.ClientSession, ticker: str, limit: int) -> list[Headline]:
    url = RSS_URL.format(ticker=ticker)
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except Exception:
        return []

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    headlines = []
    for item in root.findall(".//item")[:limit]:
        title_el = item.find("title")
        date_el = item.find("pubDate")
        link_el = item.find("link")
        if title_el is None or title_el.text is None:
            continue
        headlines.append(
            Headline(
                ticker=ticker,
                title=title_el.text,
                published=date_el.text if date_el is not None else "",
                link=link_el.text if link_el is not None else "",
            )
        )
    return headlines


async def fetch_headlines(tickers: list[str], limit_per_ticker: int = 10) -> dict[str, list[Headline]]:
    """Fetch recent headlines for every ticker concurrently."""
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[_fetch_one(session, t, limit_per_ticker) for t in tickers]
        )
    return dict(zip(tickers, results))
