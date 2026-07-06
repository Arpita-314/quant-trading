"""Live, as-of-today multi-agent signal report -- NOT a backtest.

Dispatches a news-scraping agent, a SEC insider-filing agent, and price
loading concurrently via asyncio, scores sentiment (deterministic by
default, or an LLM if ANTHROPIC_API_KEY is set), and blends with each
strategy's current signal. See `agents/orchestrator.py` for why this is a
live report and not something backtested.

Usage:
    python scripts/run_live_agents.py AAPL MSFT GOOGL
    python scripts/run_live_agents.py   # defaults to the repo's 7-ticker universe
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_trading.agents.orchestrator import generate_live_report

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "KO", "PEP"]


async def main(tickers: list[str]) -> None:
    print(f"Running live agents for {tickers} (news + SEC filings + quant signals, concurrently)...")
    reports = await generate_live_report(tickers)

    for r in reports:
        print(f"\n=== {r.ticker} ===")
        print(f"Sentiment: {r.sentiment_score:+.2f}  ({r.sentiment_rationale})  [{r.headline_count} headlines]")
        print(f"Insider net flow, trailing 90d: ${r.recent_insider_net_flow:,.0f}")
        for name, sig in r.quant_signals.items():
            print(f"  {name:<16} signal={sig:+.3f}")


if __name__ == "__main__":
    tickers = sys.argv[1:] or DEFAULT_TICKERS
    asyncio.run(main(tickers))
