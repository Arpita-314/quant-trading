"""Live, as-of-today multi-agent signal report -- NOT a backtest.

Dispatches a news-scraping agent, a SEC insider-filing agent, an IPO-scanner
agent, and price loading concurrently via asyncio, scores sentiment
(deterministic by default, or an LLM if ANTHROPIC_API_KEY is set), and
blends with each strategy's current signal. See `agents/orchestrator.py`
for why this is a live report and not something backtested.

By default this also pulls in any company that completed a US IPO in the
last 180 days (via SEC EDGAR full text search, SPAC shells excluded) --
SPCX (SpaceX) and CBRS (Cerebras) are included explicitly below as a
fallback in case that scanner call is ever unavailable, since both IPO'd
recently and have too little price history (~15-35 days) to say anything
statistically meaningful yet -- this is a live-report-only add, not part of
the backtest universe in scripts/run_demo.py.

Note on scope: Jio Platforms (India) has filed to go public but hadn't
listed as of this repo's last update, and would list on NSE/BSE under
India's SEBI, not the US SEC -- this repo's insider-trading pipeline is
SEC-EDGAR-specific and doesn't cover non-US regulators.

Usage:
    python scripts/run_live_agents.py AAPL MSFT GOOGL
    python scripts/run_live_agents.py   # defaults to the repo's core universe + recent IPOs
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_trading.agents.orchestrator import generate_live_report

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "KO", "PEP", "SPCX", "CBRS"]


async def main(tickers: list[str]) -> None:
    print(f"Running live agents for {tickers} + recently-IPO'd names (news + SEC filings + quant signals, concurrently)...")
    reports = await generate_live_report(tickers)

    for r in reports:
        tag = " [recent IPO -- thin history, signals below are mostly noise]" if r.is_recent_ipo else ""
        print(f"\n=== {r.ticker}{tag} ===")
        print(f"Sentiment: {r.sentiment_score:+.2f}  ({r.sentiment_rationale})  [{r.headline_count} headlines]")
        print(f"Insider net flow, trailing 90d: ${r.recent_insider_net_flow:,.0f}")
        for name, sig in r.quant_signals.items():
            print(f"  {name:<16} signal={sig:+.3f}")


if __name__ == "__main__":
    tickers = sys.argv[1:] or DEFAULT_TICKERS
    asyncio.run(main(tickers))
