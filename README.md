# quant-trading

[![tests](https://github.com/Arpita-314/quant-trading/actions/workflows/tests.yml/badge.svg)](https://github.com/Arpita-314/quant-trading/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade, multi-strategy quantitative trading toolkit: five signal
families (statistical arbitrage, mean reversion, momentum, ML-driven
direction prediction, and real SEC-filing-based insider trading), a proper
vectorized backtester, an adaptive "agent" that reallocates capital across
strategies based on trailing risk-adjusted performance, and an async
multi-agent live signal report modeled on the same dispatch-and-gather
pattern behind modern coding agents (Cursor-style background agents).

**What this is:** a demonstration of correct quant-research methodology —
causal signal generation, walk-forward ML validation, an honest backtest
engine, and a multi-strategy allocator — built to show how this kind of
system should be engineered.

**What this is NOT:** a recreation of Renaissance Technologies' Medallion
Fund. Medallion's actual signals, data sources, and execution stack are
trade secrets; nobody outside the firm has ever seen them. Nothing here is
investment advice, and nothing here trades real money — it's backtesting
and paper-trading infrastructure only.

## Why the results aren't dressed up

The strategies here are textbook implementations (rolling z-score mean
reversion, time-series momentum, Engle-Granger pairs trading, a gradient-
boosted directional classifier), run unmodified against a liquid mega-cap
equity universe that's picked over by every quant shop on the street. They
were **not** tuned against this backtest until the numbers looked good —
that's the single most common way "backtested alpha" turns out to be
curve-fit noise, and it's exactly the failure mode a quant recruiter will
probe for. So the results below are mixed, some strategies lose money, and
that's the honest, expected outcome for unmodified textbook signals on an
efficient, widely-traded universe. See [Results](#results) for what the
numbers actually show and why.

## Architecture

```
src/quant_trading/
  data/
    loaders.py             # yfinance -> cached parquet -> wide price DataFrame
    sec_edgar.py            # async SEC EDGAR client: real, point-in-time Form 4 filings
    news_scraper.py          # async concurrent headline scraper (live use only, see below)
  strategies/
    base.py                # Strategy interface + the causality contract
    mean_reversion.py      # rolling z-score, per-asset
    momentum.py            # time-series momentum, inverse-vol sized, 12-1 skip
    pairs_trading.py        # rolling-hedge-ratio spread z-score + state machine
    ml_signal.py            # walk-forward HistGradientBoosting on engineered features
    insider_trading.py       # real historical SEC Form 4 buy/sell flow -> position
  backtest/engine.py       # execution lag, gross-exposure risk overlay, costs, metrics
  agents/
    ensemble_allocator.py  # rolling-Sharpe adaptive capital allocation
    sentiment.py             # pluggable sentiment scoring (deterministic default, optional LLM)
    orchestrator.py          # async multi-agent live signal dispatcher (see below)
  utils/metrics.py          # Sharpe, Sortino, Calmar, drawdown, etc.
  utils/validation.py       # walk-forward (never shuffled) train/test splits
scripts/
  run_demo.py             # end-to-end backtest: download, backtest, blend, plot
  run_live_agents.py        # live, as-of-today multi-agent signal report (not a backtest)
tests/                      # includes explicit lookahead-bias regression tests
```

### The causality contract (why this isn't a toy backtester)

Every strategy's `generate_signals(prices)` may only use `prices.loc[:t]` to
decide the position at `t`. The backtest engine is the **only** place that
applies the one-bar execution lag (`signals.shift(1)`) before multiplying by
returns — so no strategy can accidentally trade on its own same-day return.
This is enforced, not just documented: [`tests/test_strategies.py`](tests/test_strategies.py)
truncates the price history and asserts each strategy's signal at the last
visible bar is bit-identical whether or not future data exists, and
[`tests/test_backtest.py`](tests/test_backtest.py) includes a strategy that
*would* show a guaranteed, impossible profit if the engine's lag weren't
applied — a regression test for the most common and most dangerous class of
backtest bug.

### Risk overlay

Strategies emit an independent per-asset conviction weight in `[-1, 1]`. On
a day when several assets in the universe fire at once, summed across the
book that can exceed 100% notional. The engine applies a `max_gross_exposure`
cap (default 1.0, i.e. no leverage) that scales the whole book down
proportionally on days it's breached — a standard risk-management layer,
not a strategy-level hack.

### ML signal: walk-forward, not fit-once

`MLSignalStrategy` refits a `HistGradientBoostingClassifier` every N bars
using only the trailing window of data strictly before the current bar, then
scores the current bar and moves on. There's no separate "fit on everything,
backtest on the same data" step — that fusion of the walk-forward validation
loop and the live signal loop is what makes the walk-forward-ness structural
rather than a claim.

## The ensemble agent

`AdaptiveEnsembleAgent` reallocates capital across the four strategies on a
fixed cadence (default: every 21 trading days), weighting each by its
trailing rolling Sharpe ratio and capping any single sleeve at 60% of
capital. It's a deterministic, fully auditable rule — not an LLM deciding
trades. That's deliberate: a fund needs an allocation rule that's
backtestable with a fixed seed and explainable to a risk committee, not one
that "sounds like AI."

## Alternative data: a real insider-trading signal

`InsiderTradingStrategy` is built from actual SEC Form 4 filings, fetched
concurrently via `data/sec_edgar.py` -- not news headlines, not a synthetic
proxy. This matters because Form 4 filings carry a real historical filing
date going back years, unlike news (see below), so this signal is honestly
backtestable rather than only usable live.

Two things about the SEC feed that were not obvious going in, and both
turned into real bugs that had to be fixed before the data could be trusted:

1. **The filing-list API's `primaryDocument` field points at the
   XSLT-*rendered* HTML view, not the raw parseable XML.** Naively
   requesting that path and feeding it to an XML parser silently produced
   zero transactions for every filing (caught, misleadingly, by a
   `ParseError` handler) rather than an error. The raw XML sits at the same
   accession folder under just the basename with any `xslF345X0*/` prefix
   stripped.
2. **A transient network failure and "this filing has zero relevant
   transactions" are not the same event**, but a naive `except Exception:
   return []` treats them identically -- silently manufacturing an insider-
   flow history with unannounced gaps that look like "no insider activity"
   rather than "the fetch failed." The client now retries with backoff and
   surfaces a warning with an explicit failure count instead of eating the
   difference.

Only transaction codes `P` (open-market purchase) and `S` (open-market sale)
are kept from each filing's non-derivative transaction table. Codes like `M`
(option/RSU exercise) and `F` (tax-withholding disposal) are administrative
vesting mechanics, not discretionary decisions -- including them would drown
the real signal in noise, which is a common mistake in naive insider-trading
signals. The strategy is long-only by default: insider *selling* is largely
non-discretionary in practice (10b5-1 pre-scheduled plans, tax
diversification), while insider *buying* is almost always a voluntary,
information-bearing decision.

## Async agents: the "Cursor for trading signals" piece

The live signal path (`agents/orchestrator.py`, `scripts/run_live_agents.py`)
is structured the way modern async coding agents are: dispatch several
independent workers concurrently, gather their results, then synthesize --
rather than one linear script that scrapes news, *then* waits, *then*
fetches filings, *then* waits, *then* computes signals. Here, price loading,
headline scraping, and SEC filing fetches all run concurrently via a single
`asyncio.gather`; CPU-bound quant-strategy scoring runs in a thread pool via
`asyncio.to_thread` alongside them instead of blocking. Sentiment scoring is
pluggable behind one interface (`agents/sentiment.py`): a deterministic
VADER-based lexicon by default (free, reproducible, zero setup), or an
actual LLM call if `ANTHROPIC_API_KEY` is set, for headlines where keyword
scoring misreads context.

```bash
python scripts/run_live_agents.py AAPL MSFT NVDA
```

**This is explicitly not a backtest.** Yahoo Finance's headline feed (and
most free news sources) only ever returns the *current* set of recent
headlines -- there is no free, point-in-time archive of "what the news said
on 2021-03-04" to backtest a sentiment strategy against. Faking one by
applying today's headlines across historical dates would be lookahead bias
dressed up as a feature. So the news+sentiment path is live-report-only; the
SEC insider-trading path, which does carry real historical dates, is the one
wired into the backtest suite above.

## Results

Backtest: `AAPL, MSFT, GOOGL, AMZN, NVDA, KO, PEP`, 2019-01-02 to 2026-07-02
(1,885 trading days), 5 bps cost per unit of turnover, gross exposure capped
at 1.0x.

> `insider_trading` is wired into `scripts/run_demo.py` and unit-tested, but
> a full historical SEC pull across this 7-ticker universe is thousands of
> HTTP calls and doesn't fit a quick CI run -- the results row below will be
> added once that data pull finishes locally. Run `python scripts/run_demo.py`
> yourself to generate it (results cache to `data_cache/` so it's a one-time cost).

| strategy | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Win rate |
|---|---|---|---|---|---|---|---|
| mean_reversion | -11.8% | 21.9% | -0.46 | -0.60 | -64.5% | -0.18 | 46.9% |
| momentum | -0.1% | 16.0% | 0.08 | 0.11 | -30.4% | 0.00 | 51.6% |
| pairs_trading (KO/PEP) | -3.3% | 7.8% | -0.39 | -0.51 | -25.3% | -0.13 | 47.7% |
| ml_signal | -2.2% | 17.3% | -0.04 | -0.06 | -30.1% | -0.07 | 48.4% |
| **ensemble_agent** | -8.6% | 14.8% | -0.53 | -0.72 | -51.9% | -0.17 | 47.9% |

Equity curves: [`outputs/equity_curves.png`](outputs/equity_curves.png) (generated by `scripts/run_demo.py`; gitignored, regenerate locally).

### What the numbers actually show

- **Momentum is the only sleeve with a (barely) positive Sharpe**, which
  tracks: 2019-2026 was a strong secular bull run for this exact mega-cap
  tech basket, punctuated by one sharp 2022 drawdown — close to the textbook
  regime where 12-1 trend following has historically had a real, if modest, edge.
- **Mean reversion loses money** because it's fighting the trend: a
  single-asset rolling z-score has no way to distinguish "temporary
  dislocation" from "this stock is re-rating," and mega-cap tech spent this
  period re-rating.
- **Pairs trading loses money for a diagnosable reason.** KO/PEP is
  frequently cited as a textbook cointegrated pair. Running this repo's own
  `engle_granger_pvalue(KO, PEP)` over the actual backtest window returns
  **p = 1.0** — i.e. no statistical evidence of cointegration in this
  sample. Cointegration is not a permanent property of two tickers; it has
  to be tested on the window you intend to trade, not assumed from a
  textbook example. The strategy lost money trading a mean-reversion premise
  that didn't hold, and the toolkit's own pair-selection helper would have
  flagged that *before* going live.
- **The ensemble agent underperforms even a naive equal-weight blend of the
  same four sleeves** (equal-weight Sharpe: -0.33 vs. agent Sharpe: -0.53,
  and the gap persists across lookback/rebalance settings from 60/21 to
  252/63 days tested). With only four sleeves, three of which have no real
  edge in this sample, chasing trailing 60-252-day Sharpe estimates has
  nothing solid to converge toward — it just adds turnover and a tendency to
  overweight whichever sleeve most recently had a lucky run right before
  mean-reverting. This is a known, real risk in tactical strategy-timing
  (performance-chasing is frequently anti-persistent), reproduced here
  rather than hidden.

None of this means the framework is broken — the lookahead-bias and
execution-lag regression tests pass, and the mechanics are verified. It
means four untuned textbook strategies on a heavily-arbitraged large-cap
universe don't have much edge left in them, which is exactly what you'd
expect and exactly why real funds spend money on data, execution, and
breadth (hundreds to thousands of instruments) that this demo doesn't have.

## What a real edge would require

- A much larger, less crowded cross-section (small/mid-cap, futures,
  international, or higher-frequency intraday data) instead of 7 mega-caps
  everyone already trades
- Proper walk-forward hyperparameter search with a held-out final test
  window, not just walk-forward *fitting*
- Alternative/orthogonal data (order flow, options-implied signals,
  fundamentals) rather than price-only features
- A real fill/slippage model instead of a linear turnover-cost proxy
- Live pair re-screening (cointegration is regime-dependent, as shown above)

## Running it

```bash
pip install -e ".[dev]"
pytest tests/ -q          # includes the lookahead-bias regression tests
python scripts/run_demo.py
```

## License

MIT.
