# quant-trading

[![tests](https://github.com/Arpita-314/quant-trading/actions/workflows/tests.yml/badge.svg)](https://github.com/Arpita-314/quant-trading/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A research-grade, multi-strategy quantitative trading toolkit: six signal
families (statistical arbitrage, mean reversion, momentum, ML-driven
direction prediction, real SEC-filing-based insider trading, and a
filing-language-drift signal from actual 10-K text), a proper vectorized
backtester, two capital allocators (a rolling-Sharpe agent and a
quantum-inspired QUBO optimizer), and an async multi-agent live signal
report modeled on the same dispatch-and-gather pattern behind modern coding
agents (Cursor-style background agents).

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
    filing_text.py            # async 10-K/10-Q text fetch + TF-IDF similarity ("Lazy Prices")
    news_scraper.py          # async concurrent headline scraper (live use only, see below)
    ipo_scanner.py            # async SEC EDGAR full-text search for recently-completed US IPOs
  strategies/
    base.py                # Strategy interface + the causality contract
    mean_reversion.py      # rolling z-score, per-asset
    momentum.py            # time-series momentum, inverse-vol sized, 12-1 skip
    pairs_trading.py        # rolling-hedge-ratio spread z-score + state machine
    ml_signal.py            # walk-forward HistGradientBoosting on engineered features
    insider_trading.py       # real historical SEC Form 4 buy/sell flow -> position
    filing_drift.py           # filing-language-change signal from real 10-K/10-Q text
  backtest/engine.py       # execution lag, gross-exposure risk overlay, costs, metrics
  agents/
    ensemble_allocator.py  # rolling-Sharpe adaptive capital allocation
    qubo_allocator.py         # quantum-inspired (QUBO) mean-variance capital allocation
    sentiment.py             # pluggable sentiment scoring (deterministic default, optional LLM)
    orchestrator.py          # async multi-agent live signal dispatcher (see below)
  utils/metrics.py          # Sharpe, Sortino, Calmar, drawdown, etc.
  utils/validation.py       # walk-forward (never shuffled) train/test splits
scripts/
  run_demo.py             # end-to-end backtest: download, backtest, blend, plot
  run_live_agents.py        # live, as-of-today multi-agent signal report (not a backtest)
tests/                      # includes explicit lookahead-bias regression tests
research/                    # standalone research pieces, independent of the src/ package
  options-mispricing-nn/       # pure-numpy NN predicting BS mispricing from vol surfaces
  qphase-cross-validation/      # real-data correctness check against an external QAOA compiler
```

`research/` holds self-contained research projects that don't share the
`quant_trading` package's dependencies or conventions -- each has its own
`requirements.txt` and README rather than being folded into the main
package, since forcing an unrelated project into a shared namespace just to
keep one repo would make both harder to read.

`qphase-cross-validation/` takes this repo's own real strategy-allocation
problem (the same K-of-6 cardinality-constrained selection the QUBO
allocator above solves) and runs it through a separate quantum-computing
compiler project (QPhase) via its existing QAOA pipeline, comparing the
result against the exact brute-force optimum and against classical
simulated annealing. It matched the optimum 25/25 times across every
cardinality and 5 seeds each -- a genuine, verifiable **correctness**
result on real data. It is explicitly **not** a performance claim: at 6
strategies, classical simulated annealing solves the same problem just as
reliably, and the README there says so directly. A second script poses a
genuinely NP-hard variant (Maximum Weight Independent Set, one of Karp's
original 21 NP-complete problems) over 6,325 real historical trade
opportunities extracted from this repo's own strategies; there, QPhase's
QAOA matched the optimum in 4 of 5 seeds rather than a clean sweep --
the expected, honest behavior of a heuristic with no convergence
guarantee. QPhase isn't a public dependency, so neither script is part of
CI or the installable package -- see the folder's own README for how to
run them against a local QPhase checkout.

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

## The ensemble agent(s)

`AdaptiveEnsembleAgent` reallocates capital across the strategies on a
fixed cadence (default: every 21 trading days), weighting each by its
trailing rolling Sharpe ratio and capping any single sleeve at 60% of
capital. It's a deterministic, fully auditable rule — not an LLM deciding
trades. That's deliberate: a fund needs an allocation rule that's
backtestable with a fixed seed and explainable to a risk committee, not one
that "sounds like AI."

### A second allocator: quantum-inspired portfolio optimization (QUBO)

`QuboEnsembleAgent` (`agents/qubo_allocator.py`) is a genuinely different
allocation *method*, not a rebrand of the same idea. Real quantum-finance
research (D-Wave, IBM, and bank quantum-computing groups) formulates
portfolio selection as a Quadratic Unconstrained Binary Optimization
(QUBO) problem — the standard input format for a quantum annealer.
This builds that exact formulation (Markowitz mean-variance: maximize
`w'mu - risk_aversion * w'Sigma*w` subject to weights summing to 1) as a
`dimod.BinaryQuadraticModel`, with each strategy's weight discretized onto
a grid (0%, 25%, 50%, 75%, 100%) and one-hot encoded into binary variables,
since QUBO variables are binary and continuous weights aren't directly
representable. It's solved here via `neal.SimulatedAnnealingSampler` —
D-Wave's open-source *classical* simulated-annealing proxy for a quantum
annealer, since no quantum hardware is used — but the same `BinaryQuadraticModel`
could be submitted to a real D-Wave annealer via `DWaveSampler` with zero
change to the objective itself. That's the actual point of building it this
way instead of just calling it "quantum" as decoration: the formulation is
hardware-ready today, whether or not a quantum backend is ever attached.

The methodological difference from `AdaptiveEnsembleAgent` is real:
trailing-Sharpe ranking scores each strategy independently and has no way
to represent "these two strategies are redundant," while the QUBO's
covariance term does — two individually-decent strategies that move
together contribute less diversification value than their Sharpes alone
would suggest, and the optimizer can see that. See
[Results](#results) for whether that theoretical advantage actually shows
up in this sample.

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

## A second alternative-data signal: filing-language drift ("Lazy Prices")

`FilingDriftStrategy` (`strategies/filing_drift.py`) implements a real,
peer-reviewed strategy: Cohen, Malloy & Nguyen, ["Lazy Prices"](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12943)
(*Journal of Finance*, 2020). The idea: measure how much a company's
*language* changes between consecutive annual/quarterly filings (10-K to
10-K, or 10-Q to 10-Q). An unusually large change predicts negative
subsequent returns -- the paper's argument is that markets underreact to
lengthy, boring textual disclosures that few investors read closely enough
to notice have changed.

The similarity measure is cosine similarity on a TF-IDF term-frequency
vector space -- that's not a simplified stand-in for the paper's method,
it's literally the measure the paper uses. `data/filing_text.py` fetches
real 10-K/10-Q HTML via SEC EDGAR (async, cached, same retry/failure-count
discipline as the insider-trading client) and strips it to plain text
before comparing. Two things about that extraction were not obvious going
in, and both were real bugs before they were fixed:

1. **Modern filings use inline XBRL**, meaning financial facts are tagged
   directly in the HTML, with large blocks of pure machine-readable
   metadata hidden via `style="display:none"`. A plain `.get_text()` call
   doesn't know about CSS visibility and happily returns that hidden
   metadata as if it were prose -- silently drowning genuine language
   changes in inline-XBRL tag noise rather than erroring. Hidden elements
   are stripped before extracting text.
2. **A transient fetch failure landing on whichever ticker happens to be
   processed first is not the same as "no filing that year."** An early
   version fetched one ticker's filings fully before moving to the next,
   sequentially -- a rate-limit window that happened to land on the first
   ticker in the list silently blanked out that one company's entire
   similarity history while every other ticker, processed later on the far
   side of the window, succeeded normally. Fixed the same way as the
   insider-trading client: fetch across every ticker together, retry
   failures in sweeps with a real pause between them, and print an explicit
   warning with a failure count if anything still doesn't come back.

Because a company's own filing-to-filing history is short (roughly 4 events
a year) and this repo's universe is only 7 tickers -- nowhere near enough
for the paper's cross-sectional decile sorts to mean anything -- the signal
here compares each filing's language change to *that company's own*
historical baseline (a rolling z-score over its trailing 4-8 filings)
rather than ranking across the universe. That's a smaller, more modest test
of the same underlying idea, not a full replication of the paper's
methodology.

Unlike news headlines, filing dates are real historical timestamps, so
(unlike the sentiment strategy below) this one is honestly backtestable
and included in the comparison table.

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

### Recent-IPO discovery

By default the live report also pulls in any company that completed a US
IPO in the last 180 days, via `data/ipo_scanner.py` -- a fourth async agent
querying SEC EDGAR's full text search for Form 424B4 (the *final* IPO
prospectus, filed when a deal actually prices and starts trading, as
opposed to Form S-1, which only signals intent and includes plenty of
deals that get withdrawn or delayed indefinitely). SPAC/blank-check shells
are filtered out by SIC code, since "recent IPOs" here means operating
companies, not empty acquisition vehicles.

`SPCX` (SpaceX, IPO'd 2026-06-12) and `CBRS` (Cerebras, IPO'd 2026-05-14)
are included by default as an explicit fallback in case that scanner call
is ever unavailable -- both have too little price history (~15-35 trading
days) for the lookback-heavy strategies to say anything statistically
meaningful, and the live report flags them as such rather than presenting
noise as signal. This is a live-report-only addition; they are not part of
the backtest universe in `scripts/run_demo.py`, where a handful of noisy
days would just add random variance to the comparison table without
teaching anything.

A ticker that shows up in the SEC scan but doesn't yet resolve on the price
data source (priced but not yet trading, or a data-source lag) is dropped
rather than taking down the whole report -- see
`orchestrator._load_prices_best_effort`.

**Jio Platforms (India)** is explicitly out of scope: as of this writing it
had filed a draft prospectus with India's SEBI but had not yet listed, and
would list on NSE/BSE once it does -- a different regulator entirely. This
repo's insider-trading pipeline is SEC-EDGAR-specific; covering Jio would
need a separate SEBI/NSE data client, not just adding a ticker to a list.

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
at 1.0x. `insider_trading` uses the real, complete SEC Form 4 history for
this universe (12,191 discretionary transactions); `filing_drift` uses real
10-K filing text (47 filing-to-filing comparisons).

| strategy | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Win rate |
|---|---|---|---|---|---|---|---|
| mean_reversion | -11.8% | 21.9% | -0.46 | -0.65 | -64.5% | -0.18 | 46.9% |
| momentum | -0.1% | 16.0% | 0.08 | 0.09 | -30.4% | 0.00 | 51.6% |
| pairs_trading (KO/PEP) | -3.3% | 7.8% | -0.39 | -0.34 | -25.3% | -0.13 | 47.7% |
| ml_signal | -2.2% | 17.3% | -0.04 | -0.06 | -30.1% | -0.07 | 48.4% |
| insider_trading | **13.4%** | 19.4% | **0.75** | 0.91 | -28.5% | 0.47 | 52.3% |
| filing_drift | -1.2% | 3.8% | -0.31 | -0.07 | -18.0% | -0.07 | 46.0% |
| ensemble_agent_sharpe | -5.2% | 15.2% | -0.27 | -0.32 | -40.0% | -0.13 | 48.8% |
| ensemble_agent_qubo | -3.5% | 12.6% | -0.22 | -0.27 | -30.3% | -0.12 | 49.6% |

Equity curves: [`outputs/equity_curves.png`](outputs/equity_curves.png) (generated by `scripts/run_demo.py`; gitignored, regenerate locally).

### What the numbers actually show

- **Momentum is the only textbook sleeve with a (barely) positive Sharpe**,
  which tracks: 2019-2026 was a strong secular bull run for this exact
  mega-cap tech basket, punctuated by one sharp 2022 drawdown — close to
  the regime where 12-1 trend following has historically had a real, if
  modest, edge.
- **Mean reversion loses money** because it's fighting the trend: a
  single-asset rolling z-score has no way to distinguish "temporary
  dislocation" from "this stock is re-rating," and mega-cap tech spent this
  period re-rating.
- **Pairs trading loses money for a diagnosable reason.** KO/PEP is
  frequently cited as a textbook cointegrated pair. Running this repo's own
  `engle_granger_pvalue(KO, PEP)` over the actual backtest window returns
  **p = 1.0** — i.e. no statistical evidence of cointegration in this
  sample. The strategy lost money trading a mean-reversion premise that
  didn't hold, and the toolkit's own pair-selection helper would have
  flagged that *before* going live.
- **`insider_trading`'s strong headline Sharpe (0.75) needs a big asterisk:
  it's substantially a single-stock effect, not broad-based signal.** NVDA
  alone accounts for ~44% of the strategy's total return contribution —
  unsurprising given NVDA returned roughly +5,700% over this window.
  Re-running the identical strategy on the same universe *minus NVDA* drops
  Sharpe from 0.75 to 0.47 and CAGR from 13.4% to 6.6% — and critically,
  **that ex-NVDA Sharpe (0.47) is still worse than simply buying and holding
  the same six remaining stocks equally weighted (Sharpe 1.04, CAGR
  21.5%)**. So excluding the one name that happened to have a historic run,
  the insider-buying signal did not beat doing nothing. The honest
  conclusion is that this run mostly demonstrates the strategy correctly
  went long NVDA some of the time during an extraordinary bull run, not that
  insider buying is predictive in this sample.
- **`filing_drift`'s -1.2% CAGR is not meaningful evidence either way — the
  strategy only actually fired twice in seven years** (once on GOOGL, once
  on PEP), because a 7-ticker universe has nowhere near enough filing events
  for the "Lazy Prices" effect to show up statistically. The original paper
  tests this cross-sectionally across thousands of firms; two independent
  bets prove essentially nothing. This is the right honest read of a small
  sample, not "the strategy doesn't work."
- **Both ensemble agents underperform a naive equal-weight blend of all six
  sleeves** (equal-weight Sharpe: ~0.02 vs. -0.27 for the trailing-Sharpe
  agent and -0.22 for the QUBO agent). The QUBO allocator is consistently a
  little better than trailing-Sharpe ranking — consistent with it being the
  only one of the two that accounts for covariance between sleeves — but
  neither escapes the same underlying problem: with `insider_trading`'s
  edge concentrated in one lucky name and most other sleeves flat-to-negative
  in this sample, there isn't a robust cross-sleeve signal for *either*
  allocator to find. A smarter allocator can't manufacture edge that the
  underlying sleeves don't have.

None of this means the framework is broken — the lookahead-bias and
execution-lag regression tests pass, and the mechanics are verified. It
means most of these strategies, on a small and heavily-arbitraged universe,
don't have much edge left in them once you look one level deeper than the
headline Sharpe ratio — which is exactly the level recruiters at real funds
will look, and exactly why real funds spend money on data, execution, and
breadth (hundreds to thousands of instruments) that this demo doesn't have.

## What a real edge would require

- A much larger, less crowded cross-section (small/mid-cap, futures,
  international, or higher-frequency intraday data) instead of 7 mega-caps
  everyone already trades -- this would also let `filing_drift` run the
  paper's actual cross-sectional decile-sort methodology instead of a
  firm-specific baseline, and give it enough independent events to say
  something statistically meaningful
- Per-name position caps, so one stock's outsized run can't single-handedly
  carry (or sink) a signal's headline numbers the way NVDA did here
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

GPL v3.
