"""Capital-constrained trade selection as Maximum Weight Independent Set
(MWIS), run through QPhase's real QAOA pipeline on real trade opportunities
extracted from this repo's own backtested strategy signals.

MWIS is one of Karp's original 21 NP-complete problems and a standard QAOA
benchmark (Pichler, Wang, Zhou, Kok & Lukin, arXiv:1808.10816) -- this is
not an invented-for-this-demo problem class, it's textbook NP-hard
regardless of the specific instance.

Setup
-----
Each contiguous nonzero run of a strategy's executed position (signal,
lagged by one bar per the backtest engine's convention) on one ticker is
treated as a discrete "trade opportunity": a real historical entry/exit
window with a real realized profit. Two opportunities conflict (can't both
be taken) if either:

  (a) their time windows overlap -- capital tied up in one can't also
      fund the other, or
  (b) they're in the same sector bucket and within a short cooldown of
      each other -- a real, common risk-management practice (concentration
      limits), and deliberately included so the conflict graph isn't a
      pure interval graph. Pure time-overlap-only conflicts would form an
      interval graph, on which Maximum Weight Independent Set is solvable
      in polynomial time by a well-known specialized algorithm -- which
      would make this instance an accidentally-easy special case, not a
      genuine test of general MWIS solving. The sector-concentration rule
      breaks that structure honestly (it's a real constraint funds impose,
      not one invented to make the graph harder).

Given the resulting conflict graph, pick the profit-maximizing independent
set three ways: brute force (ground truth), QPhase's QAOA, and classical
simulated annealing (dimod/neal) -- same three-way comparison as
run_comparison.py, same honest scope: this validates QPhase's pipeline
against real data and ground truth, it does not claim a performance
advantage over classical methods at this problem size.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUANT_TRADING_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(QUANT_TRADING_SRC))

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "KO", "PEP"]
PAIR = ("KO", "PEP")
START = "2019-01-01"
SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "tech", "NVDA": "tech",
    "KO": "staples", "PEP": "staples",
}
SECTOR_COOLDOWN_DAYS = 10
MAX_OPPORTUNITIES = 14  # keeps brute-force + exact statevector simulation fast


async def build_backtests():
    from quant_trading.backtest.engine import run_many
    from quant_trading.data.filing_text import fetch_filing_drift
    from quant_trading.data.loaders import load_prices
    from quant_trading.data.sec_edgar import build_daily_insider_flow, fetch_insider_transactions
    from quant_trading.strategies import (
        FilingDriftStrategy,
        InsiderTradingStrategy,
        MeanReversionStrategy,
        MLSignalStrategy,
        MomentumStrategy,
        PairsTradingStrategy,
    )

    prices = load_prices(UNIVERSE, start=START)
    insider_txns = await fetch_insider_transactions(UNIVERSE, start_date=START)
    daily_flow = build_daily_insider_flow(insider_txns, prices.index)
    drift_df = await fetch_filing_drift(UNIVERSE, form_type="10-K", start_date=START)
    drift_events = drift_df.pivot(index="filing_date", columns="ticker", values="similarity")

    strategies = {
        "mean_reversion": MeanReversionStrategy(lookback=20, entry_z=1.0, exit_z=0.25),
        "momentum": MomentumStrategy(lookback=90, vol_lookback=20, skip=5),
        "pairs_trading": PairsTradingStrategy(*PAIR, lookback=60, entry_z=2.0, exit_z=0.5),
        "ml_signal": MLSignalStrategy(tickers=UNIVERSE, train_window=252, retrain_every=21),
        "insider_trading": InsiderTradingStrategy(daily_flow=daily_flow, lookback=90, z_lookback=252),
        "filing_drift": FilingDriftStrategy(drift_events=drift_events, z_lookback=4, entry_z=1.0, holding_days=60),
    }
    return run_many(prices, strategies, cost_bps=5.0), prices


def extract_opportunities(results, prices) -> pd.DataFrame:
    """One row per contiguous nonzero run of executed position -- a real
    historical entry/exit window with a real realized profit, computed
    exactly the way the backtest engine scores it (lagged signal x return)."""
    rows = []
    for strat_name, result in results.items():
        executed = result.positions  # already signal.shift(1), the engine's own executed book
        asset_returns = prices.pct_change().fillna(0.0)
        for ticker in executed.columns:
            pos = executed[ticker].to_numpy()
            rets = asset_returns[ticker].to_numpy()
            dates = executed.index
            i = 0
            n = len(pos)
            while i < n:
                if abs(pos[i]) < 1e-12:
                    i += 1
                    continue
                start = i
                sign = np.sign(pos[i])
                while i < n and np.sign(pos[i]) == sign and abs(pos[i]) > 1e-12:
                    i += 1
                end = i - 1
                profit = float((pos[start:end + 1] * rets[start:end + 1]).sum())
                rows.append(
                    {
                        "strategy": strat_name,
                        "ticker": ticker,
                        "sector": SECTOR[ticker],
                        "start": dates[start],
                        "end": dates[end],
                        "profit": profit,
                    }
                )
    return pd.DataFrame(rows)


def build_conflict_graph(opps: pd.DataFrame) -> list[tuple[int, int]]:
    conflicts = []
    n = len(opps)
    starts = opps["start"].to_numpy()
    ends = opps["end"].to_numpy()
    sectors = opps["sector"].to_numpy()
    cooldown = pd.Timedelta(days=SECTOR_COOLDOWN_DAYS)

    for i in range(n):
        for j in range(i + 1, n):
            time_overlap = starts[i] <= ends[j] and starts[j] <= ends[i]
            same_sector_near = sectors[i] == sectors[j] and (
                abs((starts[i] - starts[j])) <= cooldown
            )
            if time_overlap or same_sector_near:
                conflicts.append((i, j))
    return conflicts


def neal_cross_check(profits: np.ndarray, conflicts: list[tuple[int, int]]) -> str:
    import dimod
    from neal import SimulatedAnnealingSampler

    n = len(profits)
    lam = 1.0 + float(sum(p for p in profits if p > 0))
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for i in range(n):
        bqm.add_linear(f"x{i}", -float(profits[i]))
    for i, j in conflicts:
        bqm.add_quadratic(f"x{i}", f"x{j}", lam)
    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=200, seed=0)
    best = sampleset.first.sample
    return "".join(str(best[f"x{i}"]) for i in range(n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qphase-path", default=os.environ.get("QPHASE_PATH"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if not args.qphase_path:
        raise SystemExit("Pass --qphase-path /path/to/qphase or set QPHASE_PATH.")
    sys.path.insert(0, args.qphase_path)

    from qphase.backends.base import get_backend
    from qphase.problems.trade_selection import TradeSelectionInstance
    from qphase.runners.local import LocalSimulator
    from qphase.algorithms.optimizer import optimize_qaoa

    print("Building real backtests and extracting trade opportunities (network on first run, cached after)...")
    results, prices = asyncio.run(build_backtests())
    opps = extract_opportunities(results, prices)
    print(f"Extracted {len(opps)} real trade opportunities across {len(results)} strategies.")

    # Keep the largest-magnitude opportunities so brute-force + exact
    # statevector simulation stays fast -- log what's dropped rather than
    # silently truncating.
    if len(opps) > MAX_OPPORTUNITIES:
        opps = opps.reindex(opps["profit"].abs().sort_values(ascending=False).index)
        dropped = len(opps) - MAX_OPPORTUNITIES
        opps = opps.iloc[:MAX_OPPORTUNITIES].reset_index(drop=True)
        print(f"Dropped {dropped} smaller-magnitude opportunities to keep n <= {MAX_OPPORTUNITIES}.")
    else:
        opps = opps.reset_index(drop=True)

    print(opps[["strategy", "ticker", "start", "end", "profit"]].to_string())

    conflicts = build_conflict_graph(opps)
    profits = opps["profit"].to_numpy()
    names = tuple(f"{r.strategy}/{r.ticker}" for r in opps.itertuples())

    instance = TradeSelectionInstance(
        profits=tuple(profits), conflicts=tuple(conflicts),
        opportunity_names=names, name="real_trade_selection",
    )
    print(f"\n{instance.n_variables} opportunities, {len(conflicts)} conflict edges.")

    opt_bits, opt_val = instance.brute_force_optimum()
    opt_selected = [n for n, b in zip(names, opt_bits) if b == "1"]
    print(f"\nBrute-force optimum: {opt_val:.4f}  selected={opt_selected}")

    backend = get_backend("ionq", n_qubits=instance.n_variables)
    hits = 0
    for seed in range(args.seeds):
        runner = LocalSimulator(seed=seed)
        result = optimize_qaoa(instance, backend, runner, p=2, shots=2048, maxiter=80, seed=seed, sense="max")
        if abs(opt_val - result.best_objective) < 1e-9:
            hits += 1
    print(f"QPhase QAOA: matched optimum in {hits}/{args.seeds} seeds")

    neal_bits = neal_cross_check(profits, conflicts)
    neal_val = instance.objective(neal_bits)
    print(f"Classical (neal) simulated annealing: value={neal_val:.4f}  matches optimum={neal_bits == opt_bits}")

    print(
        "\nReminder: this validates QPhase's pipeline against ground truth on a real, if small, "
        "NP-hard instance -- it is not evidence of an advantage over classical solvers at this scale."
    )


if __name__ == "__main__":
    main()
