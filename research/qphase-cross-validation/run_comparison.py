"""Cross-validate QPhase's real QAOA pipeline against this repo's own
strategy-allocation problem, using real backtested returns -- not a
synthetic toy problem invented for either project's own test suite.

What this actually demonstrates: QPhase's existing `PortfolioInstance`
ProblemIR (cardinality-constrained Markowitz selection, already built for
QPhase's fintech domain) correctly solves the "which K of this repo's 6
real strategies should get capital" problem, verified against the exact
brute-force optimum, across every cardinality and several random seeds.

What this does NOT demonstrate: any performance/speed advantage over
classical methods. n=6 (this repo's strategy count) is trivially small --
classical simulated annealing (`neal`, the same library used by this
repo's own `QuboEnsembleAgent`) finds the identical optimum just as
reliably at this scale. The honest claim here is functional correctness
on real, externally-sourced data, not quantum advantage. Anyone reporting
this as "QPhase beats classical optimization" would be overclaiming --
don't do that.

Requirements
------------
This script is NOT part of quant-trading's installable package, and QPhase
is NOT a public pip dependency (it's a separate, private repo) -- that's
why this lives under research/ rather than src/, and isn't exercised by CI.
To run it:

    1. `pip install -e .` this repo (for the `quant_trading` package)
    2. Have a local checkout of QPhase and pass its path via --qphase-path
       (or set the QPHASE_PATH environment variable)
    3. `pip install dimod dwave-neal` (for the classical cross-check)

    python run_comparison.py --qphase-path /path/to/qphase
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


async def compute_real_mu_sigma() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Reuses this repo's own backtest engine on real market/SEC data --
    the exact same strategies and numbers reported in the main README's
    Results section, not a hand-picked or synthetic example."""
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
    results = run_many(prices, strategies, cost_bps=5.0)
    strategy_returns = pd.DataFrame({name: r.returns for name, r in results.items()})

    mu = strategy_returns.mean().to_numpy() * 252
    sigma = strategy_returns.cov().to_numpy() * 252
    return mu, sigma, list(strategy_returns.columns)


def neal_cross_check(mu: np.ndarray, sigma: np.ndarray, k: int, risk_aversion: float) -> str:
    """Classical simulated-annealing baseline on the identical cardinality-
    constrained problem, for an honest side-by-side comparison."""
    import dimod
    from neal import SimulatedAnnealingSampler

    n = len(mu)
    lam = 20.0
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for i in range(n):
        bqm.add_linear(f"x{i}", -mu[i] + risk_aversion * sigma[i, i])
    for i in range(n):
        for j in range(i + 1, n):
            bqm.add_quadratic(f"x{i}", f"x{j}", 2 * risk_aversion * sigma[i, j])
    terms = [(f"x{i}", 1.0) for i in range(n)]
    bqm.add_linear_equality_constraint(terms, lagrange_multiplier=lam, constant=-k)

    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=200, seed=0)
    best = sampleset.first.sample
    return "".join(str(best[f"x{i}"]) for i in range(n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qphase-path", default=os.environ.get("QPHASE_PATH"))
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    if not args.qphase_path:
        raise SystemExit(
            "Pass --qphase-path /path/to/qphase or set QPHASE_PATH. "
            "QPhase is a separate, private repo -- not vendored here."
        )
    sys.path.insert(0, args.qphase_path)

    from qphase.backends.base import get_backend
    from qphase.problems.portfolio import PortfolioInstance
    from qphase.runners.local import LocalSimulator
    from qphase.algorithms.optimizer import optimize_qaoa

    print(f"Computing real mu/sigma from {UNIVERSE} strategies (this hits the network; cached after first run)...")
    mu, sigma, names = asyncio.run(compute_real_mu_sigma())
    print(f"Strategies: {names}\n")

    backend = get_backend("ionq", n_qubits=len(names))

    print(f"{'K':>3} {'brute-force optimum':<40} {'QAOA hits (of N seeds)':<24} {'neal matches optimum':<10}")
    for k in range(1, len(names)):
        instance = PortfolioInstance(
            mu=tuple(mu), sigma=sigma, cardinality=k,
            risk_aversion=args.risk_aversion, asset_names=tuple(names), name=f"K{k}",
        )
        opt_bits, opt_val = instance.brute_force_optimum()
        opt_selected = [n for n, b in zip(names, opt_bits) if b == "1"]

        hits = 0
        for seed in range(args.seeds):
            runner = LocalSimulator(seed=seed)
            result = optimize_qaoa(instance, backend, runner, p=2, shots=1024, maxiter=60, seed=seed, sense="max")
            if abs(opt_val - result.best_objective) < 1e-9:
                hits += 1

        neal_bits = neal_cross_check(mu, sigma, k, args.risk_aversion)
        neal_matches = neal_bits == opt_bits

        print(f"{k:>3} {opt_val:>8.4f} {str(opt_selected):<32} {f'{hits}/{args.seeds}':<24} {neal_matches!s:<10}")

    print(
        "\nReminder: n=6 is small enough that classical methods solve this just as reliably. "
        "This validates QPhase's real compile-and-run pipeline against ground truth on real "
        "financial data -- it is not evidence of a performance advantage over classical solvers."
    )


if __name__ == "__main__":
    main()
