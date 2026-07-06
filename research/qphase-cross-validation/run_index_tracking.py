"""Sparse index tracking on real market data: which K of this repo's 7
tickers best replicate the equal-weighted return of all 7, using QPhase's
QAOA pipeline?

Cardinality-constrained sparse index tracking is a well-studied NP-hard
problem in the portfolio-construction literature -- structurally different
from the return-maximization objective in run_comparison.py's
PortfolioInstance test (this minimizes squared tracking error to a
benchmark, not maximizes risk-adjusted return), so it's a genuinely
different QUBO shape, not the same problem relabeled.

Same three-way, same honest scope as the other scripts here: brute force
(ground truth -- trivial at n=7 tickers), QPhase's QAOA, and classical
simulated annealing, with the same reminder that this validates
correctness on real data, not a performance claim.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

QUANT_TRADING_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(QUANT_TRADING_SRC))

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "KO", "PEP"]
START = "2019-01-01"


def neal_cross_check(returns: np.ndarray, benchmark: np.ndarray, k: int) -> str:
    import dimod
    from neal import SimulatedAnnealingSampler

    T, n = returns.shape
    M = (returns.T @ returns) / T
    c = (returns.T @ benchmark) / T
    lam = 20.0

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for i in range(n):
        bqm.add_linear(f"x{i}", M[i, i] / (k ** 2) - 2 * c[i] / k)
    for i in range(n):
        for j in range(i + 1, n):
            bqm.add_quadratic(f"x{i}", f"x{j}", 2 * M[i, j] / (k ** 2))
    terms = [(f"x{i}", 1.0) for i in range(n)]
    bqm.add_linear_equality_constraint(terms, lagrange_multiplier=lam, constant=-k)

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
    from qphase.problems.base import AlgorithmSpec
    from qphase.problems.index_tracking import IndexTrackingInstance
    from qphase.runners.local import LocalSimulator
    from qphase.algorithms.optimizer import optimize_qaoa

    from quant_trading.data.loaders import load_prices

    print(f"Loading real prices for {UNIVERSE} from {START}...")
    prices = load_prices(UNIVERSE, start=START)
    returns = prices.pct_change().dropna().to_numpy()
    benchmark = returns.mean(axis=1)  # the equal-weighted "index" of the full universe
    names = tuple(UNIVERSE)

    backend = get_backend("ionq", n_qubits=len(names))

    print(f"\n{'K':>3} {'brute-force best subset':<40} {'tracking MSE':<14} {'QAOA hits':<12} {'neal matches'}")
    for k in range(1, len(names)):
        instance = IndexTrackingInstance(
            returns=returns, benchmark=benchmark, cardinality=k, asset_names=names, name=f"K{k}"
        )
        opt_bits, opt_val = instance.brute_force_optimum()
        opt_selected = [n for n, b in zip(names, opt_bits) if b == "1"]

        algo = AlgorithmSpec(name="qaoa", p=2, warm_start="dicke", k_target=k, mixer="xy_ring", seed=0)
        hits = 0
        for seed in range(args.seeds):
            runner = LocalSimulator(seed=seed)
            result = optimize_qaoa(instance, backend, runner, p=2, shots=1024, maxiter=60, seed=seed, sense="min")
            if abs(opt_val - result.best_objective) < 1e-9:
                hits += 1

        neal_bits = neal_cross_check(returns, benchmark, k)
        neal_matches = neal_bits == opt_bits

        print(f"{k:>3} {str(opt_selected):<40} {opt_val:<14.8f} {f'{hits}/{args.seeds}':<12} {neal_matches}")

    print(
        "\nReminder: n=7 tickers is small enough that classical methods solve this just as reliably. "
        "This validates QPhase's pipeline against ground truth on real data, not a performance claim."
    )


if __name__ == "__main__":
    main()
