"""European option pricing via quantum amplitude estimation, using a real
historical volatility estimate rather than a made-up number.

This is a genuinely different algorithm family from every other script in
this folder. Portfolio selection, trade selection, and index tracking are
all QAOA -- a NISQ-native heuristic with no proven speedup, which is why
every other README section here says "correctness, not a performance
claim." Quantum amplitude estimation (Brassard-Høyer-Mosca-Tapp 2002) is
provably better than classical Monte Carlo in the fault-tolerant regime it
targets: O(1/M) pricing error in M oracle calls, vs. classical Monte
Carlo's O(1/sqrt(N)) in N samples. That's not re-derived here -- doing so
honestly needs a scale no classical simulator can reach, and pretending
otherwise at 2 qubits would undercut the point rather than make it. What
IS validated here: does the simulated circuit recover the right price.

See qphase/problems/option_pricing.py's module docstring for the
technique (Suzuki et al. 2020's "amplitude estimation without phase
estimation" -- no QPE, no inverse QFT, no extra phase register) and the
explicit qubit-count scope decision (price register capped at 2 qubits /
4 discretized levels for this build).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

QUANT_TRADING_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(QUANT_TRADING_SRC))

TICKER = "AAPL"
START = "2019-01-01"
VOL_LOOKBACK_DAYS = 252
RISK_FREE_RATE = 0.03  # a representative constant; not the point of this demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qphase-path", default=os.environ.get("QPHASE_PATH"))
    args = parser.parse_args()
    if not args.qphase_path:
        raise SystemExit("Pass --qphase-path /path/to/qphase or set QPHASE_PATH.")
    sys.path.insert(0, args.qphase_path)

    from qphase.problems.option_pricing import EuropeanOptionInstance
    from qphase.runners.local import LocalSimulator

    from quant_trading.data.loaders import load_prices

    print(f"Loading real {TICKER} prices from {START} for a realized-volatility estimate...")
    prices = load_prices([TICKER], start=START)
    daily_returns = prices[TICKER].pct_change().dropna()
    realized_vol = float(daily_returns.tail(VOL_LOOKBACK_DAYS).std() * np.sqrt(252))
    s0 = float(prices[TICKER].iloc[-1])
    print(f"{TICKER} spot: {s0:.2f}   trailing {VOL_LOOKBACK_DAYS}-day realized vol: {realized_vol:.4f}")

    runner = LocalSimulator(seed=0)

    print(f"\n{'strike':>8} {'moneyness':>10} {'Black-Scholes':>14} {'Monte Carlo':>12} {'QAE (simulated)':>16} {'QAE rel.err':>12}")
    for moneyness in (0.85, 0.95, 1.00, 1.05, 1.15):
        strike = round(s0 * moneyness, 2)
        instance = EuropeanOptionInstance(
            s0=s0, strike=strike, r=RISK_FREE_RATE, sigma=realized_vol, t=1.0, kind="call",
        )
        bs = instance.black_scholes_price()
        mc = instance.monte_carlo_price(n_samples=200_000, seed=0)
        qae = instance.qae_price(runner, schedule=[0, 1, 2, 4, 8, 16], shots_per_power=4000, seed=0)
        rel_err = abs(qae - bs) / bs * 100 if bs > 1e-9 else float("nan")
        print(f"{strike:>8.2f} {moneyness:>10.2f} {bs:>14.4f} {mc:>12.4f} {qae:>16.4f} {rel_err:>11.1f}%")

    print(
        "\nReminder: error grows away from at-the-money because 4 discretization levels "
        "is genuinely coarse -- an inherent cost of this build's qubit-count scope, not a bug. "
        "More qubits (finer discretization) is exactly what the fault-tolerant regime this "
        "algorithm targets would provide; nothing here claims to demonstrate that at this scale."
    )


if __name__ == "__main__":
    main()
