"""Neural-network-predicted QAOA parameter initialization, evaluated
honestly against random initialization -- and the full pipeline this
research folder has been building toward: quantum-estimated tail risk
feeding a QAOA allocator, itself warm-started by a learned initializer.

Why this, not "reinventing" anything
-------------------------------------
Using a classical neural network to predict good variational parameters
for QAOA/VQE instead of random initialization is published, active
research (e.g. Verdon et al., "Learning to learn with quantum neural
networks via classical neural networks," 2019; several follow-ups on
meta-learning / transfer learning for QAOA parameters). What's built here
is a small, honestly-scoped instance of that idea, specific to this
repo's cardinality-constrained portfolio QUBO -- not a claim of a new
technique, and evaluated the same way everything else in this folder is:
against a classical baseline, with the result reported whatever it is.

What's evaluated
----------------
1. Generate ~200 synthetic 6-asset cardinality-constrained portfolio
   instances (random mu, random covariance, random cardinality K).
2. For each, run a *generous* QAOA optimization (many iterations) to get
   a high-quality (mu, sigma, K) -> theta label -- the "teacher" signal.
3. Train a small MLP (scikit-learn, a real feedforward neural network,
   not gradient boosting) to predict theta directly from problem features.
4. On held-out synthetic instances AND on this repo's own real
   tail-risk-aware portfolio problem: compare final objective quality
   after a *short, fixed* optimization budget, starting from the NN's
   prediction vs. from random initialization. Short budget specifically
   because a learned initializer's entire value proposition is doing
   better with LESS classical optimization, not eventually converging to
   the same place either way given enough iterations.

This is not a proof that the technique is real-world-ready. It is a
concrete, honestly-measured answer to "does a learned initializer help,
on this specific problem family, at this scale" -- which is reported
either way, not tuned until the answer looks good.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor

QUANT_TRADING_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(QUANT_TRADING_SRC))

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "KO", "PEP"]
PAIR = ("KO", "PEP")
START = "2019-01-01"
N_ASSETS = 6  # fixed across the synthetic corpus, matching this repo's real strategy count
P_ROUNDS = 2
N_PARAMS = 2 * P_ROUNDS

SHORT_BUDGET_MAXITER = 15  # the whole point: does the NN help with LESS optimization
TEACHER_MAXITER = 150      # generous budget used only to generate training labels


def featurize(mu: np.ndarray, sigma: np.ndarray, k: int) -> np.ndarray:
    """Fixed-size summary features -- mean/std of returns, mean/std of
    variance and covariance terms, and normalized cardinality. Summary
    statistics rather than the raw flattened matrix, so the featurizer
    doesn't silently depend on N_ASSETS staying exactly 6."""
    diag = np.diag(sigma)
    off_diag = sigma[~np.eye(len(sigma), dtype=bool)]
    return np.array([
        mu.mean(), mu.std(),
        diag.mean(), diag.std(),
        off_diag.mean(), off_diag.std(),
        k / len(mu),
    ])


def random_instance(rng: np.random.Generator, qphase_portfolio_cls, n=N_ASSETS):
    mu = rng.normal(0.05, 0.08, n)
    A = rng.normal(0, 1, (n, n))
    sigma = (A @ A.T) / n * rng.uniform(0.01, 0.1)
    k = int(rng.integers(1, n))
    return qphase_portfolio_cls(mu=tuple(mu), sigma=sigma, cardinality=k, risk_aversion=3.0), mu, sigma, k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qphase-path", default=os.environ.get("QPHASE_PATH"))
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.qphase_path:
        raise SystemExit("Pass --qphase-path /path/to/qphase or set QPHASE_PATH.")
    sys.path.insert(0, args.qphase_path)

    from qphase.backends.base import get_backend
    from qphase.problems.base import AlgorithmSpec
    from qphase.problems.portfolio import PortfolioInstance
    from qphase.problems.risk_aware_portfolio import RiskAwarePortfolioInstance
    from qphase.problems.tail_risk import TailRiskInstance
    from qphase.runners.local import LocalSimulator
    from qphase.algorithms.optimizer import optimize_qaoa

    rng = np.random.default_rng(args.seed)
    backend = get_backend("ionq", n_qubits=N_ASSETS)

    def solve(instance, k, theta0=None, maxiter=TEACHER_MAXITER, seed=0):
        algo = AlgorithmSpec(name="qaoa", p=P_ROUNDS, warm_start="dicke", k_target=k, mixer="xy_ring", seed=seed)
        runner = LocalSimulator(seed=seed)
        return optimize_qaoa(
            instance, backend, runner, p=P_ROUNDS, shots=512, maxiter=maxiter, seed=seed,
            sense="max", theta0=theta0,
        )

    # ── Step 1+2: generate synthetic corpus + teacher labels ──────
    print(f"Generating {args.n_train + args.n_test} synthetic instances and solving each "
          f"with a generous budget ({TEACHER_MAXITER} iterations) for training labels...")
    X, Y = [], []
    for i in range(args.n_train + args.n_test):
        instance, mu, sigma, k = random_instance(rng, PortfolioInstance)
        result = solve(instance, k, maxiter=TEACHER_MAXITER, seed=i)
        X.append(featurize(mu, sigma, k))
        Y.append(result.best_theta)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.n_train + args.n_test} instances solved")

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test = X[: args.n_train], X[args.n_train:]
    Y_train, Y_test = Y[: args.n_train], Y[args.n_train:]

    # ── Step 3: train the neural network ──────────────────────────
    print(f"\nTraining MLPRegressor on {len(X_train)} instances...")
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=args.seed)
    mlp.fit(X_train, Y_train)

    # ── Step 4: honest evaluation under a SHORT budget ─────────────
    print(f"\nEvaluating on {len(X_test)} held-out synthetic instances, "
          f"short budget ({SHORT_BUDGET_MAXITER} iterations)...")
    nn_better, random_better, ties = 0, 0, 0
    for i in range(args.n_train, args.n_train + args.n_test):
        instance, mu, sigma, k = random_instance(np.random.default_rng(1000 + i), PortfolioInstance)
        features = featurize(mu, sigma, k).reshape(1, -1)
        theta_nn = mlp.predict(features)[0]

        result_nn = solve(instance, k, theta0=theta_nn, maxiter=SHORT_BUDGET_MAXITER, seed=i)
        result_random = solve(instance, k, theta0=None, maxiter=SHORT_BUDGET_MAXITER, seed=i)

        if result_nn.best_objective > result_random.best_objective + 1e-9:
            nn_better += 1
        elif result_random.best_objective > result_nn.best_objective + 1e-9:
            random_better += 1
        else:
            ties += 1

    print(f"\nSynthetic held-out results (n={args.n_test}, short budget={SHORT_BUDGET_MAXITER} iters):")
    print(f"  NN-initialized better: {nn_better}")
    print(f"  Random-initialized better: {random_better}")
    print(f"  Tied: {ties}")

    # ── Tie it all together: real data, tail-risk-aware, NN-warm-started ──
    print("\nNow the full pipeline on real data: quantum tail-risk + NN-warm-started QAOA "
          "on this repo's actual 6 strategies...")

    async def build_real_instance():
        from quant_trading.backtest.engine import run_many
        from quant_trading.data.filing_text import fetch_filing_drift
        from quant_trading.data.loaders import load_prices
        from quant_trading.data.sec_edgar import build_daily_insider_flow, fetch_insider_transactions
        from quant_trading.strategies import (
            FilingDriftStrategy, InsiderTradingStrategy, MeanReversionStrategy,
            MLSignalStrategy, MomentumStrategy, PairsTradingStrategy,
        )
        import pandas as pd

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
        return strategy_returns

    strategy_returns = asyncio.run(build_real_instance())
    names = list(strategy_returns.columns)
    mu_real = strategy_returns.mean().to_numpy() * 252
    sigma_real = strategy_returns.cov().to_numpy() * 252

    tail_runner = LocalSimulator(seed=0)
    tail_risks = []
    for name in names:
        tr_instance = TailRiskInstance(returns=strategy_returns[name].dropna().to_numpy())
        tail_risks.append(tr_instance.expected_downside_qae(tail_runner, schedule=[0, 1, 2, 4, 8], shots_per_power=2000))
    print(f"Quantum-estimated tail risk per strategy: {dict(zip(names, [round(t, 5) for t in tail_risks]))}")

    real_instance = RiskAwarePortfolioInstance(
        mu=tuple(mu_real), sigma=sigma_real, tail_risk=tuple(tail_risks), cardinality=3,
        cov_risk_aversion=3.0, tail_risk_aversion=5.0, asset_names=tuple(names),
    )
    opt_bits, opt_val = real_instance.brute_force_optimum()
    opt_selected = [n for n, b in zip(names, opt_bits) if b == "1"]
    print(f"\nBrute-force optimum (K=3, real data): {opt_val:.4f}  selected={opt_selected}")

    features_real = featurize(mu_real, sigma_real, 3).reshape(1, -1)
    theta_nn_real = mlp.predict(features_real)[0]
    real_backend = get_backend("ionq", n_qubits=len(names))
    algo = AlgorithmSpec(name="qaoa", p=P_ROUNDS, warm_start="dicke", k_target=3, mixer="xy_ring", seed=0)

    result_nn_real = optimize_qaoa(
        real_instance, real_backend, LocalSimulator(seed=0), p=P_ROUNDS, shots=1024,
        maxiter=SHORT_BUDGET_MAXITER, seed=0, sense="max", theta0=theta_nn_real,
    )
    result_random_real = optimize_qaoa(
        real_instance, real_backend, LocalSimulator(seed=0), p=P_ROUNDS, shots=1024,
        maxiter=SHORT_BUDGET_MAXITER, seed=0, sense="max",
    )
    print(f"NN-warm-started QAOA (short budget): {result_nn_real.best_bitstring} "
          f"obj={result_nn_real.best_objective:.4f} feasible={result_nn_real.feasible}")
    print(f"Random-initialized QAOA (short budget): {result_random_real.best_bitstring} "
          f"obj={result_random_real.best_objective:.4f} feasible={result_random_real.feasible}")
    print(f"Matches true optimum -- NN: {abs(result_nn_real.best_objective - opt_val) < 1e-6}, "
          f"random: {abs(result_random_real.best_objective - opt_val) < 1e-6}")


if __name__ == "__main__":
    main()
