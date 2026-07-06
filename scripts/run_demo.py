"""End-to-end demo: download data, run every strategy, blend with the
adaptive ensemble agent, print a comparison table, and save an equity-curve
plot to outputs/equity_curves.png.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_trading.agents import AdaptiveEnsembleAgent, QuboEnsembleAgent
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

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "KO", "PEP"]
PAIR = ("KO", "PEP")  # classic cointegrated consumer-staples pair
START = "2019-01-01"
COST_BPS = 5.0


async def main() -> None:
    print(f"Downloading {UNIVERSE} from {START}...")
    prices = load_prices(UNIVERSE, start=START)
    print(f"Loaded {len(prices)} trading days, {prices.index.min().date()} -> {prices.index.max().date()}")

    print("Fetching real SEC Form 4 insider-transaction history (this hits the network)...")
    insider_txns = await fetch_insider_transactions(UNIVERSE, start_date=START)
    daily_flow = build_daily_insider_flow(insider_txns, prices.index)
    print(f"Loaded {len(insider_txns)} discretionary insider transactions across the universe")

    print("Fetching real 10-K filing text and computing filing-to-filing similarity...")
    drift_df = await fetch_filing_drift(UNIVERSE, form_type="10-K", start_date=START)
    drift_events = drift_df.pivot(index="filing_date", columns="ticker", values="similarity")
    print(f"Loaded {drift_df['similarity'].notna().sum()} filing-to-filing similarity scores")

    strategies = {
        "mean_reversion": MeanReversionStrategy(lookback=20, entry_z=1.0, exit_z=0.25),
        "momentum": MomentumStrategy(lookback=90, vol_lookback=20, skip=5),
        "pairs_trading": PairsTradingStrategy(*PAIR, lookback=60, entry_z=2.0, exit_z=0.5),
        "ml_signal": MLSignalStrategy(tickers=UNIVERSE, train_window=252, retrain_every=21),
        "insider_trading": InsiderTradingStrategy(daily_flow=daily_flow, lookback=90, z_lookback=252),
        "filing_drift": FilingDriftStrategy(drift_events=drift_events, z_lookback=4, entry_z=1.0, holding_days=60),
    }

    print("Running backtests...")
    results = run_many(prices, strategies, cost_bps=COST_BPS)

    strategy_returns = pd.DataFrame({name: r.returns for name, r in results.items()})

    sharpe_agent = AdaptiveEnsembleAgent(lookback=60, rebalance_every=21, max_weight=0.6)
    sharpe_ensemble_returns = sharpe_agent.combine(strategy_returns)

    qubo_agent = QuboEnsembleAgent(lookback=60, rebalance_every=21, num_reads=100, seed=42)
    qubo_ensemble_returns = qubo_agent.combine(strategy_returns)

    from quant_trading.utils.metrics import compute_all_metrics

    all_metrics = {name: r.metrics for name, r in results.items()}
    all_metrics["ensemble_agent_sharpe"] = compute_all_metrics(sharpe_ensemble_returns)
    all_metrics["ensemble_agent_qubo"] = compute_all_metrics(qubo_ensemble_returns)

    table = pd.DataFrame(all_metrics).T
    table = table[["cagr", "annualized_vol", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]]
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Performance comparison (net of costs) ===")
    print(table)

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, r in results.items():
            ax.plot(r.equity_curve.index, r.equity_curve.values, label=name, alpha=0.7)
        sharpe_equity = (1.0 + sharpe_ensemble_returns).cumprod()
        qubo_equity = (1.0 + qubo_ensemble_returns).cumprod()
        ax.plot(sharpe_equity.index, sharpe_equity.values, label="ensemble_agent_sharpe", linewidth=2.5, color="black")
        ax.plot(qubo_equity.index, qubo_equity.values, label="ensemble_agent_qubo", linewidth=2.5, color="tab:purple")
        ax.set_title("Strategy equity curves (net of transaction costs)")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        fig.tight_layout()
        fig_path = outputs_dir / "equity_curves.png"
        fig.savefig(fig_path, dpi=150)
        print(f"\nSaved plot to {fig_path}")
    except ImportError:
        print("\nmatplotlib not installed, skipping plot")


if __name__ == "__main__":
    asyncio.run(main())
