"""End-to-end demo: download data, run every strategy, blend with the
adaptive ensemble agent, print a comparison table, and save an equity-curve
plot to outputs/equity_curves.png.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_trading.agents import AdaptiveEnsembleAgent
from quant_trading.backtest.engine import run_many
from quant_trading.data.loaders import load_prices
from quant_trading.strategies import (
    MeanReversionStrategy,
    MLSignalStrategy,
    MomentumStrategy,
    PairsTradingStrategy,
)

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "KO", "PEP"]
PAIR = ("KO", "PEP")  # classic cointegrated consumer-staples pair
START = "2019-01-01"
COST_BPS = 5.0


def main() -> None:
    print(f"Downloading {UNIVERSE} from {START}...")
    prices = load_prices(UNIVERSE, start=START)
    print(f"Loaded {len(prices)} trading days, {prices.index.min().date()} -> {prices.index.max().date()}")

    strategies = {
        "mean_reversion": MeanReversionStrategy(lookback=20, entry_z=1.0, exit_z=0.25),
        "momentum": MomentumStrategy(lookback=90, vol_lookback=20, skip=5),
        "pairs_trading": PairsTradingStrategy(*PAIR, lookback=60, entry_z=2.0, exit_z=0.5),
        "ml_signal": MLSignalStrategy(tickers=UNIVERSE, train_window=252, retrain_every=21),
    }

    print("Running backtests...")
    results = run_many(prices, strategies, cost_bps=COST_BPS)

    strategy_returns = pd.DataFrame({name: r.returns for name, r in results.items()})
    agent = AdaptiveEnsembleAgent(lookback=60, rebalance_every=21, max_weight=0.6)
    ensemble_returns = agent.combine(strategy_returns)

    from quant_trading.utils.metrics import compute_all_metrics

    all_metrics = {name: r.metrics for name, r in results.items()}
    all_metrics["ensemble_agent"] = compute_all_metrics(ensemble_returns)

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
        ensemble_equity = (1.0 + ensemble_returns).cumprod()
        ax.plot(ensemble_equity.index, ensemble_equity.values, label="ensemble_agent", linewidth=2.5, color="black")
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
    main()
