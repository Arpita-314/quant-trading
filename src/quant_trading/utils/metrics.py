"""Standard performance metrics for a daily returns series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    n_years = len(returns) / TRADING_DAYS
    if n_years <= 0 or equity.empty or equity.iloc[-1] <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    vol = excess.std(ddof=0)
    if not vol or np.isnan(vol):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if not dd or np.isnan(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan")


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if not mdd or np.isnan(mdd):
        return float("nan")
    return float(cagr(returns) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return float("nan")
    return float((nonzero > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def compute_all_metrics(returns: pd.Series) -> dict:
    return {
        "cagr": cagr(returns),
        "annualized_vol": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
    }
