"""Quantum-inspired portfolio allocator: Markowitz mean-variance as a QUBO.

Real quantum-finance research (D-Wave, IBM, and bank quantum-computing
groups) formulates portfolio selection as a Quadratic Unconstrained Binary
Optimization (QUBO) problem, solvable on a quantum annealer -- or, since no
quantum hardware is used here, via simulated annealing, the QUBO's
classical proxy. The formulation is hardware-agnostic: the exact same
`dimod.BinaryQuadraticModel` built here could be submitted to a real D-Wave
annealer by swapping the sampler for `DWaveSampler`, with zero change to
the objective itself.

Unlike `AdaptiveEnsembleAgent` (ranks strategies by their own trailing
Sharpe, independently of one another), this jointly optimizes
weight' * mean_return against weight' * covariance * weight -- so two
strategies that are individually decent but move together contribute less
diversification value than their individual Sharpes alone would suggest.
That's a real methodological difference, not just a rebrand: the trailing-
Sharpe allocator has no way to represent "these two strategies are
redundant," and this one does, via the covariance matrix's off-diagonal
terms.

Weights are discretized onto a fixed grid (default 0%, 25%, 50%, 75%, 100%)
and one-hot encoded per strategy, since QUBO variables are binary --
continuous weights aren't representable directly. A finer grid gives a
closer approximation to the true continuous optimum at the cost of more
binary variables (and therefore a larger, slower-to-anneal QUBO).
"""

from __future__ import annotations

import dimod
import numpy as np
import pandas as pd
from neal import SimulatedAnnealingSampler


class QuboEnsembleAgent:
    def __init__(
        self,
        lookback: int = 60,
        rebalance_every: int = 21,
        weight_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        risk_aversion: float = 5.0,
        constraint_penalty: float = 20.0,
        num_reads: int = 100,
        seed: int | None = None,
    ):
        self.lookback = lookback
        self.rebalance_every = rebalance_every
        self.weight_levels = weight_levels
        self.risk_aversion = risk_aversion
        self.constraint_penalty = constraint_penalty
        self.num_reads = num_reads
        self.seed = seed

    def _var(self, i: int, level_idx: int) -> str:
        return f"x_{i}_{level_idx}"

    def _build_qubo(self, mu: np.ndarray, sigma: np.ndarray) -> dimod.BinaryQuadraticModel:
        n = len(mu)
        levels = self.weight_levels
        bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

        # Linear term: maximize expected return == minimize -mu' w
        for i in range(n):
            for level_idx, level in enumerate(levels):
                bqm.add_linear(self._var(i, level_idx), -mu[i] * level)

        # Quadratic term: minimize risk_aversion * w' Sigma w
        for i in range(n):
            for j in range(n):
                for li, level_i in enumerate(levels):
                    if level_i == 0.0:
                        continue
                    for lj, level_j in enumerate(levels):
                        if level_j == 0.0:
                            continue
                        coeff = self.risk_aversion * sigma[i, j] * level_i * level_j
                        var_i, var_j = self._var(i, li), self._var(j, lj)
                        if var_i == var_j:
                            bqm.add_linear(var_i, coeff)  # binary: x_i^2 == x_i
                        else:
                            bqm.add_quadratic(var_i, var_j, coeff)

        # Exactly one weight level selected per strategy (one-hot).
        for i in range(n):
            terms = [(self._var(i, level_idx), 1.0) for level_idx in range(len(levels))]
            bqm.add_linear_equality_constraint(terms, lagrange_multiplier=self.constraint_penalty, constant=-1.0)

        # Selected weights must sum to 1 across the book (fully invested, no leverage).
        budget_terms = [
            (self._var(i, level_idx), level) for i in range(n) for level_idx, level in enumerate(levels)
        ]
        bqm.add_linear_equality_constraint(
            budget_terms, lagrange_multiplier=self.constraint_penalty, constant=-1.0
        )

        return bqm

    def _solve(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        n = len(mu)
        bqm = self._build_qubo(mu, sigma)
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=self.num_reads, seed=self.seed)
        best = sampleset.first.sample

        weights = np.zeros(n)
        for i in range(n):
            for level_idx, level in enumerate(self.weight_levels):
                if best.get(self._var(i, level_idx), 0) == 1:
                    weights[i] = level
                    break

        total = weights.sum()
        return weights / total if total > 0 else np.full(n, 1.0 / n)

    def allocate(self, strategy_returns: pd.DataFrame) -> pd.DataFrame:
        """Return per-strategy capital weights, decided causally at each
        rebalance date using only data through the prior close."""
        cols = strategy_returns.columns
        n = len(cols)
        equal_weight = np.full(n, 1.0 / n)

        weights = pd.DataFrame(index=strategy_returns.index, columns=cols, dtype=float)
        current = equal_weight
        for i, date in enumerate(strategy_returns.index):
            if i > 0 and i % self.rebalance_every == 0:
                window = strategy_returns.iloc[max(0, i - self.lookback) : i]  # strictly before today
                if len(window) < max(self.lookback // 2, 2):
                    current = equal_weight
                else:
                    mu = window.mean().to_numpy() * 252
                    sigma = window.cov().to_numpy() * 252
                    try:
                        current = self._solve(mu, sigma)
                    except Exception:
                        current = equal_weight
            weights.loc[date] = current
        return weights

    def combine(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """Blend strategy return streams, applying weights decided at date t
        to date t+1's return -- consistent with the backtest engine's
        one-bar execution-lag convention."""
        weights = self.allocate(strategy_returns)
        applied = weights.shift(1).fillna(1.0 / weights.shape[1])
        return (applied * strategy_returns).sum(axis=1)
