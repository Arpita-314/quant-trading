# QPhase cross-validation

A real-data functional-correctness check: does an external quantum-computing
compiler stack (QPhase, a separate private project) correctly solve this
repo's own strategy-allocation problem?

## What this is

`run_comparison.py` takes the real mean/covariance of this repo's 6
backtested strategies (the exact numbers in the main [README](../../README.md#results))
and poses the question QPhase's existing fintech `PortfolioInstance` problem
type is built for: *which K of these strategies should get capital, to
maximize expected return minus risk-aversion times variance?*

It's solved three ways:

1. **Brute-force enumeration** — exact ground truth (trivial at n=6 strategies).
2. **QPhase's QAOA pipeline** — Farhi et al.'s Quantum Approximate Optimization
   Algorithm (arXiv:1411.4028), with a Dicke-state warm start and a
   Hamming-weight-preserving XY-ring mixer (Hadfield et al., arXiv:1907.09631)
   so the search stays inside the "exactly K selected" feasible subspace from
   the start. Compiled to a real gate set and executed via exact statevector
   simulation (no cloud hardware, no mocking).
3. **Classical simulated annealing** (`dimod` + `neal` — the same library this
   repo's own `QuboEnsembleAgent` uses) on the identical problem, as an honest
   side-by-side baseline.

## What this shows -- and, importantly, what it doesn't

Across every cardinality (K=1..5) and 5 random seeds each, QPhase's pipeline
matched the exact brute-force optimum 25/25 times. Classical simulated
annealing also matched it every time.

**This is a correctness result, not a performance claim.** This repo has 6
strategies; a 6-variable combinatorial search is small enough that classical
methods solve it just as reliably. The honest conclusion is: QPhase's real
compile-and-execute pipeline — problem definition through QAOA circuit
construction, gate compilation, and simulation — works correctly end-to-end
against real, externally-sourced financial data, not synthetic data built to
flatter either project. That's a genuine, verifiable result. It is not
evidence that this approach beats classical optimization at this scale, and
describing it that way would be overclaiming.

## A second, harder problem: capital-constrained trade selection

`run_trade_selection.py` poses a different, genuinely NP-hard question:
given real historical trade opportunities from this repo's own strategies
(6,325 of them, across all six strategies and the full backtest window),
which ones could a capital-constrained book actually have afforded to take?

Each contiguous run of a strategy's executed position on one ticker is a
real historical entry/exit window with a real realized profit (scored
exactly the way the backtest engine does — lagged signal times return, no
different accounting). Two opportunities conflict if their time windows
overlap, **or** if they're in the same sector and within a 10-day cooldown
of each other (a real, common concentration-limit practice, and
deliberately included: pure time-overlap-only conflicts form an interval
graph, on which this problem is solvable in polynomial time by a
specialized algorithm — an accidentally-easy special case that wouldn't
actually test general MWIS solving).

This is Maximum Weight Independent Set — one of Karp's original 21
NP-complete problems, and a standard QAOA benchmark in its own right
(Pichler, Wang, Zhou, Kok & Lukin, arXiv:1808.10816), not a problem class
invented for this repo.

To keep brute-force verification and exact statevector simulation fast,
the 14 largest-magnitude opportunities (by absolute profit) are kept; the
other 6,311 are dropped and that's logged explicitly, not silently
truncated. The resulting instance: 14 opportunities, 13 conflict edges.

| method | result |
|---|---|
| Brute-force optimum | 0.486 (3 non-overlapping NVDA insider-buying windows + one AMZN momentum window) |
| QPhase QAOA | matched the optimum in **4 of 5** random seeds |
| Classical simulated annealing (`neal`) | matched the optimum |

**This result is more informative than a clean sweep would have been.**
QAOA is a heuristic with no convergence guarantee — missing the optimum on
1 of 5 seeds is the expected, honest behavior of a shallow (p=2), modestly-
optimized (COBYLA, 80 iterations) circuit, not a bug. A suspiciously
perfect record across two different experiments would be the thing to be
suspicious of.

## A third, structurally different problem: sparse index tracking

`run_index_tracking.py` poses a different kind of objective entirely:
*minimize* tracking error rather than *maximize* return. Given this
repo's 7-ticker universe's real daily returns and an equal-weighted
"index" of all 7 as the benchmark, which K tickers alone best replicate
that benchmark's return series?

Cardinality-constrained sparse index tracking is a well-studied NP-hard
problem in the portfolio-construction literature. It matters as an
addition here specifically because it's not just another cardinality
selection problem with a different label -- minimizing squared tracking
error produces a different QUBO shape (a Gram matrix of the return data,
not a covariance-risk-vs-return tradeoff), and it's the first problem in
this folder where "lower is better" rather than "higher is better", which
exercises QPhase's `optimize_qaoa(..., sense="min")` path rather than the
`sense="max"` path every other script here uses.

| K | brute-force optimum | tracking MSE | QAOA hits (of 5 seeds) | neal matches |
|---|---|---|---|---|
| 1 | NVDA | 8.98e-5 | 5/5 | yes |
| 2 | AAPL, NVDA | 4.89e-5 | 5/5 | yes |
| 3 | MSFT, NVDA, PEP | 2.95e-5 | 5/5 | yes |
| 4 | AAPL, GOOGL, AMZN, KO | 1.66e-5 | 5/5 | yes |
| 5 | MSFT, GOOGL, AMZN, KO, PEP | 7.82e-6 | 5/5 | yes |
| 6 | AAPL, MSFT, GOOGL, AMZN, KO, PEP | 2.49e-6 | 5/5 | yes |

Tracking error decreases monotonically as K grows toward the full universe
(7 of 7 tickers would track itself exactly) -- the expected shape for this
problem, and a useful sanity check that the QUBO is encoding the right
thing. QPhase matched the exact optimum in all 30 (K, seed) combinations
here; so did classical simulated annealing, for the same n=7 reason as
everywhere else in this folder.

## Running it

None of the three scripts here are part of `quant-trading`'s installable
package or CI — QPhase is a separate, private repository, not a public
dependency, so nothing here can assume it's installed.

```bash
cd ../..                            # repo root
pip install -e .                    # this repo's own quant_trading package
pip install dimod dwave-neal        # classical cross-check
cd research/qphase-cross-validation
python run_comparison.py --qphase-path /path/to/your/qphase/checkout
python run_trade_selection.py --qphase-path /path/to/your/qphase/checkout
python run_index_tracking.py --qphase-path /path/to/your/qphase/checkout
```
