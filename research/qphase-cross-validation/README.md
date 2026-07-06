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

## Running it

This script is intentionally **not** part of `quant-trading`'s installable
package or CI — QPhase is a separate, private repository, not a public
dependency, so nothing here can assume it's installed.

```bash
cd ../..                            # repo root
pip install -e .                    # this repo's own quant_trading package
pip install dimod dwave-neal        # classical cross-check
cd research/qphase-cross-validation
python run_comparison.py --qphase-path /path/to/your/qphase/checkout
```
