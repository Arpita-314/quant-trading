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

## A different algorithm family entirely: option pricing via amplitude estimation

`run_option_pricing.py` is not QAOA. Every problem above is a NISQ-native
heuristic with no proven speedup, which is why every section above says
"correctness, not a performance claim." **Quantum amplitude estimation**
(Brassard, Høyer, Mosca & Tapp, 2002) is different in kind: it has a
*proven* quadratic advantage over classical Monte Carlo (pricing error
O(1/M) in M oracle calls, vs. Monte Carlo's O(1/sqrt(N)) in N samples) --
in the fault-tolerant regime it targets. This script implements the
option-pricing application of that algorithm (Stamatopoulos, Egger, Sun,
Zoufal, Iten, Shen & Woerner, "Option Pricing using Quantum Computers,"
*Quantum* 4, 291 (2020)), using Suzuki et al.'s "amplitude estimation
without phase estimation" (2020) readout -- Grover-operator power
schedule + classical maximum-likelihood estimation, instead of the
deeper canonical circuit (ancilla phase register + controlled powers +
inverse QFT).

**What's genuinely validated here is correctness, not the speedup.**
Demonstrating a quadratic query advantage empirically needs a scale no
classical simulator can reach; claiming it at a handful of qubits would
undercut the actual point. What the pipeline was checked against, at
every stage while building it: the exact analytical Grover-amplification
formula (`sin^2((2m+1)*theta)`, verified to floating-point precision
before any application code was written on top of it), then the resulting
option price against closed-form Black-Scholes and classical Monte Carlo.

The price-uncertainty register is capped at 2 qubits (4 discretized price
levels) for this build -- the general multi-controlled-gate construction
needed for more qubits requires extra "work" qubits via a compute-
uncompute cascade, and correctly provisioning + validating that path was
out of scope for this pass (see `qphase/problems/option_pricing.py`'s
module docstring for the exact bug this avoided). That's a stated scope
limit, not a hidden one, and it has a real, visible cost: pricing a real
AAPL call option (spot from real market data, volatility from real
trailing realized returns, not a placeholder number) across five strikes:

| strike | moneyness | Black-Scholes | Monte Carlo | QAE (simulated) | QAE rel. error |
|---|---|---|---|---|---|
| 266.19 | 0.85 | 63.08 | 63.14 | 61.11 | 3.1% |
| 297.51 | 0.95 | 42.73 | 42.78 | 42.10 | 1.5% |
| 313.17 | 1.00 | 34.47 | 34.52 | 34.43 | **0.1%** |
| 328.83 | 1.05 | 27.46 | 27.51 | 26.87 | 2.2% |
| 360.15 | 1.15 | 16.85 | 16.87 | 13.86 | 17.8% |

Error is smallest at-the-money and grows in both directions -- exactly the
expected shape for a 4-point discretization grid concentrated near the
strike (see the class docstring's `n_std` discussion): the further a
strike sits from where the grid was built to resolve well, the fewer of
those 4 points land where the payoff actually matters. More qubits (finer
discretization) is precisely what the fault-tolerant hardware this
algorithm targets would provide -- not something to fake by cherry-picking
a grid that happens to fit one example.

## Tying it together: quantum risk estimation feeding quantum allocation, warm-started by a neural network

`run_neural_qaoa_init.py` combines three pieces built across this folder
into one pipeline, rather than three disconnected demos:

1. **Quantum-estimated tail risk** (`TailRiskInstance`, above) for each of
   this repo's 6 real strategies -- expected downside on each strategy's
   own empirical historical return distribution, via amplitude estimation
   rather than a classical formula.
2. **A hybrid risk-aware allocation QUBO** (`RiskAwarePortfolioInstance`):
   the same cardinality-constrained selection as `PortfolioInstance`, but
   with those quantum-estimated tail-risk numbers as a real input to the
   objective, alongside classical covariance -- quantum estimation feeding
   quantum optimization, not classical statistics feeding quantum
   optimization.
3. **A learned QAOA initializer**: using a classical neural network to
   predict good starting variational parameters instead of random
   initialization is published, active research (Verdon et al., "Learning
   to learn with quantum neural networks via classical neural networks,"
   2019, and follow-ups on meta-learning for QAOA). This is a small,
   honestly-scoped instance of that idea: a feedforward network
   (scikit-learn's `MLPRegressor`, a genuine neural net, not gradient
   boosting) trained on 150 synthetic 6-asset cardinality-constrained
   portfolio instances, predicting QAOA's 4 variational parameters
   (p=2 rounds) from problem features.

**The honest result: it made no measurable difference.** On all 40
held-out synthetic test instances, and on the real 6-strategy problem,
NN-initialized and randomly-initialized QAOA converged to *identical*
final objective values under a short, fixed 15-iteration budget --
0 wins for the NN, 0 wins for random, 40 ties. Diagnosing why (not
speculating): running the same real instance from a random starting
point vs. all-zeros, both reached the exact brute-force optimum within
15 iterations, landing on *different* final parameter values but the
same objective. The Dicke-state warm start and Hamming-weight-preserving
XY mixer already do the structural heavy lifting -- constraining the
search to the feasible "exactly K selected" subspace from the start --
so the remaining classical problem (finding good gamma/beta angles, only
4 numbers) is easy enough that COBYLA reliably finds the optimum from
almost any reasonable starting point at this scale. A learned initializer
has no headroom to add value when the thing it's trying to speed up isn't
actually the bottleneck.

This is reported as a negative result because that's what it is, not
tuned until it looked better. It's also a real, useful finding: it says
the interesting place to look for a learned-initializer payoff isn't
small, structurally-warm-started problems like this one -- it's larger
instances, or problems without a domain-specific warm start doing this
much of the work already.

## Running it

None of the five scripts here are part of `quant-trading`'s installable
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
python run_option_pricing.py --qphase-path /path/to/your/qphase/checkout
python run_neural_qaoa_init.py --qphase-path /path/to/your/qphase/checkout
```
