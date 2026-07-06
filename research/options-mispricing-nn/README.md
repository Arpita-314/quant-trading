# Options mispricing neural network

A neural network that predicts next-day Black-Scholes mispricing (market
price minus flat-vol BS price) from an option's implied-vol surface, trained
and evaluated on synthetic SPX-like data with realistic GARCH-style vol
clustering, skew, and regime switching (calm vs. stress).

Two implementations of the same idea:

- **`model_pricing.py`** — self-contained, heavily commented, runs
  end-to-end with `python model_pricing.py`. Best for reading top to bottom.
- **`model.py`** — a leaner CLI scaffold driven by `config.yaml`
  (`python model.py --config config.yaml`), with structured logging and
  artifact saving (model snapshot, scaler stats, metrics) to `artifacts/`.

Everything is pure numpy for the network itself (dense layers, batch norm,
ReLU, dropout, Adam, backprop) — no PyTorch/TensorFlow — so every forward
pass and gradient update is inspectable. Black-Scholes pricing, delta,
vega, and gamma are implemented inline rather than pulled from a library.

## What this demonstrates

- Time-ordered train/val/test split (no shuffling across days -- shuffling
  a time series before splitting leaks future surfaces into training)
- Features are lagged relative to the prediction target; the label is
  *next-day* mispricing
- Feature scaling (`StandardScaler`) is fit on the training split only
- Evaluation breaks results down by regime, moneyness bucket, and maturity
  bucket rather than reporting one aggregate number, plus a residual-
  autocorrelation check (if residuals are autocorrelated, the model is
  leaving predictable signal on the table)
- An explicit self-critique in `model_pricing.py`'s output: what the model
  gets right, what it doesn't handle (transaction costs, bid-ask spread,
  corporate actions, uncertainty quantification), and what would improve it

## Honest scope

Trained and evaluated on **synthetic** data, not real options chains — the
realistic R²/directional-accuracy numbers it reports describe how well the
network recovers a *known, synthetic* mispricing process, not a claim about
real-market predictability. Swapping in real CBOE/OCC data would be the
natural next step, and is exactly the kind of change the causal, no-leakage
structure here is meant to survive without further rework.

## Running it

```bash
pip install -r requirements.txt
python model_pricing.py            # self-contained walkthrough
python model.py --config config.yaml   # CLI scaffold, writes artifacts/
```
