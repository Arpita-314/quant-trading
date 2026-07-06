"""
End-to-End Neural Network Pipeline for Options Mispricing
==========================================================

A single, self-contained neural network built from scratch using only numpy.
No PyTorch. No TensorFlow. No magic boxes.

Every forward pass, backprop, and gradient update is explicit and inspectable.

Architecture:
    Input (surface features) → Dense → BatchNorm → ReLU → Dropout
                             → Dense → BatchNorm → ReLU → Dropout
                             → Dense → BatchNorm → ReLU
                             → Output (mispricing ε)

Pipeline stages:
    1. Data generation (synthetic SPX-like option surface)
    2. Feature engineering (surface factors, Greeks, regime signals)
    3. Neural network (forward pass, backprop, Adam optimizer)
    4. Training loop (mini-batch, early stopping, LR decay)
    5. Evaluation (OOS R², directional accuracy, regime breakdown)
    6. Inference (predict mispricing on new surfaces)

Author: Quantitative Research
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which can't encode the unicode symbols
# (epsilon, checkmarks, arrows) used in the output below -- force UTF-8 so
# this doesn't crash on a fresh Windows checkout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BLACK-SCHOLES PRIMITIVES (inline — no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, q, sigma, flag="call"):
    """Vectorized Black-Scholes pricing."""
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-6)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if flag == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S, K, T, r, q, sigma, flag="call"):
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if flag == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)


def bs_vega(S, K, T, r, q, sigma):
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_gamma(S, K, T, r, q, sigma):
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def implied_vol_brent(market_price, S, K, T, r, q, flag="call", tol=1e-7, max_iter=100):
    """Brent's method IV solver — vectorized over arrays."""
    S, K, T, r, q = map(np.atleast_1d, [S, K, T, r, q])
    market_price = np.atleast_1d(market_price)
    n = len(market_price)
    ivs = np.full(n, np.nan)

    for i in range(n):
        try:
            intrinsic = max(S[i] - K[i], 0) if flag == "call" else max(K[i] - S[i], 0)
            if market_price[i] <= intrinsic * 1.001:
                ivs[i] = 0.0
                continue

            def obj(v):
                return bs_price(S[i], K[i], T[i], r[i], q[i], v, flag) - market_price[i]

            lo, hi = 1e-4, 5.0
            if obj(lo) * obj(hi) > 0:
                ivs[i] = 0.25  # fallback
                continue

            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                if obj(mid) * obj(lo) < 0:
                    hi = mid
                else:
                    lo = mid
                if (hi - lo) < tol:
                    break
            ivs[i] = 0.5 * (lo + hi)
        except Exception:
            ivs[i] = 0.25
    return ivs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA GENERATION (Realistic SPX-like option surface)
# ─────────────────────────────────────────────────────────────────────────────

def generate_surface_dataset(n_days=500, n_strikes=11, seed=42):
    """
    Generate realistic SPX-like option surface time series.

    Market features embedded:
    - Stochastic volatility (GARCH-like clustering)
    - Persistent negative put skew (~-0.15)
    - Smile curvature
    - Term structure (vol increases with maturity)
    - Volatility of volatility
    - Regime switching (calm → crisis)
    - Mean-reverting mispricing signal

    Returns:
    --------
    surfaces : list of dicts
        Each entry is one day's full option surface + metadata
    """
    np.random.seed(seed)

    # Strike grid (log-moneyness from -20% to +20%)
    log_m_grid = np.linspace(-0.20, 0.20, n_strikes)
    maturities = np.array([1/12, 3/12, 6/12, 1.0])  # 1M, 3M, 6M, 1Y

    # ─── Stochastic vol process ───────────────────────────────────────────
    # GARCH(1,1)-like
    omega, alpha, beta = 0.00001, 0.07, 0.92
    vol_t = 0.18  # starting vol
    spot = 4500.0  # SPX-like
    r, q = 0.05, 0.015

    # Regime switching
    regime = 0  # 0=calm, 1=stress
    regime_prob_switch = {0: 0.02, 1: 0.08}  # Prob of switching each day

    surfaces = []
    epsilon_lag = 0.0  # Lagged mispricing (mean-reverting)

    for day in range(n_days):
        # ─── Regime update ──────────────────────────────────────────────
        if np.random.rand() < regime_prob_switch[regime]:
            regime = 1 - regime  # flip

        # ─── Spot move ──────────────────────────────────────────────────
        daily_ret = vol_t / np.sqrt(252) * np.random.randn()
        if regime == 1:
            daily_ret -= 0.003  # drift down in stress
        spot *= np.exp(daily_ret)

        # ─── Vol update (GARCH) ─────────────────────────────────────────
        shock = (daily_ret * np.sqrt(252)) ** 2
        vol_t = np.sqrt(omega + alpha * shock + beta * vol_t**2)
        vol_t = np.clip(vol_t, 0.08, 0.90)
        if regime == 1:
            vol_t = max(vol_t, 0.30)  # floor in stress

        # ─── IV surface construction ─────────────────────────────────────
        # Surface parameterized as: IV(k, T) = base + term + skew*k + curve*k²
        #  where k = log(K/F) and F = forward
        base_vol   = vol_t
        term_slope = 0.02 * np.sqrt(maturities)  # upward-sloping typical
        skew       = -0.15 - 0.10 * (regime)      # steeper in stress
        curvature  = 0.25 + 0.15 * (regime)       # more smile in stress

        records = []
        for T_exp in maturities:
            for k in log_m_grid:
                iv_market = (
                    base_vol
                    + term_slope[np.searchsorted(maturities, T_exp)]
                    + skew * k
                    + curvature * k**2
                    + 0.005 * np.random.randn()   # market microstructure noise
                )
                iv_market = np.clip(iv_market, 0.05, 1.20)

                K = spot * np.exp(k)
                flag = "put" if k < 0 else "call"

                mkt_price = bs_price(spot, K, T_exp, r, q, iv_market, flag)
                bs_flat_price = bs_price(spot, K, T_exp, r, q, base_vol, flag)

                epsilon = mkt_price - bs_flat_price

                records.append({
                    "day": day,
                    "spot": spot,
                    "K": K,
                    "T": T_exp,
                    "r": r,
                    "q": q,
                    "log_moneyness": k,
                    "iv_market": iv_market,
                    "iv_bs_flat": base_vol,
                    "mkt_price": mkt_price,
                    "bs_flat_price": bs_flat_price,
                    "epsilon": epsilon,
                    "regime": regime,
                    "flag": flag,
                })

        surfaces.append({
            "day": day,
            "spot": spot,
            "vol": vol_t,
            "regime": regime,
            "skew": skew,
            "curvature": curvature,
            "records": pd.DataFrame(records),
        })

    return surfaces


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(surfaces, lookahead=1):
    """
    Build feature matrix X and target vector y.

    Features per (day, strike, maturity) observation:
    ─ Option-level:
        log_moneyness, maturity, BS delta, BS vega, BS gamma
        iv_market, iv_flat, iv_spread (market - flat)
    ─ Surface-level (cross-sectional summary each day):
        atm_vol_1m, atm_vol_3m, atm_vol_6m, atm_vol_1y
        skew_1m, skew_3m
        curvature_1m
        vol_of_vol (rolling std of atm_vol)
        term_spread (iv_1y - iv_1m)
    ─ Time-series:
        atm_vol_change_1d, atm_vol_change_5d
        skew_change_1d
        epsilon_lag_1d, epsilon_lag_5d (target autocorrelation)
    ─ Regime:
        realized_vol_21d (proxy for regime)
        vol_level_zscore

    Target:
        epsilon(t+lookahead) = market_price - bs_flat_price at next observation
        (mispricing we want to predict)
    """
    all_records = []

    for s in surfaces:
        df = s["records"].copy()
        day = s["day"]

        # ─── Surface-level features ──────────────────────────────────────
        for T_exp, label in [(1/12, "1m"), (3/12, "3m"), (6/12, "6m"), (1.0, "1y")]:
            slice_T = df[df["T"].round(4) == round(T_exp, 4)]
            atm = slice_T.iloc[(slice_T["log_moneyness"].abs()).argsort()[:1]]
            df[f"atm_vol_{label}"] = atm["iv_market"].values[0] if len(atm) else np.nan

        # Skew: OLS slope of iv vs log_moneyness for each maturity
        for T_exp, label in [(1/12, "1m"), (3/12, "3m")]:
            slice_T = df[df["T"].round(4) == round(T_exp, 4)]
            if len(slice_T) > 2:
                x = slice_T["log_moneyness"].values
                y_iv = slice_T["iv_market"].values
                A = np.column_stack([x, np.ones_like(x)])
                coef, *_ = np.linalg.lstsq(A, y_iv, rcond=None)
                df[f"skew_{label}"] = coef[0]
                # Curvature: quadratic term
                A2 = np.column_stack([x**2, x, np.ones_like(x)])
                coef2, *_ = np.linalg.lstsq(A2, y_iv, rcond=None)
                df[f"curvature_{label}"] = coef2[0]
            else:
                df[f"skew_{label}"] = np.nan
                df[f"curvature_{label}"] = np.nan

        df["term_spread"] = df["atm_vol_1y"] - df["atm_vol_1m"]
        df["iv_spread"] = df["iv_market"] - df["iv_bs_flat"]

        # ─── Option-level Greeks ─────────────────────────────────────────
        df["delta"] = bs_delta(
            df["spot"].values, df["K"].values, df["T"].values,
            df["r"].values, df["q"].values, df["iv_market"].values,
            "call"  # sign-flip for puts handled below
        )
        # Fix sign for puts
        put_mask = df["flag"] == "put"
        df.loc[put_mask, "delta"] = bs_delta(
            df.loc[put_mask, "spot"].values,
            df.loc[put_mask, "K"].values,
            df.loc[put_mask, "T"].values,
            df.loc[put_mask, "r"].values,
            df.loc[put_mask, "q"].values,
            df.loc[put_mask, "iv_market"].values,
            "put"
        )
        df["vega"] = bs_vega(
            df["spot"].values, df["K"].values, df["T"].values,
            df["r"].values, df["q"].values, df["iv_market"].values
        )
        df["gamma"] = bs_gamma(
            df["spot"].values, df["K"].values, df["T"].values,
            df["r"].values, df["q"].values, df["iv_market"].values
        )

        # Normalize vega and gamma to avoid scale dominance
        df["vega_normalized"] = df["vega"] / (df["spot"] * df["iv_market"] + 1e-8)
        df["gamma_normalized"] = df["gamma"] * df["spot"]**2 / 100

        df["day_index"] = day
        all_records.append(df)

    full_df = pd.concat(all_records, ignore_index=True)
    full_df = full_df.sort_values(["day_index", "T", "log_moneyness"])

    # ─── Time-series features (need per-day aggregates first) ─────────
    daily_atm = (full_df.groupby("day_index")["atm_vol_1m"]
                 .first().rename("atm_1m_daily"))
    daily_skew = (full_df.groupby("day_index")["skew_1m"]
                  .first().rename("skew_1m_daily"))

    daily_atm_delta_1d  = daily_atm.diff(1).rename("atm_change_1d")
    daily_atm_delta_5d  = daily_atm.diff(5).rename("atm_change_5d")
    daily_skew_delta_1d = daily_skew.diff(1).rename("skew_change_1d")
    daily_vol_of_vol    = daily_atm.rolling(21).std().rename("vol_of_vol")
    daily_vol_zscore    = (
        (daily_atm - daily_atm.expanding(min_periods=5).mean())
        / (daily_atm.expanding(min_periods=5).std() + 1e-8)
    ).rename("vol_zscore")

    daily_epsilon_lag1 = (full_df.groupby("day_index")["epsilon"]
                          .mean().shift(1).rename("epsilon_lag1"))
    daily_epsilon_lag5 = (full_df.groupby("day_index")["epsilon"]
                          .mean().shift(5).rename("epsilon_lag5"))

    ts_features = pd.concat([
        daily_atm_delta_1d, daily_atm_delta_5d,
        daily_skew_delta_1d, daily_vol_of_vol,
        daily_vol_zscore, daily_epsilon_lag1, daily_epsilon_lag5
    ], axis=1)

    full_df = full_df.merge(ts_features, left_on="day_index", right_index=True, how="left")

    # ─── Normalize epsilon to % of BS flat price (makes target scale ~[-1, 1]) ──
    full_df["epsilon_pct"] = full_df["epsilon"] / (full_df["bs_flat_price"].abs() + 1e-8)
    full_df["epsilon_pct"] = full_df["epsilon_pct"].clip(-3.0, 3.0)  # clip extreme deep OTM noise
    full_df["epsilon_lag1"] = (full_df["epsilon_lag1"] / (full_df["bs_flat_price"].abs() + 1e-8)).clip(-3, 3)
    full_df["epsilon_lag5"] = (full_df["epsilon_lag5"] / (full_df["bs_flat_price"].abs() + 1e-8)).clip(-3, 3)

    # ─── Target: future mispricing ───────────────────────────────────────
    # Target: next-day epsilon as % of BS price
    future_eps = (full_df.groupby(["log_moneyness", "T"])["epsilon_pct"]
                  .shift(-lookahead)
                  .rename("target_epsilon"))
    full_df["target_epsilon"] = future_eps.values

    # Drop rows with NaN targets
    full_df = full_df.dropna(subset=["target_epsilon"])
    full_df = full_df.dropna(subset=["atm_change_1d", "vol_of_vol", "epsilon_lag1"])

    return full_df


FEATURE_COLS = [
    # Option-level
    "log_moneyness", "T", "iv_market", "iv_spread",
    "delta", "vega_normalized", "gamma_normalized",
    # Surface-level
    "atm_vol_1m", "atm_vol_3m", "atm_vol_6m", "atm_vol_1y",
    "skew_1m", "skew_3m", "curvature_1m", "term_spread",
    # Time-series
    "atm_change_1d", "atm_change_5d", "skew_change_1d",
    "vol_of_vol", "vol_zscore",
    # Lagged target (autocorrelation signal)
    "epsilon_lag1", "epsilon_lag5",
]
TARGET_COL = "target_epsilon"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: NEURAL NETWORK (pure numpy — no frameworks)
# ─────────────────────────────────────────────────────────────────────────────

class Layer:
    """Dense layer: y = W·x + b"""

    def __init__(self, n_in, n_out, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # He initialization (good for ReLU networks)
        self.W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        self.b = np.zeros((n_out, 1))

        # Adam optimizer state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        self.dW = None
        self.db = None
        self.x_cache = None

    def forward(self, x):
        """x: (n_in, batch)"""
        self.x_cache = x
        return self.W @ x + self.b

    def backward(self, dout):
        """dout: (n_out, batch)"""
        batch = dout.shape[1]
        self.dW = (dout @ self.x_cache.T) / batch
        self.db = dout.mean(axis=1, keepdims=True)
        return self.W.T @ dout


class BatchNorm:
    """Batch normalization layer."""

    def __init__(self, n_features, eps=1e-8, momentum=0.1):
        self.gamma = np.ones((n_features, 1))
        self.beta  = np.zeros((n_features, 1))
        self.eps   = eps
        self.momentum = momentum
        self.running_mean = np.zeros((n_features, 1))
        self.running_var  = np.ones((n_features, 1))

        # Adam state
        self.m_g = np.zeros_like(self.gamma)
        self.v_g = np.zeros_like(self.gamma)
        self.m_b = np.zeros_like(self.beta)
        self.v_b = np.zeros_like(self.beta)
        self.dg = None
        self.db = None

        # Cached for backprop
        self._x_hat = None
        self._std   = None
        self._x_mu  = None

    def forward(self, x, training=True):
        """x: (features, batch)"""
        if training:
            mu  = x.mean(axis=1, keepdims=True)
            var = x.var(axis=1, keepdims=True)
            self._std   = np.sqrt(var + self.eps)
            self._x_mu  = x - mu
            self._x_hat = self._x_mu / self._std
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            self._x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * self._x_hat + self.beta

    def backward(self, dout):
        batch = dout.shape[1]
        self.dg = (dout * self._x_hat).sum(axis=1, keepdims=True) / batch
        self.db = dout.mean(axis=1, keepdims=True)

        dx_hat = dout * self.gamma
        dvar   = (-0.5 * dx_hat * self._x_mu * self._std**-3).sum(axis=1, keepdims=True)
        dmu    = (-dx_hat / self._std).sum(axis=1, keepdims=True) + dvar * (-2 * self._x_mu).mean(axis=1, keepdims=True)
        dx     = dx_hat / self._std + dvar * 2 * self._x_mu / batch + dmu / batch
        return dx


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class Dropout:
    def __init__(self, rate=0.1):
        self.rate = rate
        self._mask = None

    def forward(self, x, training=True):
        if not training or self.rate == 0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask


class MispricingNet:
    """
    End-to-end neural network for BS mispricing prediction.

    Architecture:
        Input(24) → Linear(64) → BN → ReLU → Dropout(0.1)
                  → Linear(128) → BN → ReLU → Dropout(0.1)
                  → Linear(64)  → BN → ReLU
                  → Linear(1)
    """

    def __init__(self, n_features, hidden_sizes=(64, 128, 64), dropout_rate=0.1, seed=0):
        sizes = [n_features] + list(hidden_sizes) + [1]
        np.random.seed(seed)

        self.layers = []
        self.bns    = []
        self.relus  = []
        self.drops  = []

        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1], seed=seed + i))
            if i < len(sizes) - 2:  # No BN/activation on output layer
                self.bns.append(BatchNorm(sizes[i + 1]))
                self.relus.append(ReLU())
                self.drops.append(Dropout(dropout_rate))

        self.t = 0  # Adam timestep

    def forward(self, x, training=True):
        """x: (features, batch)"""
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.layers) - 1:
                out = self.bns[i].forward(out, training)
                out = self.relus[i].forward(out)
                out = self.drops[i].forward(out, training)
        return out  # (1, batch)

    def backward(self, dout):
        """Backprop through all layers."""
        grad = dout
        # Reverse through layers
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                grad = self.drops[i].backward(grad)
                grad = self.relus[i].backward(grad)
                grad = self.bns[i].backward(grad)
            grad = self.layers[i].backward(grad)

    def mse_loss(self, pred, target):
        """Mean squared error loss and gradient."""
        diff = pred - target
        loss = (diff**2).mean()
        dloss = 2 * diff / diff.size
        return loss, dloss

    def adam_step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        """Adam optimizer update for all parameters."""
        self.t += 1
        bc1 = 1 - beta1**self.t
        bc2 = 1 - beta2**self.t

        def _update(param, grad, m, v):
            # L2 weight decay
            grad = grad + weight_decay * param
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / bc1
            v_hat = v / bc2
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        for layer in self.layers:
            if layer.dW is not None:
                _update(layer.W, layer.dW, layer.mW, layer.vW)
                _update(layer.b, layer.db, layer.mb, layer.vb)

        for bn in self.bns:
            if bn.dg is not None:
                _update(bn.gamma, bn.dg, bn.m_g, bn.v_g)
                _update(bn.beta, bn.db, bn.m_b, bn.v_b)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val,
          epochs=100, batch_size=256, lr=1e-3,
          patience=15, min_lr=1e-5, lr_decay=0.5, lr_decay_patience=8,
          verbose=True):
    """
    Full training loop with:
    - Mini-batch gradient descent
    - Adam optimizer
    - Early stopping (patience on val loss)
    - LR decay on plateau
    - Weight decay (L2 regularization)
    """
    n = X_train.shape[0]
    best_val_loss = np.inf
    best_weights  = None
    no_improve    = 0
    no_lr_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(epochs):
        # ─── Shuffle ─────────────────────────────────────────────────────
        idx = np.random.permutation(n)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        epoch_loss = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size].T  # (features, batch)
            yb = y_shuf[start:start + batch_size].reshape(1, -1)

            pred = model.forward(Xb, training=True)
            loss, dloss = model.mse_loss(pred, yb)
            model.backward(dloss)
            model.adam_step(lr=lr, weight_decay=1e-4)

            epoch_loss += loss
            n_batches  += 1

        train_loss = epoch_loss / n_batches

        # ─── Validation ──────────────────────────────────────────────────
        val_pred = model.forward(X_val.T, training=False)
        val_loss, _ = model.mse_loss(val_pred, y_val.reshape(1, -1))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        # ─── Early stopping ──────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            no_improve    = 0
            no_lr_improve = 0
            # Save best weights (shallow copy of all params)
            best_weights = _snapshot_weights(model)
        else:
            no_improve    += 1
            no_lr_improve += 1

        if no_lr_improve >= lr_decay_patience:
            lr = max(lr * lr_decay, min_lr)
            no_lr_improve = 0

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch % 20 == 0 or epoch < 5):
            print(f"  Epoch {epoch+1:4d} | train: {train_loss:.6f} | val: {val_loss:.6f} | lr: {lr:.2e}")

    # Restore best weights
    if best_weights is not None:
        _restore_weights(model, best_weights)

    return history


def _snapshot_weights(model):
    """Deep copy all trainable params."""
    snap = {
        "layers": [(l.W.copy(), l.b.copy()) for l in model.layers],
        "bns":    [(bn.gamma.copy(), bn.beta.copy(),
                    bn.running_mean.copy(), bn.running_var.copy()) for bn in model.bns],
    }
    return snap


def _restore_weights(model, snap):
    for i, (W, b) in enumerate(snap["layers"]):
        model.layers[i].W = W
        model.layers[i].b = b
    for i, (g, b, rm, rv) in enumerate(snap["bns"]):
        model.bns[i].gamma = g
        model.bns[i].beta  = b
        model.bns[i].running_mean = rm
        model.bns[i].running_var  = rv


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, df_test, scaler):
    """
    Comprehensive OOS evaluation:
    - R², MAE, directional accuracy
    - Regime breakdown (calm vs stress)
    - By moneyness bucket
    - By maturity bucket
    - Residual autocorrelation (are residuals predictable?)
    """
    pred_raw = model.forward(X_test.T, training=False).flatten()
    y_raw    = y_test.flatten()

    # ─── Global metrics ──────────────────────────────────────────────────
    ss_res = ((y_raw - pred_raw)**2).sum()
    ss_tot = ((y_raw - y_raw.mean())**2).sum()
    r2     = 1 - ss_res / ss_tot
    mae    = np.abs(y_raw - pred_raw).mean()
    rmse   = np.sqrt(((y_raw - pred_raw)**2).mean())

    dir_acc = ((y_raw > 0) == (pred_raw > 0)).mean()

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE EVALUATION")
    print("=" * 70)
    print(f"\n  R²                : {r2:.4f}")
    print(f"  MAE               : {mae:.6f}")
    print(f"  RMSE              : {rmse:.6f}")
    print(f"  Directional Acc   : {dir_acc:.2%}")

    # ─── By regime ───────────────────────────────────────────────────────
    if "regime" in df_test.columns:
        print("\n  By Regime:")
        for reg, label in [(0, "Calm"), (1, "Stress")]:
            mask = df_test["regime"].values == reg
            if mask.sum() < 10:
                continue
            y_r  = y_raw[mask]
            p_r  = pred_raw[mask]
            r2_r = 1 - ((y_r - p_r)**2).sum() / ((y_r - y_r.mean())**2).sum()
            print(f"    {label:8s}: R²={r2_r:.4f}  n={mask.sum():>5}")

    # ─── By moneyness ────────────────────────────────────────────────────
    if "log_moneyness" in df_test.columns:
        print("\n  By Moneyness Bucket:")
        bins = [(-0.25, -0.10, "Deep OTM Put"),
                (-0.10, -0.03, "OTM Put"),
                (-0.03,  0.03, "ATM"),
                ( 0.03,  0.10, "OTM Call"),
                ( 0.10,  0.25, "Deep OTM Call")]
        lm = df_test["log_moneyness"].values
        for lo, hi, lbl in bins:
            mask = (lm >= lo) & (lm < hi)
            if mask.sum() < 5:
                continue
            y_b  = y_raw[mask]
            p_b  = pred_raw[mask]
            r2_b = 1 - ((y_b - p_b)**2).sum() / max(((y_b - y_b.mean())**2).sum(), 1e-8)
            print(f"    {lbl:20s}: R²={r2_b:.4f}  n={mask.sum():>5}")

    # ─── By maturity ─────────────────────────────────────────────────────
    if "T" in df_test.columns:
        print("\n  By Maturity:")
        for T_val, label in [(1/12, "1M"), (3/12, "3M"), (6/12, "6M"), (1.0, "1Y")]:
            mask = np.abs(df_test["T"].values - T_val) < 0.01
            if mask.sum() < 5:
                continue
            y_m  = y_raw[mask]
            p_m  = pred_raw[mask]
            r2_m = 1 - ((y_m - p_m)**2).sum() / max(((y_m - y_m.mean())**2).sum(), 1e-8)
            print(f"    {label:6s}: R²={r2_m:.4f}  n={mask.sum():>5}")

    # ─── Residual autocorrelation check ──────────────────────────────────
    residuals = y_raw - pred_raw
    if len(residuals) > 10:
        ac1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        print(f"\n  Residual autocorrelation (lag 1): {ac1:.4f}")
        if abs(ac1) > 0.1:
            print("  ⚠ Residuals are autocorrelated — model is leaving signal on the table")
        else:
            print("  ✓ Residuals approximately white noise")

    return pred_raw


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: INFERENCE MODULE
# ─────────────────────────────────────────────────────────────────────────────

class MispricingPredictor:
    """
    Production inference wrapper.

    Usage:
        predictor = MispricingPredictor(model, scaler, feature_cols)
        predictions = predictor.predict(new_surface_df)
    """

    def __init__(self, model, scaler, feature_cols):
        self.model        = model
        self.scaler       = scaler
        self.feature_cols = feature_cols

    def predict(self, df):
        """
        Predict mispricing for a new surface.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain all columns in feature_cols

        Returns:
        --------
        predictions : np.ndarray
            Predicted ε per (strike, maturity) pair
        """
        X = df[self.feature_cols].values.astype(np.float64)
        X_scaled = self.scaler.transform(X)
        pred = self.model.forward(X_scaled.T, training=False)
        return pred.flatten()

    def rank_opportunities(self, df, top_n=10):
        """
        Rank options by predicted mispricing magnitude.

        Returns top candidates for the delta-hedged strategy.
        """
        df = df.copy()
        df["predicted_epsilon"] = self.predict(df)
        df["abs_epsilon"] = df["predicted_epsilon"].abs()
        df["direction"] = np.where(df["predicted_epsilon"] > 0, "BUY", "SELL")

        ranked = df.nlargest(top_n, "abs_epsilon")[[
            "log_moneyness", "T", "iv_market",
            "predicted_epsilon", "abs_epsilon", "direction"
        ]]
        return ranked


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 70)
    print("END-TO-END NEURAL NETWORK PIPELINE: BS MISPRICING PREDICTION")
    print("=" * 70)

    # ─── 1. Generate data ────────────────────────────────────────────────
    print("\n[1/7] Generating synthetic SPX-like option surfaces...")
    surfaces = generate_surface_dataset(n_days=500, n_strikes=11, seed=42)
    n_calm   = sum(1 for s in surfaces if s["regime"] == 0)
    n_stress = sum(1 for s in surfaces if s["regime"] == 1)
    print(f"      {len(surfaces)} trading days | calm={n_calm} | stress={n_stress}")
    print(f"      Strikes: 11 | Maturities: 4 | Total obs/day: 44")

    # ─── 2. Feature engineering ──────────────────────────────────────────
    print("\n[2/7] Engineering features...")
    df = engineer_features(surfaces, lookahead=1)
    print(f"      Total observations: {len(df):,}")
    print(f"      Features: {len(FEATURE_COLS)}")
    print(f"      Target: next-day mispricing ε")

    # ─── 3. Train / validation / test split (time-based!) ────────────────
    print("\n[3/7] Splitting data (time-based, no leakage)...")
    days = df["day_index"].unique()
    n_days_total = len(days)
    train_cutoff = days[int(0.70 * n_days_total)]
    val_cutoff   = days[int(0.85 * n_days_total)]

    train_df = df[df["day_index"] <= train_cutoff]
    val_df   = df[(df["day_index"] > train_cutoff) & (df["day_index"] <= val_cutoff)]
    test_df  = df[df["day_index"] > val_cutoff]

    print(f"      Train: {len(train_df):>7,} obs ({train_df['day_index'].nunique()} days)")
    print(f"      Val:   {len(val_df):>7,} obs ({val_df['day_index'].nunique()} days)")
    print(f"      Test:  {len(test_df):>7,} obs ({test_df['day_index'].nunique()} days)")

    X_train_raw = train_df[FEATURE_COLS].values.astype(np.float64)
    y_train     = train_df[TARGET_COL].values.astype(np.float64)
    X_val_raw   = val_df[FEATURE_COLS].values.astype(np.float64)
    y_val       = val_df[TARGET_COL].values.astype(np.float64)
    X_test_raw  = test_df[FEATURE_COLS].values.astype(np.float64)
    y_test      = test_df[TARGET_COL].values.astype(np.float64)

    # ─── 4. Scaling (fit on train only) ──────────────────────────────────
    print("\n[4/7] Normalizing features (fit on train only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)
    print(f"      Feature mean range: [{X_train.mean(axis=0).min():.3f}, {X_train.mean(axis=0).max():.3f}]")
    print(f"      Feature std  range: [{X_train.std(axis=0).min():.3f}, {X_train.std(axis=0).max():.3f}]")

    # ─── 5. Build and train neural network ───────────────────────────────
    print("\n[5/7] Training neural network...")
    print(f"      Architecture: {len(FEATURE_COLS)} → 64 → 128 → 64 → 1")
    print(f"      Params: BatchNorm, ReLU, Dropout(0.1), Adam, L2 decay")

    model = MispricingNet(
        n_features=len(FEATURE_COLS),
        hidden_sizes=(64, 128, 64),
        dropout_rate=0.1,
        seed=42
    )

    n_params = sum(l.W.size + l.b.size for l in model.layers)
    n_params += sum(bn.gamma.size + bn.beta.size for bn in model.bns)
    print(f"      Total parameters: {n_params:,}")
    print()

    history = train(
        model, X_train, y_train, X_val, y_val,
        epochs=200,
        batch_size=512,
        lr=3e-3,
        patience=20,
        lr_decay=0.5,
        lr_decay_patience=8,
        verbose=True
    )

    # ─── 6. Evaluate ─────────────────────────────────────────────────────
    print("\n[6/7] Evaluating on held-out test set...")
    predictor = MispricingPredictor(model, scaler, FEATURE_COLS)
    predictions = evaluate(model, X_test, y_test, test_df, scaler)

    # ─── 7. Inference demo ───────────────────────────────────────────────
    print("\n[7/7] Inference demo: Top mispricing opportunities...")
    latest_surface = test_df[test_df["day_index"] == test_df["day_index"].max()]
    ranked = predictor.rank_opportunities(latest_surface, top_n=8)
    print("\n  Top options by predicted mispricing:")
    print(f"  {'Moneyness':>10} {'Maturity':>10} {'IV Mkt':>8} {'Pred ε':>10} {'Signal':>6}")
    print("  " + "-" * 52)
    for _, row in ranked.iterrows():
        print(f"  {row['log_moneyness']:>10.3f} {row['T']:>10.3f} "
              f"{row['iv_market']:>8.3f} {row['predicted_epsilon']:>10.6f} "
              f"  {row['direction']:>4}")

    # ─── Training curve summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Epochs trained:       {len(history['train_loss'])}")
    print(f"  Best val loss:        {min(history['val_loss']):.6f}")
    print(f"  Final train loss:     {history['train_loss'][-1]:.6f}")
    print(f"  Final LR:             {history['lr'][-1]:.2e}")

    # ─── Self-critique ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SELF-CRITIQUE")
    print("=" * 70)
    print("""
  What this network is doing correctly:
  ✓ Predicts ε from surface factors, not raw prices (right target)
  ✓ Uses only lagged features (no look-ahead bias)
  ✓ Train/val/test split is time-ordered (no data leakage)
  ✓ Analytical Greeks as features (not black-box inputs)
  ✓ BatchNorm handles different feature scales internally
  ✓ Early stopping prevents overfitting

  What this network is NOT doing:
  ✗ Using real market data (synthetic surface only)
  ✗ Modeling transaction costs in loss function
  ✗ Accounting for bid-ask spread in training signal
  ✗ Handling corporate actions, dividends, earnings
  ✗ Predicting regime changes (treated as external)
  ✗ Uncertainty quantification (no prediction intervals)

  What would improve it:
  → Add recurrent layer (LSTM) for temporal dependencies
  → Use real options data (CBOE SPX, 10+ years)
  → Custom loss function penalizing mistraded positions
  → Ensemble: average 5+ models trained on different seeds
  → Conformal prediction for valid coverage guarantees
  → Direct policy learning (predict position size, not just ε)
""")

    return model, predictor, history, df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, predictor, history, df = run_pipeline()