#!/usr/bin/env python
"""
BS Mispricing Net — production-grade scaffold.
Run: python model.py --config config.yaml
Deps: numpy, pandas, scipy, scikit-learn, (optional) pyyaml
"""
from __future__ import annotations
import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, get_origin, get_type_hints

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = Path("artifacts")

# ── Logging ──────────────────────────────────────────────────────────────
def init_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("bs-mispricing")

# ── Config ───────────────────────────────────────────────────────────────
@dataclass
class Config:
    n_days: int = 500
    n_strikes: int = 11
    r: float = 0.05
    q: float = 0.015
    train_split: float = 0.70
    val_split: float = 0.85
    epochs: int = 200
    batch_size: int = 512
    lr: float = 3e-3
    patience: int = 20
    drop_rate: float = 0.10
    sizes: Tuple[int, ...] = (64, 128, 64)
    log_level: str = "INFO"
    seed: int = 42


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

# ── Black–Scholes ────────────────────────────────────────────────────────
def bs_price(S: float, K: float, T: float, r: float, q: float, v: float, flag: str) -> float:
    T = max(T, 1e-8)
    sigt = v * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * v * v) * T) / sigt
    d2 = d1 - sigt
    if flag == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# ── Synthetic data gen ───────────────────────────────────────────────────
def generate_rows(cfg: Config) -> List[Dict[str, float]]:
    lm = np.linspace(-0.2, 0.2, cfg.n_strikes)
    mats = [1 / 12, 3 / 12, 6 / 12, 1.0]
    spot, vol, regime = 4500.0, 0.18, 0
    rows: List[Dict[str, float]] = []
    for day in range(cfg.n_days):
        if np.random.rand() < (0.02 if regime == 0 else 0.08):
            regime ^= 1
        spot *= np.exp(vol / np.sqrt(252) * np.random.randn() - 0.003 * regime)
        vol = float(np.clip((1e-5 + 0.07 * (vol * np.random.randn()) ** 2 + 0.92 * vol**2) ** 0.5,
                            max(0.08, 0.3 * regime), 0.9))
        skew, curve = -0.15 - 0.1 * regime, 0.25 + 0.15 * regime
        for T in mats:
            for k in lm:
                iv = float(np.clip(vol + 0.02 * np.sqrt(T) + skew * k + curve * k * k + 0.005 * np.random.randn(),
                                   0.05, 1.2))
                K = spot * np.exp(k)
                flag = "put" if k < 0 else "call"
                mkt = bs_price(spot, K, T, cfg.r, cfg.q, iv, flag)
                flat = bs_price(spot, K, T, cfg.r, cfg.q, vol, flag)
                rows.append(dict(day=day, k=k, T=T, iv=iv, vol=vol, flat=flat, regime=regime,
                                 eps=float(np.clip((mkt - flat) / (flat + 1e-8), -3, 3))))
    return rows

# ── Features ─────────────────────────────────────────────────────────────
FEATS = [
    "k","T","iv","iv_spread","atm_1m","atm_3m","atm_1y",
    "skew_1m","skew_3m","term","datm_1d","datm_5d","vov","vol_z","eps_lag1","eps_lag5"
]

def featurize(rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["iv_spread"] = df["iv"] - df["vol"]

    def add_atm_and_skew(T: float, suffix: str) -> None:
        sl = df[df["T"].round(4) == round(T, 4)]
        atm = sl.loc[sl["k"].abs().groupby(sl["day"]).idxmin()].set_index("day")["iv"]
        skew = sl.groupby("day").apply(
            lambda g: np.linalg.lstsq(np.c_[g["k"], np.ones(len(g))], g["iv"], rcond=None)[0][0],
            include_groups=False,
        )
        df[f"atm_{suffix}"] = df["day"].map(atm)
        df[f"skew_{suffix}"] = df["day"].map(skew)

    add_atm_and_skew(1 / 12, "1m")
    add_atm_and_skew(3 / 12, "3m")
    add_atm_and_skew(1.0, "1y")

    df["term"] = df["atm_1y"] - df["atm_1m"]
    de = df.groupby("day")["eps"].mean()
    da = df.groupby("day")["atm_1m"].mean()
    df["eps_lag1"] = df["day"].map(de.shift(1)).clip(-3, 3)
    df["eps_lag5"] = df["day"].map(de.shift(5)).clip(-3, 3)
    df["datm_1d"] = df["day"].map(da.diff(1))
    df["datm_5d"] = df["day"].map(da.diff(5))
    df["vov"] = df["day"].map(da.rolling(21).std())
    df["vol_z"] = df["day"].map((da - da.expanding(5).mean()) / (da.expanding(5).std() + 1e-8))
    df["target"] = df.groupby(["k", "T"])["eps"].shift(-1)
    return df.dropna(subset=["target", "eps_lag1", "vov", "datm_1d"])

# ── Minimal NN (same architecture) ───────────────────────────────────────
class Dense:
    def __init__(self, ni: int, no: int):
        self.W = np.random.randn(no, ni) * np.sqrt(2 / ni)
        self.b = np.zeros((no, 1))
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.x = None; self.dW = None; self.db = None

    def fwd(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return self.W @ x + self.b

    def bwd(self, d: np.ndarray) -> np.ndarray:
        self.dW = d @ self.x.T / d.shape[1]
        self.db = d.mean(1, keepdims=True)
        return self.W.T @ d

class BN:
    def __init__(self, n: int):
        self.g = np.ones((n, 1)); self.b = np.zeros((n, 1))
        self.rm = np.zeros((n, 1)); self.rv = np.ones((n, 1))
        self.mg = np.zeros((n, 1)); self.vg = np.zeros((n, 1))
        self.mb2 = np.zeros((n, 1)); self.vb2 = np.zeros((n, 1))
        self._xh = self._s = self._xm = None
        self.dg = self.db2 = None

    def fwd(self, x: np.ndarray, train: bool) -> np.ndarray:
        if train:
            mu = x.mean(1, keepdims=True); var = x.var(1, keepdims=True)
            self._s = (var + 1e-8) ** 0.5; self._xm = x - mu; self._xh = self._xm / self._s
            self.rm = 0.9 * self.rm + 0.1 * mu; self.rv = 0.9 * self.rv + 0.1 * var
        else:
            self._xh = (x - self.rm) / (self.rv + 1e-8) ** 0.5
        return self.g * self._xh + self.b

    def bwd(self, d: np.ndarray) -> np.ndarray:
        n = d.shape[1]
        self.dg = (d * self._xh).sum(1, keepdims=True) / n
        self.db2 = d.mean(1, keepdims=True)
        dxh = d * self.g
        dv = (-0.5 * dxh * self._xm * self._s**-3).sum(1, keepdims=True)
        dm = (-dxh / self._s).sum(1, keepdims=True) + dv * (-2 * self._xm).mean(1, keepdims=True)
        return dxh / self._s + dv * 2 * self._xm / n + dm / n

class ReLU:
    def __init__(self): self.m = None
    def fwd(self, x): self.m = x > 0; return x * self.m
    def bwd(self, d): return d * self.m

class Drop:
    def __init__(self, r: float): self.r = r; self.m = None
    def fwd(self, x, train: bool):
        if not train: return x
        self.m = (np.random.rand(*x.shape) > self.r) / (1 - self.r)
        return x * self.m
    def bwd(self, d): return d * self.m

class Net:
    def __init__(self, ni: int, sizes: Tuple[int, ...], drop: float):
        s = [ni] + list(sizes) + [1]
        self.D = [Dense(s[i], s[i + 1]) for i in range(len(s) - 1)]
        self.B = [BN(h) for h in sizes]
        self.R = [ReLU() for _ in sizes]
        self.P = [Drop(drop) for _ in sizes]
        self.t = 0

    def fwd(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        for d, b, r, p in zip(self.D[:-1], self.B, self.R, self.P):
            x = p.fwd(r.fwd(b.fwd(d.fwd(x), train)), train)
        return self.D[-1].fwd(x)

    def bwd(self, g: np.ndarray):
        g = self.D[-1].bwd(g)
        for d, b, r, p in zip(reversed(self.D[:-1]), reversed(self.B), reversed(self.R), reversed(self.P)):
            g = d.bwd(b.bwd(r.bwd(p.bwd(g))))

    def step(self, lr: float, b1: float = 0.9, b2: float = 0.999, wd: float = 1e-4):
        self.t += 1; c1 = 1 - b1**self.t; c2 = 1 - b2**self.t
        def adam(p: np.ndarray, g: np.ndarray, m: np.ndarray, v: np.ndarray) -> None:
            g = g + wd * p
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            p -= lr * (m / c1) / ((v / c2) ** 0.5 + 1e-8)
        for l in self.D: adam(l.W, l.dW, l.mW, l.vW); adam(l.b, l.db, l.mb, l.vb)
        for bn in self.B: adam(bn.g, bn.dg, bn.mg, bn.vg); adam(bn.b, bn.db2, bn.mb2, bn.vb2)

    def snapshot(self):
        return ([(l.W.copy(), l.b.copy()) for l in self.D],
                [(b.g.copy(), b.b.copy(), b.rm.copy(), b.rv.copy()) for b in self.B])

    def load(self, snap):
        for i, (W, b) in enumerate(snap[0]): self.D[i].W = W; self.D[i].b = b
        for i, (g, b, rm, rv) in enumerate(snap[1]): self.B[i].g = g; self.B[i].b = b; self.B[i].rm = rm; self.B[i].rv = rv

# ── Data pipeline helpers ────────────────────────────────────────────────
def split_by_day(df: pd.DataFrame, train_split: float, val_split: float) -> SplitData:
    if not 0 < train_split < val_split < 1:
        raise ValueError("Expected 0 < train_split < val_split < 1")

    days = df["day"].unique()
    n_days = len(days)
    train_cutoff = days[int(train_split * n_days)]
    val_cutoff = days[int(val_split * n_days)]

    train_df = df[df["day"] <= train_cutoff]
    val_df = df[(df["day"] > train_cutoff) & (df["day"] <= val_cutoff)]
    test_df = df[df["day"] > val_cutoff]
    return SplitData(train=train_df, val=val_df, test=test_df)


def _coerce_to_annotated_type(value: Any, annotated_type: Any) -> Any:
    """PyYAML's safe_load treats bare exponential notation like `3e-3`
    (no decimal point) as a string, not a float -- a well-known YAML 1.1
    resolver quirk. Config is a plain dataclass, so constructing it from a
    merged dict does NOT coerce types based on annotations; a string `lr`
    silently reaches the Adam optimizer and blows up several calls deep.
    Coerce every loaded value to its declared type here, once, instead of
    trusting the config file's authors to always write `3.0e-3`."""
    if annotated_type in (int, float, str) and not isinstance(value, annotated_type):
        try:
            return annotated_type(value)
        except (TypeError, ValueError):
            return value
    if get_origin(annotated_type) is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def load_config(config_path: str | None) -> Config:
    cfg = Config()
    if not config_path:
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML configs") from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError("Config extension must be .json, .yaml, or .yml")

    if not isinstance(data, dict):
        raise ValueError("Config file must parse to a dictionary/object")

    merged = {**cfg.__dict__, **data}
    hints = get_type_hints(Config)
    merged = {k: _coerce_to_annotated_type(v, hints[k]) if k in hints else v for k, v in merged.items()}
    return Config(**merged)


# ── Training ─────────────────────────────────────────────────────────────
def train(
    net: Net,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    cfg: Config,
    log: logging.Logger,
) -> Dict[str, float]:
    best = np.inf; no_imp = 0; snap = net.snapshot(); lr = cfg.lr
    final_epoch = 0
    for ep in range(cfg.epochs):
        final_epoch = ep + 1
        idx = np.random.permutation(len(Xtr)); tl = 0.0; nb = 0
        for s in range(0, len(Xtr), cfg.batch_size):
            Xb = Xtr[idx[s:s + cfg.batch_size]].T
            yb = ytr[idx[s:s + cfg.batch_size]].reshape(1, -1)
            p = net.fwd(Xb, True)
            tl += float(((p - yb) ** 2).mean()); nb += 1
            net.bwd(2 * (p - yb) / yb.size); net.step(lr)
        vl = float(((net.fwd(Xv.T, False) - yv.reshape(1, -1)) ** 2).mean())
        if vl < best - 1e-7:
            best = vl; no_imp = 0; snap = net.snapshot()
        else:
            no_imp += 1
            if no_imp == cfg.patience // 2: lr *= 0.5
            if no_imp == cfg.patience:
                log.info("Early stop at ep %d", ep + 1); break
        if ep % 20 == 0 or ep < 3:
            log.info("ep %4d  train %.5f  val %.5f  lr %.1e", ep + 1, tl / nb, vl, lr)
    net.load(snap)
    return {"best_val_mse": float(best), "epochs_ran": float(final_epoch), "final_lr": float(lr)}


def save_artifacts(net: Net, scaler: StandardScaler, metrics: Dict[str, float]) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    # net.snapshot() is a ragged nested structure (per-layer arrays of
    # different shapes) -- not a homogeneous array. np.save's allow_pickle
    # used to paper over this by silently boxing it as a 0-d object array;
    # current numpy raises instead. pickle is the right tool for an
    # arbitrary Python object graph like this.
    with (ARTIFACTS_DIR / "net_snapshot.pkl").open("wb") as f:
        pickle.dump(net.snapshot(), f)
    with (ARTIFACTS_DIR / "scaler.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)
    with (ARTIFACTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

# ── Main ─────────────────────────────────────────────────────────────────
def main(cfg: Config, log: logging.Logger):
    np.random.seed(cfg.seed)
    rows = generate_rows(cfg)
    df = featurize(rows)
    splits = split_by_day(df, cfg.train_split, cfg.val_split)
    tr, vl, te = splits.train, splits.val, splits.test
    log.info("Train %d  Val %d  Test %d", len(tr), len(vl), len(te))

    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[FEATS].values.astype(float))
    Xvl = sc.transform(vl[FEATS].values.astype(float))
    Xte = sc.transform(te[FEATS].values.astype(float))
    ytr, yvl, yte = tr["target"].values, vl["target"].values, te["target"].values

    net = Net(len(FEATS), cfg.sizes, cfg.drop_rate)
    train_stats = train(net, Xtr, ytr, Xvl, yvl, cfg, log)

    pred = net.fwd(Xte.T, False).flatten()
    r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    mae = np.abs(yte - pred).mean()
    direction = ((yte > 0) == (pred > 0)).mean()
    log.info("OOS R2 %.4f | MAE %.4f | Dir %.2f%%", r2, mae, direction * 100)

    metrics = {
        "test_r2": float(r2),
        "test_mae": float(mae),
        "test_directional_accuracy": float(direction),
        **train_stats,
    }
    save_artifacts(net, sc, metrics)
    log.info("Artifacts saved to %s/", ARTIFACTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = init_logger(cfg.log_level)
    main(cfg, log)
