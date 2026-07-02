"""Walk-forward backtesting engine + evaluation metrics.

The only honest way to evaluate a time-series model: expanding window,
refit at each step, predict strictly out-of-sample. No shuffled K-fold,
no peeking. Bootstrap confidence intervals on every headline metric,
because a point estimate of MAE without an error bar is not a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- metrics

def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def pinball(y: np.ndarray, yhat_q: np.ndarray, q: float) -> float:
    diff = y - yhat_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical coverage of a prediction interval. Compare to nominal:
    a q10–q90 interval should cover ~80% — if it covers 60%, the model
    is overconfident regardless of how good its MAE looks."""
    return float(np.mean((y >= lo) & (y <= hi)))


def bootstrap_ci(y: np.ndarray, yhat: np.ndarray, metric=mae,
                 n_boot: int = 2000, level: float = 0.68,
                 block: int = 24, seed: int = 42) -> tuple[float, float]:
    """Block bootstrap CI (blocks of `block` hours to respect autocorrelation)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    n_blocks = max(n // block, 1)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        stats[i] = metric(y[idx], yhat[idx])
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# ---------------------------------------------------------------- engine

@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame            # y_true, y_pred (+ quantile cols)
    metrics: dict = field(default_factory=dict)


def walk_forward(model_factory, X: pd.DataFrame, y: pd.Series,
                 initial_train: str = "365D", step: str = "30D",
                 quantile: bool = False) -> WalkForwardResult:
    """Expanding-window walk-forward evaluation.

    model_factory: zero-arg callable returning a fresh, unfitted model
                   (fresh per fold — no state leaks between folds).
    initial_train: minimum history before the first prediction.
    step:          refit frequency (predict the next `step`, then refit).
    """
    t0 = X.index.min() + pd.Timedelta(initial_train)
    step_td = pd.Timedelta(step)
    frames = []

    t = t0
    while t < X.index.max():
        train = X.index < t
        test = (X.index >= t) & (X.index < t + step_td)
        if test.sum() == 0:
            break
        model = model_factory().fit(X[train], y[train])
        chunk = pd.DataFrame({"y_true": y[test]}, index=X.index[test])
        if quantile:
            qpred = model.predict_quantiles(X[test])
            chunk = chunk.join(qpred)
            chunk["y_pred"] = qpred["q0.5"]
        else:
            chunk["y_pred"] = model.predict(X[test])
        frames.append(chunk)
        t += step_td

    preds = pd.concat(frames)
    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    m = {
        "mae": mae(yt, yp),
        "mae_ci68": bootstrap_ci(yt, yp, mae),
        "rmse": rmse(yt, yp),
        "n_pred": len(preds),
    }
    if quantile:
        for qcol in [c for c in preds.columns if c.startswith("q")]:
            q = float(qcol[1:])
            m[f"pinball_{qcol}"] = pinball(yt, preds[qcol].to_numpy(), q)
        if {"q0.1", "q0.9"}.issubset(preds.columns):
            m["coverage_80"] = coverage(yt, preds["q0.1"].to_numpy(),
                                        preds["q0.9"].to_numpy())
    return WalkForwardResult(predictions=preds, metrics=m)
