"""Walk-forward backtest of the full model hierarchy.

Run:  python notebooks/02_backtest.py

Produces:
  - markdown results table printed to stdout (paste into README)
  - reports/figures/07_forecast_intervals.png  (the money figure)
  - reports/figures/08_calibration.png         (PI coverage check)

Expect 10-20 min runtime: ~30 LightGBM refits on ~30k rows.
"""
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elecprice.features import build_features, TARGET
from elecprice.models import NaiveLag, LGBMPoint, LGBMQuantile
from elecprice.backtest import walk_forward, pinball

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
from elecprice.conformal import CQRQuantile

# ------------------------------------------------------------------ data
df = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
df = df.interpolate(limit=3, limit_direction="both")
X, y = build_features(df)
print(f"[backtest] features: {X.shape[1]} cols, {len(X)} rows "
      f"({X.index.min().date()} -> {X.index.max().date()})")

# ------------------------------------------------------------------ models
# Same walk-forward protocol for every model: expanding window,
# first prediction after 365 days of history, refit every 30 days.
RUNS = {
    "Naive (D-1)":        dict(factory=lambda: NaiveLag(lag=24),  quantile=False),
    "Seasonal naive (D-7)": dict(factory=lambda: NaiveLag(lag=168), quantile=False),
    "LightGBM (point)":   dict(factory=LGBMPoint,                 quantile=False),
    "LightGBM (quantile)": dict(factory=LGBMQuantile,             quantile=True),
    "LightGBM + CQR": dict(factory=CQRQuantile, quantile=True),
}

results, preds_q = {}, None
for name, cfg in RUNS.items():
    t0 = time.time()
    res = walk_forward(cfg["factory"], X, y,
                       initial_train="365D", step="30D",
                       quantile=cfg["quantile"])
    results[name] = res.metrics
    if cfg["quantile"]:
        preds_q = res.predictions
    print(f"[backtest] {name}: MAE={res.metrics['mae']:.2f} "
          f"({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ table
def fmt(m):
    lo, hi = m["mae_ci68"]
    mae = f"{m['mae']:.2f} [{lo:.2f}, {hi:.2f}]"
    pin = "–"
    if "pinball_q0.1" in m:
        pin = f"{m['pinball_q0.1']:.2f} / {m['pinball_q0.5']:.2f} / {m['pinball_q0.9']:.2f}"
    cov = f"{m['coverage_80']*100:.1f}%" if "coverage_80" in m else "–"
    return f"| {mae} | {m['rmse']:.2f} | {pin} | {cov} |"

print("\n--- paste into README ---")
print("| Model | MAE (EUR/MWh) [68% CI] | RMSE | Pinball (q10/q50/q90) | Coverage 80% PI |")
print("|---|---|---|---|---|")
for name, m in results.items():
    print(f"| {name} {fmt(m)}")
print("--- end ---\n")

# ------------------------------------------------------------------ money figure
# Show the last 14 days of quantile predictions vs realized prices.
#window = preds_q.last("14D") # deprecated
window = preds_q.loc[preds_q.index.max() - pd.Timedelta("14D"):]
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.fill_between(window.index, window["q0.1"], window["q0.9"],
                color="#A8C3D9", alpha=0.6, label="80% prediction interval")
ax.plot(window.index, window["q0.5"], color="#33658D", lw=1.2, label="median forecast")
ax.plot(window.index, window["y_true"], color="#16222C", lw=1.0,
        ls="-", alpha=0.85, label="realized price")
ax.set_title("Out-of-sample day-ahead forecast with 80% prediction intervals — last 14 days")
ax.set_ylabel("EUR/MWh"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "07_forecast_intervals.png", dpi=150)

# ------------------------------------------------------------------ calibration
# Empirical coverage per nominal quantile: are our quantiles honest?
fig, ax = plt.subplots(figsize=(5.5, 5))
qs = [0.1, 0.5, 0.9]
emp = [(preds_q["y_true"] <= preds_q[f"q{q}"]).mean() for q in qs]
ax.plot([0, 1], [0, 1], color="#C13527", ls="--", lw=1, label="perfect calibration")
ax.scatter(qs, emp, color="#16222C", zorder=3, s=45)
for q, e in zip(qs, emp):
    ax.annotate(f"  {e:.2f}", (q, e), fontsize=9)
ax.set_xlabel("nominal quantile"); ax.set_ylabel("empirical frequency")
ax.set_title("Quantile calibration (out-of-sample)")
ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(FIG / "08_calibration.png", dpi=150)

print(f"[backtest] figures saved to {FIG}")
if getattr(results.get("LightGBM (quantile)"), "crossing_rate_", None):
    print(f"[backtest] quantile crossing rate: {results['LightGBM (quantile)'].crossing_rate_:.3%}")
