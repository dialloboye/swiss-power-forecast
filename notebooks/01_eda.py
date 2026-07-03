"""Exploratory data analysis — produces the figures for reports/figures/.

Run:  python notebooks/01_eda.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")

# --- gap handling: interpolate only short gaps (<=3h), document the rest ---
n_before = int(df.isna().sum().sum())
df = df.interpolate(limit=3, limit_direction="both")
print(f"[eda] filled {n_before - int(df.isna().sum().sum())} of {n_before} missing values (gaps <= 3h)")

p = df["price_CH"]

# ---------------------------------------------------------------- fig 1: full history
fig, ax = plt.subplots(figsize=(12, 4))
p.resample("1D").mean().plot(ax=ax, lw=0.8, color="#33658D")
ax.set_title("Swiss day-ahead price — daily mean, 2023–2026")
ax.set_ylabel("EUR/MWh"); ax.set_xlabel("")
ax.axhline(0, color="#C13527", lw=0.8, ls="--")
fig.tight_layout(); fig.savefig(FIG / "01_price_history.png", dpi=150)

# ---------------------------------------------------------------- fig 2: distribution & negative prices
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(p.dropna(), bins=150, color="#A8C3D9", edgecolor="none")
ax.hist(p[p < 0].dropna(), bins=30, color="#C13527", edgecolor="none",
        label=f"negative prices: {(p < 0).mean()*100:.1f}% of hours")
ax.set_yscale("log")
ax.set_title("Price distribution (log scale) — note the asymmetric tails")
ax.set_xlabel("EUR/MWh"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "02_distribution.png", dpi=150)
print(f"[eda] negative-price hours: {(p < 0).sum()} ({(p < 0).mean()*100:.2f}%)")
print(f"[eda] price > 200 EUR/MWh: {(p > 200).sum()} hours")

# ---------------------------------------------------------------- fig 3: hourly profile by season
fig, ax = plt.subplots(figsize=(8, 4))
seasons = {"Winter (DJF)": [12, 1, 2], "Spring (MAM)": [3, 4, 5],
           "Summer (JJA)": [6, 7, 8], "Autumn (SON)": [9, 10, 11]}
colors = ["#16222C", "#33658D", "#C13527", "#A8C3D9"]
for (label, months), c in zip(seasons.items(), colors):
    prof = p[p.index.month.isin(months)].groupby(p[p.index.month.isin(months)].index.hour).mean()
    ax.plot(prof.index, prof.values, label=label, color=c, lw=1.8)
ax.set_title("Mean price by hour of day — the solar 'duck curve' emerges in summer")
ax.set_xlabel("hour"); ax.set_ylabel("EUR/MWh"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "03_hourly_profile.png", dpi=150)

# ---------------------------------------------------------------- fig 4: rolling volatility
fig, ax = plt.subplots(figsize=(12, 4))
p.rolling(24 * 30).std().plot(ax=ax, color="#33658D", lw=1)
ax.set_title("30-day rolling std of hourly prices — volatility regimes")
ax.set_ylabel("EUR/MWh"); ax.set_xlabel("")
fig.tight_layout(); fig.savefig(FIG / "04_volatility.png", dpi=150)

# ---------------------------------------------------------------- fig 5: CH vs neighbours
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, other in zip(axes, ["price_DE_LU", "price_FR"]):
    ax.scatter(df[other], p, s=1.5, alpha=0.15, color="#33658D")
    lim = [-200, 350]
    ax.plot(lim, lim, color="#C13527", lw=1, ls="--")
    r = df[[other, "price_CH"]].corr().iloc[0, 1]
    ax.set_title(f"CH vs {other.replace('price_', '')}  (r = {r:.2f})")
    ax.set_xlabel(f"{other} (EUR/MWh)"); ax.set_ylabel("price_CH")
    ax.set_xlim(lim); ax.set_ylim(lim)
fig.tight_layout(); fig.savefig(FIG / "05_neighbours.png", dpi=150)

# ---------------------------------------------------------------- fig 6: German solar vs CH price (midday)
midday = df[df.index.hour.isin([12, 13, 14])]
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(midday["de_solar"] / 1e3, midday["price_CH"], s=2, alpha=0.2, color="#33658D")
ax.set_title("German solar forecast vs Swiss midday price — imports matter")
ax.set_xlabel("DE solar forecast (GW)"); ax.set_ylabel("price_CH (EUR/MWh)")
ax.axhline(0, color="#C13527", lw=0.8, ls="--")
fig.tight_layout(); fig.savefig(FIG / "06_solar_effect.png", dpi=150)

print(f"[eda] 6 figures saved to {FIG}")
