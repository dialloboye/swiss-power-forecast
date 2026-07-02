"""Feature engineering for day-ahead price forecasting.

Design rule (non-negotiable): every feature for target hour t on delivery day D
must be computable BEFORE the day-ahead auction gate closure (12:00 CET on D-1).
That means:
  - prices: lags of at least 24h (yesterday's auction is known), i.e. D-1, D-2, D-7
  - load / renewables forecasts: allowed for day D (published before gate closure)
  - anything else from day D: forbidden.

This mirrors blinding discipline in particle physics: decide what you are allowed
to look at *before* you look.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "price_CH"

# Lags in hours. 24h = same hour yesterday; 168h = same hour last week.
PRICE_LAGS = [24, 48, 72, 168]
ROLLING_WINDOWS = [24, 168]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    # Cyclical encodings help linear models; harmless for trees.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def add_price_lags(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    out = df.copy()
    for lag in PRICE_LAGS:
        out[f"{target}_lag{lag}"] = out[target].shift(lag)
    for w in ROLLING_WINDOWS:
        # shift(24) first so the window ends at D-1 23:00 — no leakage from day D.
        past = out[target].shift(24)
        out[f"{target}_rollmean{w}"] = past.rolling(w).mean()
        out[f"{target}_rollstd{w}"] = past.rolling(w).std()
    # Neighbour price lags (DE is the dominant driver).
    for col in [c for c in out.columns if c.startswith("price_") and c != target]:
        out[f"{col}_lag24"] = out[col].shift(24)
    return out


def build_features(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned and NaN-free, ready for modelling."""
    out = add_calendar(df)
    out = add_price_lags(out, target=target)
    y = out[target]
    # Drop raw contemporaneous prices (target + neighbours at time t = leakage).
    leak_cols = [c for c in out.columns if c.startswith("price_") and "lag" not in c and "roll" not in c]
    X = out.drop(columns=leak_cols)
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]
