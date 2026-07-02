"""ENTSO-E data ingestion with local parquet caching.

Downloads day-ahead prices, load forecasts and renewable generation forecasts
for the Swiss bidding zone (plus neighbours as exogenous drivers), and caches
everything as parquet so the API is only hit once per (series, period).

Usage:
    python -m elecprice.data --start 2023-01-01 --end 2026-06-30
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TZ = "Europe/Zurich"

ZONES = {
    "CH": "CH",       # target market
    "DE_LU": "DE_LU", # main price driver via imports
    "FR": "FR",
}


def _client() -> EntsoePandasClient:
    load_dotenv()
    token = os.environ.get("ENTSOE_API_KEY")
    if not token:
        raise RuntimeError("Set ENTSOE_API_KEY in your .env file (see .env.example).")
    return EntsoePandasClient(api_key=token)


def _cached(name: str, fetch_fn, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | pd.DataFrame:
    """Fetch a series from cache if present, otherwise from the API."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}_{start.date()}_{end.date()}.parquet"
    if path.exists():
        return pd.read_parquet(path).squeeze()
    obj = fetch_fn(start=start, end=end)
    frame = obj.to_frame(name=name) if isinstance(obj, pd.Series) else obj
    frame.to_parquet(path)
    return obj


def download(start: str, end: str) -> pd.DataFrame:
    """Download all series and return a single hourly DataFrame indexed in Europe/Zurich."""
    client = _client()
    t0 = pd.Timestamp(start, tz=TZ)
    t1 = pd.Timestamp(end, tz=TZ)

    out: dict[str, pd.Series] = {}
    for label, zone in ZONES.items():
        out[f"price_{label}"] = _cached(
            f"price_{label}",
            lambda start, end, z=zone: client.query_day_ahead_prices(z, start=start, end=end),
            t0, t1,
        )

    out["load_forecast_CH"] = _cached(
        "load_forecast_CH",
        lambda start, end: client.query_load_forecast("CH", start=start, end=end).squeeze(),
        t0, t1,
    )

    # Wind & solar forecasts for DE (dominant renewables driver in the region).
    ws = _cached(
        "wind_solar_forecast_DE",
        lambda start, end: client.query_wind_and_solar_forecast("DE_LU", start=start, end=end),
        t0, t1,
    )
    if isinstance(ws, pd.DataFrame):
        for col in ws.columns:
            out[f"de_{col.lower().replace(' ', '_')}"] = ws[col]

    df = pd.DataFrame(out)
    df.index = df.index.tz_convert(TZ)
    # 15-min series -> hourly means; prices are already hourly
    df = df.resample("1h").mean()
    return df


def load_dataset(start: str, end: str) -> pd.DataFrame:
    """Public entry point used by notebooks and the backtest."""
    df = download(start, end)
    # Basic sanity: report but do not silently fill missing values.
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        print(f"[data] warning: {n_missing} missing values across {df.shape} — inspect before modelling.")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()
    df = load_dataset(args.start, args.end)
    out = RAW_DIR.parent / "processed" / "dataset.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"[data] saved {df.shape[0]} rows x {df.shape[1]} cols -> {out}")
