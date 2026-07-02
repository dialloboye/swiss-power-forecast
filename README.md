# Swiss Day-Ahead Electricity Price Forecasting

Probabilistic forecasting of Swiss day-ahead electricity prices (EPEX spot, CH bidding zone)
using public ENTSO-E data — with an emphasis on **rigorous backtesting** and
**calibrated uncertainty quantification**.

> **Why this project?** Electricity prices are among the most volatile time series in any
> market: strong multiple seasonality, regime shifts, negative prices, and fat-tailed spikes.
> This makes them an ideal testbed for methods I developed in experimental particle physics —
> extracting non-stationary trends from noisy data with controlled, sub-percent precision
> (LHC luminosity calibration) and quantifying uncertainty honestly.

## Results (snapshot)

*To be updated as the project progresses.*

| Model | MAE (EUR/MWh) | RMSE | Pinball loss (q10/q50/q90) | Coverage 80% PI |
|---|---|---|---|---|
| Naive (D-1) | – | – | – | – |
| Seasonal naive (D-7) | – | – | – | – |
| SARIMAX | – | – | – | – |
| LightGBM (point) | – | – | – | – |
| LightGBM (quantile) | – | – | – | – |

Key figure: *(forecast vs. realised prices with 80% prediction intervals — coming soon)*

## Methodology

1. **Data** — ENTSO-E Transparency Platform (free API): day-ahead prices, load forecast,
   generation forecast (wind/solar), cross-border flows. Hourly resolution, CH zone,
   with DE-LU and FR as exogenous neighbours.
2. **Features** — calendar effects (hour, day-of-week, holidays), lagged prices (D-1, D-2, D-7),
   rolling statistics, load/renewables forecasts available *before* gate closure
   (strict no-look-ahead policy).
3. **Models** — hierarchy of increasing complexity, each judged against honest baselines:
   naive & seasonal naive → SARIMAX → LightGBM point forecast → LightGBM quantile regression.
4. **Backtesting** — expanding-window walk-forward validation. No random shuffling, ever.
   Metrics: MAE, RMSE, pinball loss, empirical coverage of prediction intervals.
5. **Uncertainty** — quantile models are only useful if calibrated: we report empirical
   coverage vs. nominal and reliability diagrams, not just point metrics.

## Limitations (read this first)

- Day-ahead auction prices only; no intraday or balancing markets.
- Fuel prices (gas, CO2) not yet included — a known driver of price levels.
- Results are in-sample-period specific: performance during the backtest window does not
  guarantee performance out of it (regime changes, market coupling changes).

## Project structure

```
src/elecprice/
    data.py        # ENTSO-E ingestion + local parquet caching
    features.py    # feature engineering (leak-free by construction)
    models.py      # baselines, SARIMAX wrapper, LightGBM point & quantile
    backtest.py    # walk-forward engine + metrics
tests/             # unit tests (features, backtest integrity)
notebooks/         # EDA and result notebooks
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add your free ENTSO-E API token
python -m elecprice.data --start 2023-01-01 --end 2026-06-30   # download & cache
pytest                       # run tests
```

Get a free API token: register on [ENTSO-E Transparency](https://transparency.entsoe.eu/),
then email transparency@entsoe.eu requesting RESTful API access.

## Author

**Diallo Boye** — PhD in particle physics (CERN/ATLAS), Goldhaber Distinguished Fellow at
Brookhaven National Laboratory. Background in statistical inference, rare-signal extraction
and precision time-series calibration on LHC data.
