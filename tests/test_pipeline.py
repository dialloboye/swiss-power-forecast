"""Tests that guard the two things that silently ruin forecasting projects:
data leakage and broken backtest logic."""
import numpy as np
import pandas as pd
import pytest

from elecprice.features import build_features, add_price_lags, TARGET
from elecprice.backtest import walk_forward, mae, coverage
from elecprice.models import NaiveLag


@pytest.fixture
def synthetic():
    """Two years of hourly synthetic prices with daily+weekly seasonality."""
    idx = pd.date_range("2023-01-01", "2025-01-01", freq="1h", tz="Europe/Zurich")
    rng = np.random.default_rng(0)
    h, d = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    price = 80 + 20 * np.sin(2 * np.pi * h / 24) + 10 * (d < 5) + rng.normal(0, 5, len(idx))
    return pd.DataFrame({TARGET: price, "price_DE_LU": price * 0.9,
                         "load_forecast_CH": 6000 + 500 * np.sin(2 * np.pi * h / 24)},
                        index=idx)


def test_no_leakage_in_lags(synthetic):
    """The 24h lag at time t must equal the raw price at t-24h, never anything later."""
    out = add_price_lags(synthetic)
    t = synthetic.index[100]
    assert out.loc[t, f"{TARGET}_lag24"] == synthetic.loc[t - pd.Timedelta("24h"), TARGET]


def test_rolling_features_exclude_today(synthetic):
    """Rolling mean must be computable at gate closure: perturbing today's prices
    must not change today's rolling feature values."""
    out1 = add_price_lags(synthetic)
    perturbed = synthetic.copy()
    day = synthetic.index[-24:]
    perturbed.loc[day, TARGET] += 1000.0
    out2 = add_price_lags(perturbed)
    pd.testing.assert_series_equal(out1.loc[day, f"{TARGET}_rollmean24"],
                                   out2.loc[day, f"{TARGET}_rollmean24"])


def test_build_features_drops_contemporaneous_prices(synthetic):
    X, y = build_features(synthetic)
    assert TARGET not in X.columns
    assert "price_DE_LU" not in X.columns
    assert len(X) == len(y) > 0


def test_walk_forward_is_out_of_sample(synthetic):
    """Every prediction timestamp must be strictly after the initial training window."""
    X, y = build_features(synthetic)
    res = walk_forward(lambda: NaiveLag(lag=24), X, y,
                       initial_train="180D", step="30D")
    assert res.predictions.index.min() >= X.index.min() + pd.Timedelta("180D")
    assert res.metrics["mae"] > 0


def test_seasonal_naive_beats_random(synthetic):
    """Sanity: on seasonal synthetic data, the naive lag-24 model must clearly
    beat a shuffled prediction — otherwise the engine is wired wrong."""
    X, y = build_features(synthetic)
    res = walk_forward(lambda: NaiveLag(lag=24), X, y,
                       initial_train="180D", step="90D")
    yt = res.predictions["y_true"].to_numpy()
    rng = np.random.default_rng(1)
    shuffled_mae = mae(yt, rng.permutation(yt))
    assert res.metrics["mae"] < 0.5 * shuffled_mae


def test_coverage_metric():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 2.5, 2.0, 3.0])
    hi = np.array([2.0, 3.0, 4.0, 3.5])
    assert coverage(y, lo, hi) == 0.5
