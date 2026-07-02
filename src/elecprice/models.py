"""Model zoo: honest baselines first, then increasing complexity.

Every model exposes fit(X, y) / predict(X). Quantile models expose
predict_quantiles(X) -> DataFrame with columns like 'q0.1', 'q0.5', 'q0.9'.

A model only earns its complexity if it beats the seasonal naive baseline
out-of-sample. This is the physics equivalent of requiring significance
above background before claiming a signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # keeps tests runnable without lightgbm installed
    lgb = None

QUANTILES = [0.1, 0.5, 0.9]


class NaiveLag:
    """Predict y(t) = y(t - lag). lag=24 is 'same hour yesterday',
    lag=168 is the seasonal naive 'same hour last week'."""

    def __init__(self, lag: int = 24, target: str = "price_CH"):
        self.col = f"{target}_lag{lag}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLag":
        if self.col not in X.columns:
            raise KeyError(f"{self.col} not in features — run features.build_features first.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.col].to_numpy()


class LGBMPoint:
    """LightGBM point forecast with sensible defaults (tune later, honestly)."""

    def __init__(self, **params):
        defaults = dict(
            objective="regression", n_estimators=500, learning_rate=0.05,
            num_leaves=63, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbosity=-1,
        )
        defaults.update(params)
        self.params = defaults
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMPoint":
        self.model = lgb.LGBMRegressor(**self.params).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class LGBMQuantile:
    """One LightGBM model per quantile. Note: quantile crossing is possible;
    we sort predictions as a simple post-hoc fix and report how often it occurs."""

    def __init__(self, quantiles: list[float] = QUANTILES, **params):
        self.quantiles = quantiles
        self.params = params
        self.models: dict[float, "lgb.LGBMRegressor"] = {}
        self.crossing_rate_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMQuantile":
        for q in self.quantiles:
            defaults = dict(
                objective="quantile", alpha=q, n_estimators=500, learning_rate=0.05,
                num_leaves=63, min_child_samples=50, random_state=42, verbosity=-1,
            )
            defaults.update(self.params)
            self.models[q] = lgb.LGBMRegressor(**defaults).fit(X, y)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        sorted_ = np.sort(raw, axis=1)
        self.crossing_rate_ = float((raw != sorted_).any(axis=1).mean())
        return pd.DataFrame(sorted_, index=X.index,
                            columns=[f"q{q}" for q in self.quantiles])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median as the point forecast."""
        return self.predict_quantiles(X)["q0.5"].to_numpy()
