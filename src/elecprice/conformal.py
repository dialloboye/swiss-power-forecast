"""Conformalized Quantile Regression (CQR) — Romano, Patterson & Candès (2019).

Problem it solves: our quantile model is well-calibrated in location but
overconfident in scale (56.5% empirical coverage vs 80% nominal).

Idea: hold out a calibration window from the training data. Measure how far
outside the predicted interval the truth falls on that window (conformity
scores). Take the appropriate quantile of those scores and widen the interval
by exactly that margin. Result: finite-sample coverage guarantee under
exchangeability — the interval is honest *by construction*, whatever the
base model got wrong.

Physics analogy: it is an in-situ calibration — measure your resolution on a
control region, then propagate it, instead of trusting the simulation.

Caveat we accept and document: hourly prices are not exchangeable (they are
autocorrelated and regime-switching), so the theoretical guarantee is only
approximate here. The empirical coverage in the walk-forward backtest is the
real test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import LGBMQuantile


class CQRQuantile:
    """Wraps a quantile model; widens its [lo, hi] interval using split-conformal
    calibration on the most recent `cal_hours` of the training window.

    Compatible with backtest.walk_forward(quantile=True).
    """

    def __init__(self, base_factory=LGBMQuantile, cal_hours: int = 90 * 24,
                 alpha: float = 0.2, **base_params):
        self.base_factory = base_factory
        self.base_params = base_params
        self.cal_hours = cal_hours
        self.alpha = alpha          # 1 - nominal coverage (0.2 -> 80% PI)
        self.correction_: float | None = None
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CQRQuantile":
        if len(X) <= self.cal_hours + 24:
            raise ValueError("Training window too short for the calibration split.")
        # Temporal split: fit on the past, calibrate on the most recent window.
        X_fit, y_fit = X.iloc[:-self.cal_hours], y.iloc[:-self.cal_hours]
        X_cal, y_cal = X.iloc[-self.cal_hours:], y.iloc[-self.cal_hours:]

        self.model = self.base_factory(**self.base_params).fit(X_fit, y_fit)

        q = self.model.predict_quantiles(X_cal)
        lo, hi = q.iloc[:, 0].to_numpy(), q.iloc[:, -1].to_numpy()
        # Conformity score: signed distance outside the interval (<=0 if inside).
        scores = np.maximum(lo - y_cal.to_numpy(), y_cal.to_numpy() - hi)
        n = len(scores)
        # Finite-sample corrected quantile of the scores.
        level = min((1 - self.alpha) * (1 + 1 / n), 1.0)
        self.correction_ = float(np.quantile(scores, level))
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        q = self.model.predict_quantiles(X).copy()
        lo_col, hi_col = q.columns[0], q.columns[-1]
        q[lo_col] = q[lo_col] - self.correction_
        q[hi_col] = q[hi_col] + self.correction_
        return q

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(X)["q0.5"].to_numpy()
