"""Benchmark forecasters: naive persistence and LEAR (Lasso-Estimated AutoRegressive).

Both are notoriously hard to beat in electricity price forecasting -- they
are the bar the LightGBM and LSTM models (see `gbm.py`, `lstm.py`) have to
clear, not a formality to skip past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_TO_LAG_PREFIX = {
    "target_spread_eur_mwh": "spread",
    "target_day_ahead_price_eur_mwh": "day_ahead_price",
}


def naive_lag_column(target_col: str, lag_hours: int) -> str:
    """Name of the pre-built lag feature (see `fr_power.features.build`) that a naive
    persistence forecast for `target_col` should read its prediction from."""
    if target_col not in TARGET_TO_LAG_PREFIX:
        raise ValueError(f"Unknown target column: {target_col!r}")
    return f"{TARGET_TO_LAG_PREFIX[target_col]}_lag_{lag_hours}h"


class NaivePersistenceForecaster:
    """Predicts the value observed exactly `lag_hours` before each row.

    This reads a pre-built lag feature column (from `fr_power.features.build`)
    rather than re-deriving it, so it needs no fitting.
    """

    def __init__(self, lag_column: str) -> None:
        self.lag_column = lag_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.lag_column not in X.columns:
            raise ValueError(f"Naive forecaster expects feature column {self.lag_column!r}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.lag_column].rename("prediction")


class LearForecaster:
    """Lasso-Estimated AutoRegressive model: standardized linear regression with an
    L1 penalty, alpha selected by time-ordered cross-validation (`TimeSeriesSplit`)."""

    def __init__(self, alpha_grid: list[float], cv_folds: int, random_seed: int = 42) -> None:
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.feature_columns_: list[str] = []
        self.pipeline_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_columns_ = list(X.columns)
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[mask].astype(float)
        y_clean = y.loc[mask].astype(float)
        if len(X_clean) < self.cv_folds + 1:
            raise ValueError(
                f"Not enough complete rows ({len(X_clean)}) to fit LEAR with "
                f"cv_folds={self.cv_folds}"
            )

        cv = TimeSeriesSplit(n_splits=self.cv_folds)
        model = LassoCV(
            alphas=self.alpha_grid, cv=cv, max_iter=10_000, random_state=self.random_seed
        )
        self.pipeline_ = make_pipeline(StandardScaler(), model)
        self.pipeline_.fit(X_clean, y_clean)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline_ is None:
            raise RuntimeError("LearForecaster.fit() must be called before predict()")
        X = X[self.feature_columns_].astype(float)
        preds = pd.Series(np.nan, index=X.index, name="prediction")
        mask = X.notna().all(axis=1)
        if mask.any():
            preds.loc[mask] = self.pipeline_.predict(X.loc[mask])
        return preds
