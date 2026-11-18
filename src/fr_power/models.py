"""Forecasting models and the expanding-window walk-forward harness.

Every model exposes `fit(X, y)` / `predict(X) -> pd.Series`:

* `NaiveForecaster` -- the standard EPF naive benchmark: the price of the same hour one
  week earlier on Mondays and weekends, one day earlier otherwise.
* `LassoForecaster` -- standardized linear regression with an L1 penalty, alpha chosen by
  time-ordered cross-validation. A LEAR-style model: a single regression across hours on
  the shared feature set, rather than LEAR's 24 separate hourly models.
* `LightGBMForecaster` -- gradient-boosted trees, early-stopped on the most recent 10% of
  the training window (never a random split).
* `ExtraTreesForecaster` -- extremely randomized trees, a robust benchmark.

`add_ensemble` combines models by their hour-by-hour median.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from fr_power.config import Config, ExtraTreesConfig, LassoConfig, LightGBMConfig

logger = logging.getLogger(__name__)


class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...


class NaiveForecaster:
    DAILY, WEEKLY = "price_lag_24h", "price_lag_168h"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        missing = {self.DAILY, self.WEEKLY} - set(X.columns)
        if missing:
            raise ValueError(f"Naive forecaster needs feature columns {missing}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        weekly = X["day_of_week"].isin([0, 5, 6])  # Monday, Saturday, Sunday
        return X[self.WEEKLY].where(weekly, X[self.DAILY]).rename("prediction")


class LassoForecaster:
    def __init__(self, config: LassoConfig, random_seed: int = 42) -> None:
        self.config = config
        self.random_seed = random_seed
        self.pipeline_: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        complete = X.notna().all(axis=1) & y.notna()  # linear model needs complete rows
        self.pipeline_ = make_pipeline(
            StandardScaler(),
            LassoCV(
                alphas=self.config.alpha_grid,
                cv=TimeSeriesSplit(n_splits=self.config.cv_folds),
                max_iter=10_000,
                random_state=self.random_seed,
            ),
        )
        self.pipeline_.fit(X[complete].astype(float), y[complete])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline_ is None:
            raise RuntimeError("fit() must be called before predict()")
        preds = pd.Series(np.nan, index=X.index, name="prediction")
        complete = X.notna().all(axis=1)
        if complete.any():
            preds[complete] = self.pipeline_.predict(X[complete].astype(float))
        return preds


class LightGBMForecaster:
    def __init__(self, config: LightGBMConfig, random_seed: int = 42) -> None:
        self.config = config
        self.random_seed = random_seed
        self.model_: lgb.LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X, y = X[y.notna()], y[y.notna()]
        c = self.config
        self.model_ = lgb.LGBMRegressor(
            n_estimators=c.n_estimators,
            learning_rate=c.learning_rate,
            num_leaves=c.num_leaves,
            min_child_samples=c.min_child_samples,
            subsample=c.subsample,
            subsample_freq=1,
            colsample_bytree=c.colsample_bytree,
            random_state=self.random_seed,
            deterministic=True,
            force_row_wise=True,
            verbosity=-1,
        )
        n_val = max(1, len(X) // 10)  # early stopping on the most recent 10%
        self.model_.fit(
            X.iloc[:-n_val],
            y.iloc[:-n_val],
            eval_X=X.iloc[-n_val:],
            eval_y=y.iloc[-n_val:],
            callbacks=[lgb.early_stopping(c.early_stopping_rounds, verbose=False)],
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("fit() must be called before predict()")
        return pd.Series(self.model_.predict(X), index=X.index, name="prediction")


class ExtraTreesForecaster:
    def __init__(self, config: ExtraTreesConfig, random_seed: int = 42) -> None:
        self.config = config
        self.random_seed = random_seed
        self.model_: ExtraTreesRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        c = self.config
        self.model_ = ExtraTreesRegressor(
            n_estimators=c.n_estimators,
            min_samples_leaf=c.min_samples_leaf,
            max_features=c.max_features,
            n_jobs=-1,
            random_state=self.random_seed,
        )
        self.model_.fit(X[y.notna()], y[y.notna()])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("fit() must be called before predict()")
        return pd.Series(self.model_.predict(X), index=X.index, name="prediction")


def build_models(cfg: Config) -> dict[str, Callable[[], Forecaster]]:
    """Fresh-model factories, one per model name, configured from `cfg`."""
    m, seed = cfg.models, cfg.random_seed
    return {
        "naive": NaiveForecaster,
        "lasso": lambda: LassoForecaster(m.lasso, seed),
        "lightgbm": lambda: LightGBMForecaster(m.lightgbm, seed),
        "extra_trees": lambda: ExtraTreesForecaster(m.extra_trees, seed),
    }


def add_ensemble(predictions: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Append the hour-by-hour median of `members` as an `ensemble` column."""
    out = predictions.copy()
    out["ensemble"] = predictions[members].median(axis=1, skipna=False)
    return out


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Callable[[], Forecaster]],
    initial_train_days: int,
    retrain_every_days: int,
) -> pd.DataFrame:
    """Expanding-window out-of-sample predictions, one column per model.

    Folds start at local midnight: each model is fitted on every row before the fold start,
    predicts the next `retrain_every_days` delivery days, then the window expands and it is
    refitted from scratch. Aligning folds on days guarantees no model ever sees a price
    from the auction it is forecasting. Rows in the initial training window stay NaN.
    """
    predictions = pd.DataFrame(np.nan, index=X.index, columns=list(models))
    step = pd.DateOffset(days=retrain_every_days)  # calendar days, DST-safe
    fold_starts = pd.date_range(
        X.index.min().normalize() + pd.DateOffset(days=initial_train_days),
        X.index.max(),
        freq=step,
    )
    for fold, start in enumerate(fold_starts, start=1):
        train = X.index < start
        test = (X.index >= start) & (X.index < start + step)
        for name, make_model in models.items():
            model = make_model()
            model.fit(X[train], y[train])
            predictions.loc[test, name] = model.predict(X[test]).to_numpy()
        logger.info("fold %d/%d done (%s)", fold, len(fold_starts), start.date())
    return predictions
