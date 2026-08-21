"""LightGBM forecaster.

Gradient-boosted trees handle NaN feature values natively (missing-value
splits), so unlike LEAR only the target needs to be non-null for training.
A trailing, chronological slice of the training window is held out
internally for early stopping -- never a random split, which would leak
future information into model selection.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd


class LightGBMForecaster:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        num_leaves: int,
        max_depth: int,
        min_child_samples: int,
        subsample: float,
        colsample_bytree: float,
        early_stopping_rounds: int,
        validation_fraction: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_seed = random_seed
        self.feature_columns_: list[str] = []
        self.model_: lgb.LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_columns_ = list(X.columns)
        mask = y.notna()
        X_clean, y_clean = X.loc[mask], y.loc[mask]

        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_seed,
            verbosity=-1,
        )

        n_val = int(len(X_clean) * self.validation_fraction)
        if n_val >= 1 and (len(X_clean) - n_val) >= 1:
            X_train, y_train = X_clean.iloc[:-n_val], y_clean.iloc[:-n_val]
            X_val, y_val = X_clean.iloc[-n_val:], y_clean.iloc[-n_val:]
            self.model_.fit(
                X_train,
                y_train,
                eval_X=X_val,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
        else:
            self.model_.fit(X_clean, y_clean)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("LightGBMForecaster.fit() must be called before predict()")
        preds = self.model_.predict(X[self.feature_columns_])
        return pd.Series(preds, index=X.index, name="prediction")
