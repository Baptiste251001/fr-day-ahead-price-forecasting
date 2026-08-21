"""Common interface shared by every forecasting model in this package.

Every model is fit and evaluated inside `fr_power.models.walk_forward`,
which only ever calls `fit(X, y)` then `predict(X)`. Point models return a
plain prediction; quantile-capable models additionally expose
`predict_quantiles`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...


@runtime_checkable
class QuantileForecaster(Forecaster, Protocol):
    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame: ...
