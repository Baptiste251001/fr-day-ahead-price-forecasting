"""Generic walk-forward evaluation harness.

Every model in this package is evaluated the same way: never a single
train/test split. Starting from an initial training window, each model is
fit on all data available so far and used to predict the next
`test_window_hours`; the training window then expands and the model is
retrained. This is deliberately an *expanding*-window scheme (not a
rolling fixed-size one) so later folds always benefit from the full
history available at that point in real trading.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from fr_power.models.base import Forecaster

logger = logging.getLogger(__name__)


def walk_forward_predict(
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_factories: dict[str, Callable[[], Forecaster]],
    initial_train_hours: int,
    retrain_every_hours: int,
    test_window_hours: int,
) -> dict[str, pd.Series]:
    """Produce out-of-sample predictions for each model in `model_factories`.

    Returns one prediction Series per model name, indexed like
    `feature_matrix` with NaN outside the periods that were ever in a test
    window (e.g. the initial training period, or any gap if
    `retrain_every_hours` exceeds `test_window_hours`).
    """
    index = feature_matrix.index
    n = len(feature_matrix)
    predictions = {name: pd.Series(np.nan, index=index, name=name) for name in model_factories}

    if initial_train_hours >= n:
        logger.warning(
            "initial_train_hours (%d) >= number of rows (%d); no out-of-sample predictions "
            "can be produced.",
            initial_train_hours,
            n,
        )
        return predictions

    train_end = initial_train_hours
    n_folds = 0
    while train_end < n:
        test_end = min(train_end + test_window_hours, n)
        train_slice = feature_matrix.iloc[:train_end]
        test_slice = feature_matrix.iloc[train_end:test_end]

        X_train, y_train = train_slice[feature_cols], train_slice[target_col]
        X_test = test_slice[feature_cols]

        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions[name].loc[test_slice.index] = preds

        n_folds += 1
        train_end += retrain_every_hours

    logger.info("Walk-forward evaluation completed: %d fold(s).", n_folds)
    return predictions
