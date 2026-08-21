"""Tests for forecast accuracy metrics and the Diebold-Mariano test."""

import numpy as np
import pandas as pd
import pytest

from fr_power.models.metrics import diebold_mariano_test, evaluate_predictions, mae, mdae, rmse


def test_rmse_mae_mdae_on_known_errors() -> None:
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0])
    y_pred = pd.Series([12.0, 18.0, 33.0, 40.0])  # errors: 2, -2, 3, 0

    assert mae(y_true, y_pred) == pytest.approx(1.75)
    assert mdae(y_true, y_pred) == pytest.approx(2.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 4 + 9 + 0) / 4))


def test_metrics_drop_nan_pairs() -> None:
    y_true = pd.Series([10.0, np.nan, 30.0], index=[0, 1, 2])
    y_pred = pd.Series([12.0, 5.0, np.nan], index=[0, 1, 2])
    # only index 0 has both values present -> error = 2
    assert mae(y_true, y_pred) == pytest.approx(2.0)


def test_evaluate_predictions_bundle() -> None:
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 3.0])
    out = evaluate_predictions(y_true, y_pred)
    assert out == {"rmse": 0.0, "mae": 0.0, "mdae": 0.0}


def test_diebold_mariano_identical_forecasts_gives_zero_stat() -> None:
    rng = np.random.default_rng(0)
    errors = rng.normal(size=200)
    dm_stat, p_value = diebold_mariano_test(errors, errors)
    assert dm_stat == pytest.approx(0.0, abs=1e-8)
    assert p_value == pytest.approx(1.0, abs=1e-6)


def test_diebold_mariano_favors_lower_error_model() -> None:
    rng = np.random.default_rng(1)
    n = 500
    errors_a = rng.normal(scale=1.0, size=n)  # smaller errors
    errors_b = rng.normal(scale=5.0, size=n)  # much larger errors

    dm_stat, p_value = diebold_mariano_test(errors_a, errors_b, power=2)

    assert dm_stat < 0  # negative favors model A (lower loss)
    assert p_value < 0.01


def test_diebold_mariano_requires_matching_shapes() -> None:
    with pytest.raises(ValueError):
        diebold_mariano_test(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_diebold_mariano_requires_min_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        diebold_mariano_test(np.array([1.0]), np.array([1.0]))
