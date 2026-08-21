"""Tests for OU calibration, verified against synthetic paths with known parameters."""

import numpy as np
import pandas as pd
import pytest

from fr_power.ou.estimate import compute_z_score, estimate_ou, estimate_ou_mle, estimate_ou_ols


def _simulate_ou_path(
    kappa: float, mu: float, sigma: float, dt: float, n: int, x0: float, seed: int
) -> np.ndarray:
    """Exact discretization of dX = kappa(mu - X)dt + sigma dW."""
    rng = np.random.default_rng(seed)
    a = np.exp(-kappa * dt)
    std_eps = sigma * np.sqrt((1 - a**2) / (2 * kappa))
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = mu + a * (x[t - 1] - mu) + std_eps * rng.standard_normal()
    return x


def _synthetic_series(
    kappa: float = 0.1,
    mu: float = 5.0,
    sigma: float = 2.0,
    dt: float = 1.0,
    n: int = 5000,
    seed: int = 0,
) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="Europe/Paris")
    values = _simulate_ou_path(kappa, mu, sigma, dt, n, x0=mu, seed=seed)
    return pd.Series(values, index=index)


@pytest.mark.parametrize("method_fn", [estimate_ou_ols, estimate_ou_mle])
def test_ou_estimation_recovers_known_parameters(method_fn) -> None:
    true_kappa, true_mu, true_sigma = 0.1, 5.0, 2.0
    series = _synthetic_series(true_kappa, true_mu, true_sigma, n=5000, seed=1)

    params = method_fn(series, dt=1.0)

    assert params.kappa == pytest.approx(true_kappa, rel=0.2)
    assert params.mu == pytest.approx(true_mu, rel=0.2, abs=0.3)
    assert params.sigma == pytest.approx(true_sigma, rel=0.15)
    assert params.half_life == pytest.approx(np.log(2) / true_kappa, rel=0.2)
    assert 0.0 < params.r_squared < 1.0
    assert params.n_observations == len(series) - 1


def test_ols_and_mle_agree_on_kappa_and_mu_but_not_sigma() -> None:
    series = _synthetic_series(seed=2)
    ols = estimate_ou_ols(series)
    mle = estimate_ou_mle(series)

    assert ols.kappa == pytest.approx(mle.kappa, rel=1e-9)
    assert ols.mu == pytest.approx(mle.mu, rel=1e-9)
    assert ols.sigma != pytest.approx(mle.sigma)  # different variance denominators (n-2 vs n)


def test_estimate_ou_dispatches_by_method_name() -> None:
    series = _synthetic_series(seed=3)
    assert estimate_ou(series, method="ols").method == "ols"
    assert estimate_ou(series, method="mle").method == "mle"


def test_estimate_ou_rejects_unknown_method() -> None:
    series = _synthetic_series(seed=4)
    with pytest.raises(ValueError, match="Unknown OU estimation method"):
        estimate_ou(series, method="bogus")  # type: ignore[arg-type]


def test_estimate_ou_requires_minimum_observations() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Need at least"):
        estimate_ou_ols(series, min_observations=20)


def test_estimate_ou_rejects_non_mean_reverting_series() -> None:
    # Deterministically explosive AR(1) (b > 1) -- not representable as a valid OU process.
    n = 200
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="Europe/Paris")
    rng = np.random.default_rng(0)
    x = np.empty(n)
    x[0] = 1.0
    for t in range(1, n):
        x[t] = 1.05 * x[t - 1] + rng.standard_normal() * 0.1
    series = pd.Series(x, index=index)

    with pytest.raises(ValueError, match="no valid mean reversion"):
        estimate_ou_ols(series)


def test_compute_z_score_matches_manual_calculation() -> None:
    series = _synthetic_series(seed=5, n=1000)
    params = estimate_ou_ols(series)
    z = compute_z_score(series, params)
    expected = ((series - params.mu) / params.stationary_std).rename("z_score")
    pd.testing.assert_series_equal(z, expected)
