# French Day-Ahead Electricity Price Forecasting

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Hour-by-hour forecasts of the French day-ahead electricity price (SDAC auction), built from
free public data (ENTSO-E Transparency Platform, Open-Meteo, Yahoo Finance) and evaluated
with a strict **walk-forward** protocol over 2021–2026. This period covers the 2022 energy
crisis and the surge in negative prices.

The project focuses on doing the basics rigorously:

- **Gate-aligned information set.** Every input must be public before the 12:00 auction
  gate on D-1. Tests check this by perturbing future data.
- **Strong benchmarks.** Models are compared with Diebold–Mariano tests.
- **Calibrated prediction intervals.** Their real coverage is reported, not only the
  nominal one.

**Key results** (51,246 out-of-sample hours, January 2021 to November 2026):

| Model | MAE (€/MWh) | RMSE (€/MWh) | rMAE vs naive | 90% interval coverage |
|---|---:|---:|---:|---:|
| naive | 29.41 | 53.21 | 1.00 | 86.9% |
| **lasso** | **24.58** | **41.01** | **0.84** | 87.4% |
| lightgbm | 35.53 | 60.95 | 1.21 | 85.1% |
| extra_trees | 26.36 | 43.10 | 0.90 | 86.6% |
| ensemble (median) | 25.59 | 42.56 | 0.87 | 86.7% |

- **The Lasso is the best model over the full period.** It is 16% more accurate than the
  naive benchmark, and better than every other model (Diebold–Mariano p < 0.001).
- **Since 2023, the median ensemble is the most accurate model in every year.** In
  2021–2022, tree models broke down because prices rose far above anything in their
  training data.

![MAE by year](reports/figures/mae_by_year.png)

---

## Contents

- [The forecasting problem](#the-forecasting-problem)
- [Method](#method)
- [Results](#results)
- [Quickstart](#quickstart)
- [Project structure](#project-structure)
- [Limitations and next steps](#limitations-and-next-steps)

## The forecasting problem

Every day before 12:00 CET, buyers and sellers submit bids for each hour of the next day to
the European day-ahead auction. Forecasting those 24 prices on the morning of D-1 is the
canonical electricity price forecasting (EPF) task.

France is an interesting case:

- roughly two-thirds of its electricity is nuclear;
- demand is very temperature-sensitive because of electric heating;
- the market went through extreme regimes: the 2021–2022 crisis (gas shock combined with
  record-low nuclear availability), then a return to lower prices with frequent negative
  prices driven by solar.

## Method

### Information set: what is known on D-1 before the 12:00 gate

The usual way a price model "cheats" is by using data that is not yet available when the
forecast is made. A common example in the literature is the TSO's day-ahead wind and solar
forecasts. ENTSO-E publishes them for day D only *after* the auction (RTE at about 16:15,
legal deadline 18:00), so they cannot be used as inputs for day D.

| Feature group | Source | Timing used |
|---|---|---|
| Past prices: lags 24h / 48h / 168h, 24h and 168h rolling mean and std | ENTSO-E | Day D-1 and earlier (lag ≥ 24h) |
| Load forecast for day D | ENTSO-E (TSO, published ≥ 2h before the gate) | As published for day D |
| Wind and solar forecasts, residual load proxy (load minus wind minus solar) | ENTSO-E (TSO) | Forecasts **for day D-1** (published on D-2) |
| Gas cost of power: TTF front-month close / 0.52 efficiency | Yahoo Finance | Last close dated D-2 or earlier |
| Realized generation by type (nuclear, hydro, gas, …) | ENTSO-E | Lagged 48h (day D-2) |
| Population-weighted temperature, heating and cooling degree days | Open-Meteo archive | Lagged 48h (day D-2) |
| Hour, weekday, month, weekend, French public holidays | Calendar | Known in advance |

`tests/test_features.py` perturbs, in turn, the prices of day D, the wind/solar forecasts
for day D, the gas closes of D-1, and the realized data of D-1. Each time, it asserts that
none of day D's features change.

### Models

| Model | Description |
|---|---|
| `naive` | Standard EPF benchmark: same hour one week earlier on Mondays and weekends, one day earlier otherwise |
| `lasso` | Standardized linear regression with an L1 penalty, alpha chosen by time-series CV. A LEAR-style model, using one regression across hours rather than 24 hourly models |
| `lightgbm` | Gradient-boosted trees, early-stopped on the most recent 10% of each training window |
| `extra_trees` | Extremely randomized trees, a robust tree benchmark |
| `ensemble` | Hour-by-hour median of `lasso`, `lightgbm` and `extra_trees` |

### Evaluation protocol

- **Expanding-window walk-forward.** Each model is trained on two years of history,
  predicts the next 30 days, then the window expands and every model is refitted from
  scratch. Folds start at local midnight, so no model ever sees a price from the auction
  it forecasts.
- **Same hours for every model.** Metrics are computed only on hours where all models have
  a forecast.
- **Metrics.** MAE, RMSE, median absolute error, and rMAE (MAE relative to the naive model;
  below 1 beats naive).
- **Diebold–Mariano tests.** The 24 forecasts of a day are issued together, so their errors
  are correlated. The absolute-loss differential is averaged per day before testing.
- **90% prediction intervals: rolling split-conformal.** For each delivery hour, the
  half-width is the 90% quantile of that hour's absolute errors over the previous 60 days.
  It uses only errors known at decision time. Intervals are scored on coverage, width, and
  interval (Winkler) score (`reports/intervals.csv`).

## Results

Out-of-sample MAE by year (€/MWh):

| Year | naive | lasso | lightgbm | extra_trees | ensemble | Market context |
|---|---:|---:|---:|---:|---:|---|
| 2021 | 20.19 | **17.86** | 49.10 | 22.23 | 21.99 | Gas prices take off in H2 |
| 2022 | 53.00 | **45.41** | 77.77 | 50.29 | 49.27 | Energy crisis, mean price €276/MWh |
| 2023 | 23.45 | 19.43 | 18.93 | 19.33 | **18.46** | Nuclear fleet recovers |
| 2024 | 21.74 | 18.08 | 18.77 | 18.05 | **17.54** | 352 negative-price hours |
| 2025 | 24.39 | 19.09 | 19.29 | 19.10 | **18.56** | 509 negative-price hours |
| 2026 | 34.39 | 28.13 | 28.93 | 29.72 | **28.21** | Higher, more volatile prices (to mid-November) |

**What this shows**

1. **Fundamentals plus recent prices beat persistence.** The Lasso cuts the naive error by
   12–22% every year. All pairwise Diebold–Mariano tests reject equal accuracy
   (`reports/dm_tests.csv`).
2. **Regime shifts hurt tree models.** When 2021–2022 prices rose far above anything in the
   training window, LightGBM, which can only predict values within the range of its
   training targets, was much worse than the naive forecast. Once the crisis is part of
   the training history, trees catch up, and the median ensemble becomes the best model
   every year from 2023.
3. **What drives the forecast.** The previous day's price dominates, followed by the
   two-day lag, the residual load proxy, the weekly lag and the gas cost (LightGBM split
   gain, notebook 02).
4. **Intervals are slightly too narrow.** The 90% conformal intervals cover 85–87% of
   hours overall. Coverage is close to target in calm years (88–90% in 2023–2025) and
   drops when the regime shifts (81% in 2021), because a 60-day window adapts with a lag.

![Last two weeks out-of-sample](reports/figures/last_two_weeks.png)

### What changed the numbers

| Step | Lasso MAE | Comment |
|---|---:|---|
| TSO wind/solar forecasts for day D used as inputs | 22.90 | Not a valid result: those forecasts are published after the gate |
| Same forecasts shifted to day D-1 (published on D-2) | 24.83 | The honest baseline, 8% worse |
| + gas cost (TTF close of D-2) | **24.58** | Final feature set |
| + Extra Trees and median ensemble | 25.59 (ensemble) | Best model every year since 2023 |

Also tested and **not kept**, because they did not improve overall accuracy:

- trees learning a scale-free target (price minus last week's mean, or price divided by
  gas cost). These helped in 2021–2022 but hurt afterwards;
- exponential recency weighting of training samples (one-year half-life). No measurable
  effect.

The two notebooks go further:

- [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb): price regimes,
  negative prices, the solar dip, generation mix, temperature sensitivity.
- [`02_forecast_evaluation.ipynb`](notebooks/02_forecast_evaluation.ipynb): accuracy by year
  and hour, Diebold–Mariano tests, interval coverage by year, sample weeks, LightGBM
  feature importance.

## Quickstart

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Baptiste251001/fr-day-ahead-price-forecasting.git
cd fr-day-ahead-price-forecasting
uv sync --extra dev               # add --extra notebooks for Jupyter
cp .env.example .env              # then paste your ENTSO-E API token
```

**ENTSO-E token.** Create a free account on the
[Transparency Platform](https://transparency.entsoe.eu/), then email
`transparency@entsoe.eu` with the subject "Restful API access" from the account's email
address. Once access is granted (usually within a few days), generate the token under
*My Account Settings*.

```bash
uv run fr-power fetch      # download 2019 -> today to data/raw/ (~15 min, cached)
uv run fr-power features   # build data/processed/feature_matrix.parquet
uv run fr-power train      # walk-forward evaluation (~10 min) -> reports/
```

All settings (dates, lags, walk-forward schedule, hyperparameters, ensemble members) live in
[`configs/default.yaml`](configs/default.yaml). To run a variant, copy the file and pass it
with `--config`.

## Project structure

```
configs/default.yaml        all run settings
src/fr_power/
├── data.py                 ENTSO-E, Open-Meteo and gas-price download, parquet cache
├── features.py             point-in-time feature matrix
├── models.py               naive / lasso / LightGBM / Extra Trees, ensemble, walk-forward
├── evaluation.py           metrics, Diebold-Mariano tests, conformal intervals
└── cli.py                  `fr-power fetch | features | train`
tests/                      offline tests (all network calls mocked)
notebooks/                  data exploration and forecast evaluation
reports/                    metric tables and figures produced by `train`
```

Development checks (ruff and mypy also run as pre-commit hooks):

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## Limitations and next steps

- **No weather forecasts for day D.** The model sees observed temperature from D-2 and the
  TSO's wind and solar forecasts for D-1. Open-Meteo's archive of forecasts issued before
  the gate starts only in early 2024 for wind and solar, too short to train on yet. As
  that archive grows, adding them is the most promising extension. In the live
  EnergyBench benchmark (Genov et al., 2026), giving models weather forecasts and other
  covariates cut price errors by 15–34%.
- **No carbon price.** A free daily EU carbon (EUA) series exists only from late 2021, so
  the fuel-cost feature uses gas alone.
- **Hourly resolution.** Since October 2025 the auction clears 15-minute products. This
  project forecasts hourly prices; moving to the 15-minute grid would match the market.
- **Simplified LEAR.** A full LEAR uses 24 hourly models and a variance-stabilizing
  transform (e.g. asinh) of prices.
- **Untuned hyperparameters.** No search was run inside the walk-forward loop.

## License

[MIT](LICENSE)
