# FR Power: Intraday vs Day-Ahead Spread Forecasting and OU Trading

**Status: work in progress, built phase by phase. This README grows with each phase.**

## Pitch

At around 13:00 CET on day D-1, EPEX SPOT publishes day-ahead (DA) auction
prices for every hour of day D. Each of those hourly products then keeps
trading continuously on the intraday (ID) market throughout day D, right up
to (near) real time. The gap between where a given hour eventually settles
intraday and where it cleared in the day-ahead auction is driven by forecast
errors in wind, solar, and load that get revised as day D approaches.

This project:

1. **Forecasts the intraday-vs-day-ahead spread** for the French (FR)
   bidding zone, benchmarked honestly against a naive persistence forecast
   and a LEAR (Lasso-Estimated AutoRegressive) model — both notoriously hard
   to beat in electricity price forecasting (EPF) — before trying LightGBM
   and an LSTM (point and quantile forecasts).
2. **Models the intraday price path with an Ornstein-Uhlenbeck (OU)
   process** and derives an optimal-stopping execution rule (entry/exit
   thresholds with a no-trade band) for timing the close of a position
   during the trading window.
3. **Backtests the combined strategy honestly**: walk-forward
   out-of-sample only, net of transaction costs, with position sizing,
   bootstrap confidence intervals, and a cost-sensitivity sweep.

It adapts the idea in Kılıç, Nielsen & Thibbotuwawa (*Energies*, 2024),
"Intraday Electricity Price Forecasting via LSTM and Trading Strategy"
(originally applied to the Danish/Nordic market), to the French market —
whose price dynamics are shaped by a very different generation mix
(nuclear baseload, hydro flexibility, strong winter thermosensitivity,
structural export position, and a rising share of solar-driven negative
price hours) — and extends the execution side with an explicit OU /
optimal-stopping model instead of a fixed-horizon rule.

The goal of this repository is to demonstrate rigorous, honest quantitative
research practice, not to advertise an inflated backtested PnL number. If a
result does not survive transaction costs or does not beat the naive
benchmark, that will be reported plainly — see
[Limitations and realism](#limitations-and-realism) (added in later phases)
and [`reports/conclusions.md`](reports/conclusions.md).

## Guiding principles

- **Point-in-time discipline.** Every feature and forecast uses only
  information available at the decision moment (e.g. day-ahead forecasts of
  wind/solar/load, never realized values, when realized values would not
  yet be known). The exact information timeline is documented in the
  feature-engineering module once implemented (Phase 2).
- **Strong benchmarks.** Naive persistence and LEAR are the bar. Beating
  them, not just running a fancier model, is the point.
- **Realistic backtest.** Walk-forward out-of-sample, transaction costs, a
  no-trade band, position sizing. No single-split backtests.
- **Honest metrics.** Per-MWh PnL, annualized Sharpe, hit rate, max
  drawdown, turnover, net-of-costs PnL, and a perfect-foresight upper bound.
  Never a total-EUR headline built by multiplying a per-MWh edge across
  national production.
- **Reproducibility.** Config-driven, seeded, cached data, one CLI command
  per pipeline stage.

## Repository layout

```
configs/            YAML configs (dates, bidding zone, costs, model params)
data/                gitignored: raw/ interim/ processed/ parquet caches
src/fr_power/
  config.py          pydantic config loader
  data/              ENTSO-E, RTE, intraday, weather ingestion (Phase 1)
  features/          point-in-time feature matrix (Phase 2)
  models/            benchmarks, LEAR, LightGBM, LSTM, metrics (Phase 3)
  ou/                OU calibration + optimal-stopping execution (Phase 4)
  backtest/          walk-forward engine + metrics (Phase 5)
  cli.py             typer CLI: fetch / features / train / calibrate / backtest / report
notebooks/           exploration, forecast evaluation, OU + backtest notebooks
reports/             figures/ and the written French-market conclusions
tests/
reference/           local copy of the source paper (gitignored, not redistributed)
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency
management (falls back to `poetry` if you prefer; a `pyproject.toml` works
with either).

```powershell
# Install uv if you don't have it
pip install uv

# Create the environment and install dependencies (including dev tools)
uv sync --extra dev

# Copy the environment template and fill in your ENTSO-E API token
copy .env.example .env
```

You need a free ENTSO-E Transparency Platform account and API token — see
the comments in [`.env.example`](.env.example) for how to request one.
Intraday (EPEX ID3/ID1) granular data is usually a paid feed; if you don't
have one, drop a CSV/parquet file at the path configured under
`data.intraday.path` in [`configs/default.yaml`](configs/default.yaml), or
leave it unset — the pipeline will log a loud warning and gracefully
degrade to a day-ahead-price forecasting task (see Phase 1).

## How to run

Each pipeline stage is its own CLI command (`fetch`, `features`, `train`,
`calibrate` and `backtest` are implemented; `report` is a stub -- see the
build plan below for when it lands):

```powershell
uv run fr-power fetch      --config configs/default.yaml   # add --force-refresh to bypass the parquet cache
uv run fr-power features   --config configs/default.yaml
uv run fr-power train      --config configs/default.yaml
uv run fr-power calibrate  --config configs/default.yaml
uv run fr-power backtest   --config configs/default.yaml
uv run fr-power report     --config configs/default.yaml
```

### Data sources (Phase 1)

`fr-power fetch` pulls and caches (as parquet under `data/raw/`):

- **ENTSO-E Transparency Platform** (via [`entsoe-py`](https://github.com/EnergieID/entsoe-py)),
  for FR: day-ahead auction prices, day-ahead wind & solar generation
  forecast, day-ahead load forecast, actual generation by production type
  (to see nuclear/hydro/thermal shares), and the day-ahead net
  cross-border position. Requires `ENTSOE_API_TOKEN`.
- **Intraday prices** via a *pluggable* source
  (`src/fr_power/data/intraday.py`): drop a CSV/parquet file with an
  `intraday_price_eur_mwh` column at the path set in
  `data.intraday.path` and it will be picked up automatically. Granular
  EPEX ID1/ID3 continuous-trading indices are usually a paid feed and are
  not bundled here. **With no file present, the pipeline logs a loud
  warning and degrades gracefully to a day-ahead-price forecasting task**
  rather than failing -- this is a known, explicitly-flagged limitation,
  not a silent gap.
- **Weather** (optional, on by default) via the free
  [Open-Meteo](https://open-meteo.com/) historical archive API: daily
  temperature for a population-weighted basket of French cities, combined
  into national HDD/CDD (France's demand is strongly thermosensitive due
  to electric heating).
- **RTE eCO2mix** (optional, off by default): a cross-check/enrichment
  source from RTE's own Data Portal. Requires `RTE_CLIENT_ID` /
  `RTE_CLIENT_SECRET` and `data.rte_eco2mix.enabled: true`.

### Point-in-time feature matrix (Phase 2)

`fr-power features` reads the cached raw data and assembles an hourly
feature matrix, saved to `data/processed/feature_matrix.parquet` (plus a
`.meta.json` sidecar recording the target column name, since parquet
doesn't preserve pandas' `DataFrame.attrs`).

The full information-timeline rationale lives in the module docstring of
[`src/fr_power/features/build.py`](src/fr_power/features/build.py); in
short, everything is built around what is actually knowable at the ~13:00
CET/CEST D-1 decision point:

- The day-ahead price, ENTSO-E's day-ahead wind/solar and load forecasts,
  and the day-ahead net cross-border position are all D-1-published
  quantities *for* day D, so they're used unlagged.
- Realized generation by type and actual temperature are **not** known for
  day D at that point, so both are lagged (`generation_availability_lag_hours`,
  `weather_availability_lag_hours`, default 24h each) -- the most recent
  values knowable as of D-1 13:00, used as persistence proxies. Using
  lagged *actual* weather instead of a true weather *forecast* is a
  deliberate, documented simplification (no free long-history archive of
  historical weather forecasts exists) -- see `reports/conclusions.md`.
- When intraday data is unavailable (the default, pending a paid feed),
  the target degrades to the day-ahead price level itself, and the
  current-hour day-ahead price is excluded from the *feature* set so the
  model is never handed its own target.

`tests/features/test_build.py` uses sentinel data (values equal to a
traceable integer position) to directly verify each lag/rolling feature
only ever pulls from strictly earlier delivery hours.

### Forecasting suite (Phase 3)

`fr-power train` reads the cached feature matrix and walk-forward
evaluates four models, in increasing order of sophistication -- each is
meant to beat the previous one, not just add complexity for its own sake:

1. **Naive persistence** (`models/benchmarks.py`): predicts the value
   observed exactly `models.naive.lag_hours` (default 168h, i.e. same
   hour last week) earlier, reading a pre-built lag feature column.
2. **LEAR** (`models/benchmarks.py`): a standardized Lasso regression,
   alpha selected by time-ordered cross-validation (`TimeSeriesSplit`,
   never a random/shuffled split). The classic hard-to-beat EPF benchmark.
3. **LightGBM** (`models/gbm.py`): gradient-boosted trees, with early
   stopping on a trailing chronological validation slice (never random).
4. **LSTM** (`models/lstm.py`): a sequence model trained with **pinball
   (quantile) loss** across `models.lstm.quantiles`, so it produces a full
   predictive distribution, not just a point forecast; the median quantile
   doubles as the point prediction. Early stopping again uses a trailing
   chronological slice.

All four share one interface (`fit(X, y)` / `predict(X)`, see
`models/base.py`) and are run through a single generic **expanding-window
walk-forward harness** (`models/walk_forward.py`, reusing the
`backtest.walk_forward` config block): starting from
`initial_train_days`, each model is retrained every `retrain_every_days`
and evaluated on the next `test_window_days` -- never a single train/test
split.

Evaluation reports RMSE/MAE/MdAE per model (`models/metrics.py`) plus
pairwise **Diebold-Mariano tests** (LEAR vs naive, LightGBM vs naive/LEAR,
LSTM vs naive/LEAR) so "is the fancier model actually better, or just
different" gets a statistical answer rather than a vibe. Results are
saved to `reports/forecast_metrics.csv`, `reports/dm_tests.csv`, and
`reports/figures/forecast_rmse.png`. If LSTM does not beat LEAR, that
will show up directly in this table -- see
[Limitations and realism](#limitations-and-realism) once real data is run
through the pipeline.

### OU spread model and optimal stopping (Phase 4, the differentiator)

`fr-power calibrate` fits an Ornstein-Uhlenbeck process,
`dX = kappa(mu - X)dt + sigma dW`, to the intraday-vs-day-ahead spread
series and derives an execution rule for *when* to close a position --
this is the extension beyond the source paper, which left an explicit
threshold rule as future work.

**Calibration** (`ou/estimate.py`) fits the exact discrete-time AR(1)
transition of the OU process (`ou.estimation.method: "ols"` or `"mle"`).
A close look at the math shows these two methods necessarily produce
*identical* kappa/mu point estimates -- under the exact Gaussian
transition density, MLE for the mean parameters is exactly the
least-squares fit -- and differ only in the degrees-of-freedom used for
the volatility (sigma) estimate. Reporting both anyway, rather than
picking one and hiding the redundancy, is itself the kind of honesty this
project is trying to model. Reported diagnostics: kappa (mean-reversion
speed), mu, sigma, the mean-reversion half-life, and R^2.

**Execution** (`ou/optimal_stopping.py`) is a threshold-band rule on
z = (spread - mu) / stationary_std:

- `entry_z_score`: open a position (short the spread if z > 0, long if
  z < 0) once |z| crosses this far from fair value.
- `exit_z_score`: close it once |z| reverts inside this band, or once
  `max_holding_hours` elapses (pure OU mean reversion gives no
  guaranteed reversion time).
- `no_trade_z_score`: a **re-arming** threshold -- after a position
  closes, no new one opens, even if |z| immediately re-crosses
  `entry_z_score`, until the spread has first revisited this tighter band
  near `mu`. Without it, a spread oscillating between the exit and entry
  bands would generate a costly round-trip on every oscillation; this is
  what makes it a genuine *no-trade zone* rather than one threshold
  wearing two hats. (An earlier draft nested `no_trade_z_score` inside
  `exit_z_score` directly, which made it mathematically redundant --
  worth naming here since catching that was part of getting this layer
  right.)

`tests/ou/test_estimate.py` validates calibration against synthetic OU
paths with known parameters; `tests/ou/test_optimal_stopping.py` traces
the state machine (entry, exit, forced-exit on max holding, and the
re-arming rule) against hand-computed expected sequences, including the
whipsaw scenario the re-arming rule exists to prevent.

Combining this timing signal with the Phase 3 forecast's directional
edge/magnitude and with cost-aware position sizing is the backtest
engine's job (Phase 5).

### Backtest engine (Phase 5)

`fr-power backtest` runs the walk-forward, cost-aware backtest that
combines the OU timing signal (Phase 4) with each Phase 3 model's
forecast, and reports honest, non-cherry-picked metrics.

**Trade structure.** Given this repository's default data -- a settled
spread series per hourly product, not a granular intraday tick path
within one product's trading window (see `fr_power.data.intraday`) --
each hourly product is a **one-shot** trade: a position is sized once, at
the D-1 cutoff, and settles automatically against the realized spread.
This is a deliberate scope decision, not an oversight: it's why
`backtest/engine.py` reuses the OU z-score and `entry_z_score` from
Phase 4 for timing/conviction, but not the full `generate_positions`
state machine (whose entry/exit/holding-period logic is for a
continuously-tradeable position *within* one product's trading window --
exactly what it will be for once a granular feed is plugged in).

**Signal combination**, entirely walk-forward and point-in-time safe (OU
recalibrated per fold on training data only; the "current" deviation from
`mu` is read off a lagged, already-realized spread feature -- never the
row's own not-yet-settled target):

- **OU direction**: short the spread if the lagged z-score exceeds
  `+entry_z_score`, long if below `-entry_z_score`, flat otherwise.
- **Forecast confirmation**: the trade is skipped unless the forecast's
  implied edge agrees in sign with the OU direction -- this is "the
  forecast gives the directional edge" from the original brief, made
  concrete as a hard gate rather than a vague blend.
- **Confidence-weighted sizing**: size scales with the forecast edge's
  magnitude relative to the OU stationary std, capped at
  `backtest.max_position_mwh`.
- **Cost-aware no-trade band**: skipped unless the edge clears
  `backtest.no_trade_band_eur_per_mwh` plus the transaction cost -- a
  real but tiny edge that costs more to trade than it's worth isn't a
  trade.

**Metrics** (`backtest/metrics.py`), computed over the *full*
out-of-sample period including untraded (zero-PnL) hours, not just the
active subset: gross and net PnL, PnL per MWh, hit rate, **annualized
Sharpe with correct hourly scaling** (`sqrt(24*365)`, not a daily-return
`sqrt(252)` copy-paste), max drawdown, and turnover. A **block bootstrap**
(resampling contiguous chunks of hourly PnL, not individual hours, to
respect autocorrelation) gives confidence intervals on Sharpe and total
PnL for the configured `backtest.signal_model`. A **cost-sensitivity
sweep** re-runs the full engine at each multiplier in
`backtest.cost_sensitivity_multipliers` -- cost multipliers change *which*
trades clear the no-trade band, not just the realized cost, so this
genuinely re-runs the strategy rather than rescaling a fixed PnL curve
after the fact. Every model is compared against a **perfect-foresight**
upper bound (full size in the *correct* direction, same cost/band
economics) so results are read relative to a ceiling, never in isolation.

Results save to `reports/backtest_metrics.csv`,
`reports/backtest_cost_sweep.csv`, `reports/backtest_bootstrap.json`, and
`reports/figures/backtest_cumulative_pnl.png`.

`tests/backtest/test_engine.py` tests the sizing/gating formulas exactly
against hand-picked OU parameters (no estimation noise in the mix) and
directly verifies point-in-time safety: perturbing a fold's *realized*
spread changes its settled PnL but never its (already-decided) direction,
confidence, or size.

### Analysis and conclusions (Phase 6)

**[`reports/conclusions.md`](reports/conclusions.md) is the core written
deliverable** -- read that first. It answers the brief's questions (does
the strategy survive costs, which features matter most, how do French and
Danish market dynamics differ and what that implies for spread
predictability, a 2021-2022 energy-crisis regime split) and consolidates
every data/modelling limitation documented across the codebase into one
place.

No live ENTSO-E API token was configured while Phase 6 was originally
built, so `src/fr_power/demo/synthetic.py` generates a synthetic,
illustrative FR-like dataset (a genuinely OU-distributed spread with a
regime shift, nuclear maintenance dips, thermosensitive load, seasonal
wind/solar, occasional negative prices) so the full pipeline --
feature building, walk-forward forecasting, OU calibration, the
backtest -- runs end-to-end and produces real (if synthetic-data) output.
The three notebooks under [`notebooks/`](notebooks/) exercise this
end-to-end and are executed, not just written (their output cells are
real): `01_data_exploration.ipynb`, `02_forecast_eval.ipynb`,
`03_ou_and_backtest.ipynb`. Every synthetic figure is clearly labeled as
such, both in the notebooks and in `conclusions.md` -- the qualitative
French-vs-Danish market discussion there is the genuine analytical
content and does not depend on this synthetic data at all.

**A real ENTSO-E token was since obtained**, and `fetch`/`features`/`train`
have been run against real FR data (2019-2026, ~66,950 hourly rows) --
see the "Real FR day-ahead price forecasting results" section of
[`reports/conclusions.md`](reports/conclusions.md) for genuine results
(LEAR beats LightGBM and LSTM by a wide, statistically significant
margin). `calibrate`/`backtest` still await a paid granular intraday
feed, without which there is no genuine spread to calibrate the OU model
on. [`02_forecast_eval_real.ipynb`](notebooks/02_forecast_eval_real.ipynb)
mirrors `02_forecast_eval.ipynb`'s analysis on this real data -- it loads
the already-computed `data/processed/forecast_predictions.parquet` rather
than retraining (the full walk-forward took ~2h15), so it stays fast to
re-run.

Running the same notebooks against real data only requires swapping the
`generate_synthetic_market_data` call for `fr_power.data.cache.read_cache`
against `data/raw/`/`data/processed/`; none of the analysis code differs.

## Development

```powershell
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src               # type check
uv run pytest                 # tests
uv run pre-commit install     # install git hooks (run once)
```

CI (GitHub Actions) runs lint, format check, mypy, and pytest on every push
and pull request; see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Build plan / status

- [x] **Phase 0 — Scaffolding.** Repo structure, config system, CLI
      skeleton, CI, pre-commit.
- [x] **Phase 1 — Data ingestion.** ENTSO-E (DA prices, DA wind/solar/load
      forecasts, generation by type, net cross-border position), pluggable
      intraday source, optional weather (HDD/CDD) and RTE eCO2mix.
- [x] **Phase 2 — Feature engineering.** Point-in-time feature matrix,
      documented information timeline, leakage tests.
- [x] **Phase 3 — Forecasting.** Naive, LEAR, LightGBM, LSTM (point +
      quantile), walk-forward evaluation, Diebold-Mariano tests.
- [x] **Phase 4 — OU spread model and optimal-stopping execution.**
- [x] **Phase 5 — Backtest engine.** Walk-forward, costs, sizing, bootstrap
      CIs, cost-sensitivity sweep.
- [x] **Phase 6 — Analysis and conclusion on the French market.**
- [ ] **Phase 7 — Polish.** Final README, methodology diagram, results
      table, limitations section.

## Relation to prior work

This project reproduces, and then extends the execution side of, Kılıç,
Nielsen & Thibbotuwawa (*Energies*, 2024), applying it to the French rather
than Nordic market and replacing the fixed-horizon exit with an OU-based
optimal-stopping rule. A short note connecting this to optimal-stopping /
two-stage stochastic optimization background will be added in Phase 7.

## License

MIT (see [`pyproject.toml`](pyproject.toml)). The reference paper under
`reference/` is not redistributed — it is a local, gitignored copy for
personal use only.
