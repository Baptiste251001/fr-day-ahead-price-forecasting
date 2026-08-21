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

Each pipeline stage is its own CLI command (`fetch` and `features` are
implemented; the rest are stubs -- see the build plan below for when each
lands):

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
- [ ] **Phase 3 — Forecasting.** Naive, LEAR, LightGBM, LSTM (point +
      quantile), walk-forward evaluation, Diebold-Mariano tests.
- [ ] **Phase 4 — OU spread model and optimal-stopping execution.**
- [ ] **Phase 5 — Backtest engine.** Walk-forward, costs, sizing, bootstrap
      CIs, cost-sensitivity sweep.
- [ ] **Phase 6 — Analysis and conclusion on the French market.**
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
