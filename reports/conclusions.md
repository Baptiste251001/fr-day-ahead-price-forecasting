# Conclusions: FR intraday-vs-day-ahead spread predictability and tradability

## Status: what is real here, and what is not

This document has two very different kinds of content, and keeping them
visibly separate is the whole point of writing it this way rather than
producing a single polished-looking report.

1. **A qualitative discussion of the real French power market** (below,
   "French vs. Danish market structure") — drawn from public domain
   knowledge of the French generation mix, RTE/ENTSO-E market structure,
   and the 2021-2022 energy crisis. This part is a genuine analytical
   contribution and does not depend on this build having pulled live data.
2. **Illustrative numbers from a synthetic demonstration dataset**
   (`fr_power.demo.synthetic`, see [`notebooks/`](../notebooks/)) — used
   throughout this document to show the *methodology* (how the forecast
   evaluation, OU calibration, regime-split analysis, and backtest work,
   and what their output looks like) end-to-end, without a live ENTSO-E
   API token. **These numbers are not a claim about the real French
   market.** Every such figure below is explicitly labeled "(synthetic
   demo)".

**Update: a real ENTSO-E API token was obtained and `fr-power fetch` /
`features` / `train` have now been run against real FR data** (2019-01-01
through 2026-08-22, ~66,950 hourly rows). The forecasting results in the
new section below ("Real FR day-ahead price forecasting results") are
genuine, not synthetic. **`fr-power calibrate` and `fr-power backtest`
still have not run on real data**, because no granular intraday feed is
configured (see Limitations) -- the target degraded to the day-ahead
price *level*, and OU calibration / the backtest both correctly refuse to
run against a price level rather than a genuine spread. The OU/backtest
sections below therefore remain synthetic-demo-only. Getting real numbers
for those specifically requires a paid EPEX ID1/ID3 feed, not just an
ENTSO-E token.

## Real FR day-ahead price forecasting results (real data, 2019-2026)

`fr-power train`, run walk-forward (69 folds, expanding window, retrained
every 30 days) on 66,953 hours of real FR day-ahead prices and ENTSO-E
day-ahead forecasts, gives an unambiguous, honest result:

| model | RMSE (EUR/MWh) | MAE (EUR/MWh) | MdAE (EUR/MWh) |
|---|---:|---:|---:|
| naive (persistence, 168h) | 61.73 | 37.69 | 23.14 |
| **LEAR** | **33.41** | **18.91** | **12.52** |
| LightGBM | 56.95 | 29.14 | 14.11 |
| LSTM | 56.73 | 32.43 | 19.06 |

**LEAR wins by a wide margin.** It cuts RMSE by 46% versus naive
persistence; LightGBM and LSTM barely beat naive at all (8-8.4% RMSE
reduction) with the default, untuned hyperparameters in
`configs/default.yaml`. Diebold-Mariano tests confirm this is not noise:

| challenger | baseline | DM stat | p-value |
|---|---|---:|---:|
| LEAR | naive | -14.27 | <0.001 |
| LightGBM | naive | -2.87 | 0.004 |
| **LightGBM** | **LEAR** | **+38.26** | **<0.001** |
| LSTM | naive | -2.40 | 0.017 |
| **LSTM** | **LEAR** | **+31.14** | **<0.001** |

A positive DM statistic means the challenger has *higher* loss than the
baseline: LightGBM and LSTM are both significantly *worse* than LEAR
here, not just non-significantly different. This is exactly the honesty
this project was built around -- the brief's own framing was "LEAR is
notoriously hard to beat... beating it is the bar," and on real French
data, at these default hyperparameters, neither of the fancier models
clears that bar. Two candidate explanations, both plausible and not
mutually exclusive: (1) the LightGBM/LSTM hyperparameters in
`configs/default.yaml` were never tuned against real data (they were
chosen to be reasonable defaults, then validated for correctness on
synthetic data, which has a much cleaner signal); (2) day-ahead FR price
dynamics over 2019-2026 -- spanning the 2021-2022 energy crisis, a
structural regime shift -- may genuinely favor a regularized linear model
that adapts smoothly over LightGBM/LSTM's higher-variance fits, especially
walk-forward with only 30-day retraining windows relative to the
crisis-era volatility. Distinguishing these (e.g. via a hyperparameter
sweep, or a regime-split re-run once real backtest data is available)
is the natural next step, not a conclusion this document reaches.

A sklearn `ConvergenceWarning` appeared for LEAR's `LassoCV` on a couple
of the very largest (latest) folds -- worth a `max_iter` increase if this
pipeline is extended, though it did not prevent those folds from
completing.

## Does the strategy survive transaction costs? (synthetic demo)

In the synthetic demonstration (`notebooks/03_ou_and_backtest.ipynb`,
~1.5 years of hourly data with an illustrative regime shift), the combined
OU-timing + LightGBM-forecast strategy is net-PnL-positive after costs and
clearly ahead of the naive persistence strategy:

| | gross PnL (EUR) | net PnL (EUR) | PnL/MWh (EUR) | hit rate | Sharpe (ann.) | max drawdown (EUR) |
|---|---:|---:|---:|---:|---:|---:|
| naive | 11,131 | 9,853 | 5.01 | 68.6% | 17.8 | 786 |
| **lightgbm** | 20,392 | 19,117 | 9.75 | 85.8% | 32.3 | 58 |
| perfect foresight | 74,092 | 69,064 | 8.61 | 100% | 78.6 | 0 |

The cost-sensitivity sweep shows net PnL degrading smoothly and
predictably as the transaction-cost multiplier rises from 0x to 3x
(20,406 &rarr; 17,827 EUR net) rather than collapsing sharply at some
threshold, and the 95% block-bootstrap confidence interval for the
LightGBM strategy's net PnL, [13,522, 24,069] EUR, stays entirely
positive.

**Why this number is not a real result.** The synthetic generator builds
the spread as a genuinely Ornstein-Uhlenbeck process with a clean,
known signal-to-noise ratio, plus a designed (if modest) dependence on an
observable residual-load-anomaly term. Real electricity markets are far
noisier, the true data-generating process is unknown, and a real
forecast's edge over persistence is typically much smaller than the ~46%
RMSE reduction LEAR/LightGBM achieve here. **The honest, load-bearing
result of this exercise is that the analysis pipeline correctly detects a
real, designed edge when one exists and correctly prices it net of costs
and a no-trade band** — not that a ~10 EUR/MWh net edge should be expected
on the real French market. Whether the *real* strategy survives costs is
an open question this repository is built to answer once real data is
available, not one this document can answer today.

An important secondary finding from the demo, worth taking seriously
precisely because it was *not* the result initially obtained: an earlier
version of the backtest engine set trade direction from the OU z-score's
sign (a naive "reversion toward `mu`" rule) and lost money by
construction, because the spread's long-run mean `mu` is not zero (here,
1.1-3.3 EUR/MWh depending on regime) — betting on reversion toward a
non-zero `mu` does not imply the right sign of an absolute-level PnL
payoff. The fix (direction from the forecast's predicted level; OU
z-score as a pure timing/conviction gate) is documented in
`src/fr_power/backtest/engine.py`'s module docstring. This is exactly the
kind of naive-implementation trap the OU pairs-trading literature warns
about, and it is a stronger reason to trust the final design than if it
had "worked" on the first attempt.

## Which features matter most? (synthetic demo)

In the synthetic demo, the top LightGBM feature importances (by split
count) are dominated by the spread's own autoregressive structure:

1. `spread_rollmean_24h`
2. `spread_rollstd_24h`
3. `spread_lag_24h`
4. `weather_temperature_c_lag24h`
5. `spread_rollmean_168h`
6. `day_ahead_price_rollstd_168h`
7. `spread_rollstd_168h`
8. `spread_lag_168h`

This is a useful sanity check on the *methodology*, not a market finding:
the generator's spread is built mostly from OU autoregressive dynamics
plus a comparatively weak (coefficient 0.4, standardized) dependence on a
residual-load anomaly, and the feature-importance ranking correctly
surfaces that structure -- rolling/lagged spread features dominate,
weather enters next (the generator ties load, and hence indirectly the
residual-load-anomaly term, to HDD), and residual-load/nuclear-generation
features individually rank lower than their aggregate influence would
suggest, because their effect is folded into the spread's own history
rather than appearing as a separable feature. That the method recovers a
ranking consistent with how the data was actually generated is reassuring
before trusting it on real data, where the ground truth is unknown.

**On real FR data**, that hypothesis holds up. LightGBM's top-12 feature
importances (by split count), fit on the full real 2019-2026 sample:

1. `day_ahead_price_rollstd_24h`
2. `residual_load_forecast_mw`
3. `day_ahead_price_rollmean_24h`
4. `hour`
5. `wind_forecast_mw`
6. `day_ahead_price_rollstd_168h`
7. `day_of_week`
8. `day_ahead_price_lag_168h`
9. `generation_Nuclear_lag24h`
10. `generation_Wind Onshore_lag24h`
11. `day_ahead_price_rollmean_168h`
12. `day_ahead_price_lag_24h`

`residual_load_forecast_mw` lands at #2 -- the single most consistently
important EPF feature in the literature, confirmed here -- and
`generation_Nuclear_lag24h` and `wind_forecast_mw` both make the top 12,
exactly the nuclear-availability and renewable-forecast channels
hypothesized above. The price's own recent volatility
(`rollstd_24h`, #1) and level (`rollmean_24h`, #3) dominate overall,
which is unsurprising for a level-forecasting task (recall: the target
here is the day-ahead price *level*, not the spread, since no intraday
feed is configured) and consistent with price dynamics being strongly
autocorrelated/regime-dependent over a period that includes the 2021-2022
crisis. Calendar features (`hour`, `day_of_week`) also rank highly,
reflecting FR's pronounced daily/weekly demand and solar-generation
cycles. Temperature/HDD features do not make the top 12 here -- plausibly
because their signal is already substantially captured by
`residual_load_forecast_mw` (ENTSO-E's own load forecast already
internalizes weather effects) and by the 24h-lagged-actual-weather
simplification (see Limitations) being a noisier proxy than a same-day
forecast would be.

## French vs. Danish market structure, and what it implies for spread predictability

This is the genuine, non-synthetic analytical content of this document.

The source paper (Kılıç, Nielsen & Thibbotuwawa, *Energies* 2024) studies
a market whose price formation is dominated by **wind and solar variability**:
Danish/Nordic prices move primarily on forecast errors in weather-driven
renewable output, and the intraday-vs-day-ahead spread is largely a
reflection of how much the wind/solar forecast has been revised between
the D-1 auction and real time. France is structurally a different market
along several axes that should change what drives the FR spread and how
predictable it is:

- **Nuclear baseload dominance.** France generates roughly two-thirds of
  its electricity from nuclear (versus a wind/solar-led mix in Denmark).
  Nuclear output is *plannable but not fully predictable*: scheduled
  refuelling/maintenance outages are known well in advance, but unplanned
  outages, life-extension inspections, and (as in 2021-2022, see below)
  systemic issues are not. This makes **nuclear availability** a
  candidate first-order driver of the FR day-ahead price level and,
  plausibly, of the ID-vs-DA spread whenever real-time nuclear
  availability is revised intraday -- a mechanism with no real Danish
  analogue, where the corresponding revision channel is wind/solar
  forecast error instead.
- **Hydro flexibility.** French hydro (reservoir and pumped storage, more
  dispatchable than Danish wind/solar) provides a buffering, price-smoothing
  role that Denmark's flatter, more interconnection-dependent system lacks
  to the same degree. This should make the FR spread *less* mechanically
  tied to renewable forecast errors than the Danish spread, all else
  equal, and more tied to residual demand and nuclear-availability
  surprises.
- **Winter thermosensitivity.** French demand is unusually sensitive to
  temperature because of widespread electric resistive heating -- a cold
  snap can shift national demand by several GW. This argues for
  temperature/HDD (and its *forecast*, not just realized values -- see
  Limitations below) being a more important day-ahead-forecast-error
  driver in France than in Denmark, where district heating reduces the
  electricity-demand-temperature link.
- **Structural export position and interconnection.** France is typically
  a net electricity exporter; cross-border flows and interconnector
  capacity to a *larger* number of neighbors (GB, Belgium, Germany,
  Italy, Spain, Switzerland) than Denmark's smaller set of connections
  give the FR price more channels through which external shocks can
  transmit -- and more channels through which the day-ahead net position
  can be revised intraday.
- **Rising negative-price hours from solar.** As French solar capacity has
  grown, negative day-ahead prices during high-solar, low-demand midday
  hours (particularly spring) have become more frequent -- a phenomenon
  long-familiar in the Danish/German markets and now a growing feature of
  the French one too. Wherever this occurs, it is plausibly one of the
  more genuinely comparable dynamics between the two markets, since it is
  driven by the same solar-oversupply mechanism in both.

**Implication for spread predictability.** The Danish paper's forecasting
approach leans on wind/solar forecast-revision features because that is
where the ID-vs-DA spread's information content concentrates in that
market. In France, the analogous, most information-dense features are
plausibly nuclear-availability revisions and residual-load surprises
(itself thermosensitive), with wind/solar forecast error playing a
smaller relative role given nuclear's larger share of the mix -- a
hypothesis directly testable via the same feature-importance methodology
demonstrated on synthetic data above, once real data is available. It also
implies the FR spread may be *less* frequently and *less* severely
revised intraday than the Danish spread during "normal" periods (hydro
buffering, more baseload predictability) but capable of much larger
revisions during nuclear-fleet stress events -- which is exactly what the
regime-split analysis below is built to detect.

## Regime split: the 2021-2022 energy crisis and nuclear outage period

France experienced an unusually severe combination in 2021-2022: a
significant fraction of the nuclear fleet was taken offline for stress-corrosion
inspections and repairs (discovered from late 2021), compounding the
broader European gas-price-driven energy crisis. Both the price *level*
and its *volatility* were structurally different during this period than
in a normal year, and it is a natural, real (not synthetic-only) test of
whether the strategy's edge (if any) survives a genuine, severe regime
shift, or whether it was only ever calibrated to "normal" dynamics.

**Methodology** (already implemented, demonstrated on the generator's
illustrative regime shift in `notebooks/03_ou_and_backtest.ipynb`): split
the historical spread series into a "normal" window and the crisis window
(dates to be finalized against the actual nuclear-outage timeline once
real data is available -- approximately late 2021 through 2022), calibrate
OU parameters (`fr_power.ou.estimate.estimate_ou`) separately on each, and
compare `kappa`, `mu`, `sigma`, and the mean-reversion half-life
side-by-side; then run the full walk-forward backtest
(`fr-power backtest`) restricted to each window via `data.start_date`/
`data.end_date` and compare net PnL, Sharpe, and hit rate.

In the synthetic demo, this split shows the mean-reversion level (`mu`)
shifting sharply (1.15 &rarr; 3.27 EUR/MWh) while the reversion speed
(`kappa`, hence the half-life, ~17 &rarr; ~15 periods) stays comparable --
by construction, since the generator only shifts the regime's mean, not
its dynamics. **Whether the real 2021-2022 shift in the FR spread was
similarly a pure mean shift, or also changed the reversion speed and
volatility (plausible, given how much more actively nuclear-availability
news was being priced intraday during the crisis), is an open, genuinely
interesting question this same code answers once run on real data.**

## Data limitations and modelling assumptions (consolidated)

Collected here from what is otherwise scattered across module docstrings,
so a reader gets the full picture in one place:

- **No granular intraday feed.** `fr_power.data.intraday` is a pluggable
  interface with no free, granular EPEX ID1/ID3 source; without a paid
  feed configured, the pipeline degrades to forecasting the day-ahead
  price *level* rather than the intraday-vs-day-ahead spread, and OU
  calibration / the backtest are skipped entirely (both explicitly refuse
  to run against a price level, since it is not economically
  mean-reverting the way this project's models assume). This is exactly
  what happened on the real 2019-2026 FR run: Phases 1-3 (fetch, features,
  forecasting) ran on genuine real data and produced the results above;
  Phases 4-5 (OU calibration, backtest) remain synthetic-demo-only pending
  a paid intraday feed.
- **ENTSO-E discontinued FR's net cross-border position feed.** A live
  fetch found `query_net_position` returns data for FR only through
  2025-09-30 -- ENTSO-E/RTE appear to have stopped publishing it after
  that date (the reason is not documented on the Transparency Platform;
  possibly related to changing market-coupling arrangements). Keeping it
  as a mandatory feature would have made LEAR and LSTM (both of which
  require complete rows) produce zero predictions for the entire most
  recent ~11 months of data -- exactly the period most relevant to
  report on. `fr_power.features.build.build_feature_matrix` now treats
  `net_position` as optional, and `configs/default.yaml` disables
  fetching it by default; it can still be re-enabled for a deliberately
  pre-2025-10 analysis. This is the kind of real-world data-availability
  gap no amount of synthetic testing would have surfaced.
- **Lagged actual weather stands in for a weather forecast.** No free,
  long-history archive of *historical weather forecasts* exists, so
  `fr_power.features.build` uses lagged *actual* temperature/HDD/CDD (by
  default, the previous day's) as a persistence proxy for expected
  weather on the delivery day. This is a real, deliberate simplification,
  not an oversight -- see the module docstring for the full reasoning --
  and it likely understates how informative weather *would* be with a
  true forecast, since actual persistence-based weather is noisier than a
  same-day meteorological forecast.
- **Realized generation is lagged, not same-day.** Actual generation by
  production type (used for e.g. nuclear-availability features) is not
  known at the D-1 13:00 decision point; it is used lagged by
  `features.generation_availability_lag_hours` (24h by default), a
  documented simplification with the same "persistence proxy" character
  as the weather case.
- **The traded instrument is a one-shot bet, not a held position.** The
  backtest engine (`fr_power.backtest.engine`) treats each hourly
  delivery product as a single decide-once, settle-once trade, because
  the available data is a settled spread per product, not a granular
  intraday price path *within* one product's trading window. The
  Phase 4 OU optimal-stopping state machine
  (`fr_power.ou.optimal_stopping.generate_positions`, with its
  entry/exit/max-holding-period logic) is fully built and tested on
  synthetic OU paths but is *not* exercised by the backtest for this
  reason -- it is the right tool for a genuinely continuous intraday
  price path, which this repository does not have access to.
- **OLS and MLE OU calibration are the same estimator wearing two
  hats.** Under the exact discrete-time Gaussian OU transition, the MLE
  and OLS estimators for `kappa` and `mu` are mathematically identical
  (both are the least-squares AR(1) fit); they differ only in the
  degrees-of-freedom used for the `sigma` estimate. Both are exposed via
  `ou.estimation.method` because the difference is real, if narrow.
- **Position sizing is confidence-weighted by forecast magnitude, gated
  by OU conviction and cost-aware no-trade bands, but not otherwise
  risk-managed.** There is no portfolio-level risk limit, no correlation
  structure across simultaneously-open positions in different hourly
  products, and no explicit stop-loss beyond the no-trade-band/cost gate
  itself. A production system would need all three.
- **The synthetic demo's signal-to-noise ratio is generous by
  design**, as discussed above -- Sharpe ratios and hit rates in this
  document's synthetic tables should not be read as achievable targets
  for a real strategy.
- **Costs are a simple per-MWh fee plus half-spread**, applied once per
  traded hourly product (`costs.fee_eur_per_mwh` +
  `costs.half_spread_eur_per_mwh`); real intraday execution costs (market
  impact, partial fills, bid-ask dynamics that widen under stress) are
  not modelled beyond this.

## Reproducing this analysis on real data

```powershell
copy .env.example .env   # add a real ENTSOE_API_TOKEN
uv run fr-power fetch
uv run fr-power features
uv run fr-power train
uv run fr-power calibrate    # currently refuses: no intraday feed configured
uv run fr-power backtest     # currently refuses: no intraday feed configured
```

**Status: `fetch`, `features` and `train` have been run against real FR
data** (2019-01-01 to 2026-08-22) and produced the results in this
document. `calibrate` and `backtest` still require a paid granular
intraday feed (see Limitations) -- plugging one in at
`data.intraday.path` and re-running the full pipeline is the single
remaining step to replace the OU/backtest sections above with real
numbers too.

Once that's done, re-run `notebooks/01_data_exploration.ipynb` through
`03_ou_and_backtest.ipynb` with the loaders swapped from
`generate_synthetic_market_data` to `fr_power.data.cache.read_cache`
against `data/raw/` and `data/processed/` -- the analysis code itself
(feature building, walk-forward evaluation, OU calibration, backtesting)
does not change at all between the synthetic demo and a real run; only
the data source does.
