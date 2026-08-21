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

No `.env` with a real `ENTSOE_API_TOKEN` was configured during this build,
so `fr-power fetch` has never pulled real FR market data in this
environment. Everything the pipeline needs to run on real data instead is
already built and tested (Phases 1-5); the gap here is purely one of data
access, not missing code. Whoever runs this next need only add a token
(see `.env.example`), run `fr-power fetch features train calibrate
backtest`, and the real numbers replace the synthetic ones in this same
structure — nothing about the analysis framework changes.

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

**On real FR data**, based on established EPF literature and the French
generation mix (see below), the expected top drivers are residual load
(load minus wind minus solar -- the single most consistently important
feature across EPF studies), nuclear availability/outage indicators, and
temperature/HDD in winter months. Confirming or overturning that
expectation is exactly what `fr-power train` run on real data, followed
by the same `LGBMRegressor.feature_importances_` inspection shown in
`notebooks/02_forecast_eval.ipynb`, will answer.

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
  mean-reverting the way this project's models assume). All results
  described in this document as "synthetic demo" assume synthetic
  intraday data was supplied; without any intraday data at all (this
  build's actual current state), only Phases 1-3's day-ahead forecasting
  machinery is exercised on real data.
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
uv run fr-power calibrate
uv run fr-power backtest
```

Then re-run `notebooks/01_data_exploration.ipynb` through
`03_ou_and_backtest.ipynb` with the loaders swapped from
`generate_synthetic_market_data` to `fr_power.data.cache.read_cache`
against `data/raw/` and `data/processed/` -- the analysis code itself
(feature building, walk-forward evaluation, OU calibration, backtesting)
does not change at all between the synthetic demo and a real run; only
the data source does.
