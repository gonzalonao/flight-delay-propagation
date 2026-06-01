# Power BI Report Spec — Flight Delay Propagation (TFM)

> Build spec for the TFM Power BI deliverable. The data + model are produced by
> code (`scripts/export_powerbi.py`, the Fabric pipeline); the `.pbix` itself is
> assembled in **Power BI Desktop** following this document. Apply the theme
> `powerbi/theme.json` (View → Themes → Browse for themes).

## 1. Architecture (hybrid)

Two data paths feed one report:

| Path | Source | Feeds | Mode |
|---|---|---|---|
| **Local (offline)** | `scripts/export_powerbi.py` → `outputs/powerbi/*.csv` (and `.parquet`) | Pages 1–2 (base EDA), Page 4 (accuracy), and Page 3 as an **offline fallback** | **Import** |
| **Cloud (live)** | OneLake Delta `predictions_latest` (Fabric `pl_hourly_predict`) | Page 3 live dashboard | **DirectLake** |

The exporter is the single source of truth for the static reference tables
(`dim_airport`, `agg_baseline_volume`); the same two are uploaded once to OneLake
so the live Page 3 can compute "traffic vs. usual". `fact_predictions` (local)
and `predictions_latest` (cloud) share the **same long schema**, so Page 3 visuals
can be pointed at either.

### How to generate the local data
```
# base tables only (fast, no GPU):
python scripts/export_powerbi.py --no-predictions

# full export incl. predictions over the test split (needs the champion ckpt):
set PYTHONIOENCODING=utf-8
python scripts/export_powerbi.py ^
  --checkpoint outputs/runs/20260514-213242/seq2seq_gnn_large.pt ^
  --config configs/weekend/seq2seq_gnn_large.yaml --split test
```
Import the CSVs in `outputs/powerbi/` via **Get Data → Text/CSV** (or Folder).

## 2. Data dictionary (`outputs/powerbi/`)

- **dim_airport** — `airport_code` (key), `name`, `latitude`, `longitude`, `elevation_m`.
- **dim_date** — `date` (key), `year`, `month`, `month_name`, `day`, `day_of_week`,
  `day_name`, `is_weekend`, `quarter`, `week_of_year`, `season`.
- **dim_hour** — `hour` (key 0–23), `hour_label`, `part_of_day`.
- **fact_airport_hour** — grain (`airport_code`, `ts_hour`). `sched_arr`, `sched_dep`,
  `arr_operated`, `dep_operated`, `arr_cancelled`, `dep_cancelled`, `arr_delay_mean`,
  `arr_delay_median`, `dep_delay_mean`, `arr_del15`, `dep_del15`, plus `date`, `hour`,
  `day_of_week`, `month`, `year`, and (optional) `est_arr_passengers`, `est_dep_passengers`.
- **agg_baseline_volume** — grain (`airport_code`, `day_of_week`, `hour`):
  `baseline_sched_arr`, `baseline_sched_dep`, `baseline_arr_delay_mean`, `n_observations`, `day_name`.
- **agg_airline** — (`Airline`, `month`): `flights`, `arr_delay_mean`, `dep_delay_mean`, `arr_del15_rate`.
- **agg_route** — (`Origin`, `Dest`): `flights`, `arr_delay_mean`, `arr_del15_rate`, `mean_distance`.
- **agg_distance_bucket** — (`distance_bucket`): `flights`, `arr_delay_mean`, `arr_del15_rate`.
- **fact_predictions** — long schema: `airport_code`, `horizon_h`, `predicted_arr_delay_min`,
  `pct_arr_delayed_15`, `target_ts`, `prediction_ts`; ground truth `actual_arr_delay_min`,
  `actual_arr_del15`, `pct_target`; enrichment `sched_arr`, `sched_dep`,
  `baseline_sched_arr`, `expected_delayed_flights`, `traffic_vs_usual`, plus `dow`, `hour`.

> **Passengers are estimated.** The BTS data is flight-level; `est_*_passengers`
> = flights × `seats_per_flight` (config, default 130). Label every passenger
> visual as "estimated" and disclose the assumption in the thesis.
>
> **Delay-cause attribution is unavailable** in the local 2018–19 parquet (BTS
> `CarrierDelay`/`WeatherDelay`/… columns absent). Page 2 is therefore
> *associational* (delay vs. time/airline/route/distance), not causal.

## 3. Data model (star schema)

Relationships (single-direction, one-to-many from dim → fact):
- `dim_airport[airport_code]` → `fact_airport_hour[airport_code]`, `agg_baseline_volume[airport_code]`, `fact_predictions[airport_code]`.
- `dim_date[date]` → `fact_airport_hour[date]` (mark `dim_date` as the date table).
- `dim_hour[hour]` → `fact_airport_hour[hour]` (and to `agg_baseline_volume[hour]`).
- `agg_route` relates to `dim_airport` twice (Origin, Dest) — use one active
  relationship (Origin) + `USERELATIONSHIP` for the Dest analysis, or keep `agg_route`
  standalone for a routes table/matrix.

Keep `agg_airline`, `agg_distance_bucket` as standalone (no relationships needed —
they are pre-aggregated factor tables).

## 4. DAX measures (create in a dedicated `_Measures` table)

```DAX
Total Flights        = SUM(fact_airport_hour[sched_arr])
Operated Arrivals    = SUM(fact_airport_hour[arr_operated])
Cancelled Arrivals   = SUM(fact_airport_hour[arr_cancelled])
Cancellation %       = DIVIDE([Cancelled Arrivals], [Total Flights])
Avg Arr Delay (min)  = DIVIDE(SUMX(fact_airport_hour, fact_airport_hour[arr_delay_mean]*fact_airport_hour[arr_operated]), [Operated Arrivals])
Avg Dep Delay (min)  = DIVIDE(SUMX(fact_airport_hour, fact_airport_hour[dep_delay_mean]*fact_airport_hour[dep_operated]), SUM(fact_airport_hour[dep_operated]))
Delayed >=15 Arr     = SUM(fact_airport_hour[arr_del15])
Delayed >=15 %       = DIVIDE([Delayed >=15 Arr], [Operated Arrivals])
On-Time %            = 1 - [Delayed >=15 %]
Est. Passengers      = SUM(fact_airport_hour[est_arr_passengers])   -- label "estimated"

-- Predictions (Page 3 / Page 4)
Predicted Avg Delay      = AVERAGE(fact_predictions[predicted_arr_delay_min])
Avg P(delay>=15)         = AVERAGE(fact_predictions[pct_arr_delayed_15])
Expected Delayed Flights = SUM(fact_predictions[expected_delayed_flights])
Scheduled Departures     = SUM(fact_predictions[sched_dep])
Traffic vs Usual %       = AVERAGE(fact_predictions[traffic_vs_usual])
Prediction MAE           = AVERAGEX(fact_predictions, ABS(fact_predictions[predicted_arr_delay_min] - fact_predictions[actual_arr_delay_min]))
```
Note the weighted averages for delay: `arr_delay_mean` is a per-hour mean, so weight
by `arr_operated` when rolling up. Add a `[Status]` measure (SWITCH on `Traffic vs Usual %`
/ `Avg P(delay>=15)`) returning "Calm" / "Busy" / "Disrupted" for the Page-3 badge.

## 5. Pages

### Page 1 — Network Overview (Import)
- Header KPI cards: **Total Flights**, **On-Time %**, **Delayed ≥15 %**, **Avg Arr Delay**,
  **Cancellation %**, **Est. Passengers** (labelled estimated).
- Line/area: flights and **Avg Arr Delay** over `dim_date` (monthly toggle).
- **Map** (Azure Maps): bubbles at `dim_airport` lat/long, size = Total Flights,
  colour = Avg Arr Delay (theme diverging min→center→max).
- Bar: top airports by Avg Arr Delay; bar: flights by `Airline` (from `agg_airline`).
- Slicers: `dim_date` (date range), `season`, `airport_code`.

### Page 2 — Delay Factors (Import, associational)
- Matrix/heatmap: Avg Arr Delay by `hour` (rows) × `day_name` (cols).
- Column: Avg Arr Delay by `month_name` / `season`.
- Bar: Avg Arr Delay & Delayed ≥15 % by `Airline` (`agg_airline`).
- Scatter: `agg_route` flights (x) vs `arr_delay_mean` (y), size = flights.
- Column: `agg_distance_bucket` Avg Arr Delay by bucket.
- Caption text box: BTS cause columns unavailable → associational analysis.

### Page 3 — Live Delay Dashboard ("landing page"; DirectLake live, or local fallback)
Designed as a website landing page; clean, few numbers, big.
- Top bar: **airport** dropdown slicer + **horizon/hours** slicer (`horizon_h`) +
  an "as of `prediction_ts`" card.
- Hero cards: **Predicted Avg Delay (next Nh)**, **Expected Delayed Flights**,
  **Scheduled Departures (next Nh)**, **Traffic vs Usual %** (e.g. "+20%"), **Status** badge.
- Map: airports coloured by Predicted Avg Delay (severity), size = Scheduled Departures.
- Column: per-airport **5-horizon** Predicted Avg Delay with a **15-min reference line**.
- Optional KPI sparkline of `predictions_history` (cloud) for the trend.
- Source toggle: point visuals at `predictions_latest` (DirectLake) for live, or
  `fact_predictions` (Import) for an offline demo — same fields.

### Page 4 — Model Accuracy (optional, Import)
- Predicted vs Actual scatter (`predicted_arr_delay_min` vs `actual_arr_delay_min`),
  reference diagonal.
- **Prediction MAE** by `horizon_h` (column).
- Calibration: bucket `pct_arr_delayed_15` deciles vs observed `actual_arr_del15` rate.

## 6. Live page setup (DirectLake)
1. In Fabric, create a **DirectLake semantic model** on `FlightData_Lakehouse`
   including `predictions_latest` (+ uploaded `dim_airport`, `agg_baseline_volume`).
2. Use the enrichment columns (`sched_arr_count`, `sched_dep_count`,
   `expected_delayed_flights`) added by the updated `nb_inference.ipynb` — re-run the
   notebook once so the tables carry them (see the notebook's header note).
3. Relate `predictions_latest[airport_code]` → `dim_airport`; build the Page-3 visuals.

## 7. (Light) Part 2 — end-user web/API sketch
Deferred per scope; recommended approach when built: a small **FastAPI** app
reading `outputs/powerbi/fact_predictions.parquet` (offline) or the OneLake table,
reusing `src/inference/predictor.py`. Suggested endpoints:
- `GET /flight-delay?airport=ATL&horizon=2` → predicted delay + P(delay≥15).
- `GET /airport-traffic?airport=ATL&hours=4` → expected (est.) passengers/flights and
  **traffic vs usual** for the next N hours.
Add `fastapi`/`uvicorn` under a new optional extra in `pyproject.toml`. Mirrors the
cloud `GetFlightPredictions` contract; demoable offline.

## 8. Verification checklist
- `outputs/powerbi/` has all 9 CSVs; row counts logged by the exporter.
- `dim_airport` has 70 rows; `fact_predictions` `airport_code` count = 70.
- KPI sanity: On-Time % ≈ 0.78–0.82; Avg Arr Delay ≈ 8–12 min on 2018–19.
- Page 3: changing the airport/horizon slicer updates all cards and the map.
- Page 4: MAE ≈ the eval MAE (~10 min) and rises with horizon.
