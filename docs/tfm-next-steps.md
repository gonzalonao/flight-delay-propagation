# TFM Productivization — Next Steps Guide

> Action checklist for finishing the TFM deliverables now that the **data +
> inference layer is built, validated, and pushed** (branch `feat/tfm-powerbi`).
> Companion to `docs/powerbi-report-spec.md` (the visual/measure spec). This
> file is the *procedure*; that file is the *blueprint*.

## Where things stand

| Piece | Status |
|---|---|
| Local export (`scripts/export_powerbi.py`) → `outputs/powerbi/*.csv` | ✅ Built + run (9 tables) |
| Local batch inference (`scripts/predict.py`, `src/inference/predictor.py`) | ✅ Built + tested (MAE 10.05 = eval baseline) |
| Cloud notebook enrichment + pct sigmoid fix (`nb_inference.ipynb`) | ✅ Edited (needs Fabric re-run) |
| Predictions API schema fix (`GetFlightPredictions`) | ✅ Fixed (needs redeploy to take effect) |
| Report spec + theme (`docs/powerbi-report-spec.md`, `powerbi/theme.json`) | ✅ Written |
| **The `.pbix` itself** | ⬜ **Manual build in Power BI Desktop (Step 2 below)** |
| Live DirectLake semantic model | ⬜ Manual in Fabric (Step 3) |
| Part 2 web/API | ⬜ Optional (Step 5) |

---

## Step 1 — (Re)generate the data  *(only when you want fresh numbers)*

The CSVs already exist in `outputs/powerbi/`. To regenerate:

```powershell
$env:PYTHONIOENCODING = "utf-8"
# base tables only (fast, ~25s, no GPU):
uv run python scripts/export_powerbi.py --no-predictions
# full export incl. predictions over the test split (~90s, uses the champion + GPU):
uv run python scripts/export_powerbi.py `
  --checkpoint outputs/runs/20260514-213242/seq2seq_gnn_large.pt `
  --config configs/weekend/seq2seq_gnn_large.yaml --split test
```

Output (in `outputs/powerbi/`, both `.csv` and `.parquet`): `dim_airport`,
`dim_date`, `dim_hour`, `fact_airport_hour`, `agg_baseline_volume`,
`agg_airline`, `agg_route`, `agg_distance_bucket`, `fact_predictions`.

---

## Step 2 — Build the Power BI report (Power BI Desktop)  ← do this first

This is the mandatory deliverable. ~half a day of GUI work.

### 2.1 Import the data
- **Get Data → Text/CSV**, import each file from `outputs/powerbi/`. (Or use the
  `.parquet` files via **Get Data → Parquet** if you prefer.)
- In Power Query, confirm types: `date`/`ts_hour`/`target_ts`/`prediction_ts` →
  *Date/Time*; `airport_code`, `Airline`, `Origin`, `Dest` → *Text*; counts → *Whole number*;
  delays/rates → *Decimal*. Click **Close & Apply**.

### 2.2 Model the relationships (Model view)
Create one-to-many (dim → fact), single direction:
- `dim_airport[airport_code]` → `fact_airport_hour`, `agg_baseline_volume`, `fact_predictions`.
- `dim_date[date]` → `fact_airport_hour[date]`.
- `dim_hour[hour]` → `fact_airport_hour[hour]`.
- Leave `agg_airline`, `agg_route`, `agg_distance_bucket` standalone (already aggregated).
- Select `dim_date` → **Table tools → Mark as date table** → `date`.

### 2.3 Measures
- **Home → Enter Data** → create an empty table named `_Measures` (delete its
  one blank column after creating a first measure).
- Add the measures from `docs/powerbi-report-spec.md §4` (Total Flights,
  On-Time %, Delayed ≥15 %, Avg Arr/Dep Delay, Cancellation %, Est. Passengers,
  Predicted Avg Delay, Expected Delayed Flights, Traffic vs Usual %, Prediction MAE,
  Status badge). Copy the DAX verbatim.

### 2.4 Theme + pages
- **View → Themes → Browse for themes** → `powerbi/theme.json`.
- Build the four pages per `docs/powerbi-report-spec.md §5`:
  - **Page 1 — Network Overview** (KPI cards, time trend, airport map, top airports/airlines).
  - **Page 2 — Delay Factors** (hour×day heatmap, by airline/route/distance; add the "associational, no BTS causes" caption).
  - **Page 3 — Live Delay Dashboard** (airport + horizon slicers, hero cards, severity map, 5-horizon bar with 15-min line). For the offline demo, point its visuals at `fact_predictions`.
  - **Page 4 — Model Accuracy** (predicted vs actual, MAE by horizon, pct calibration).
- For the map: use the **Azure Map** or **Map** visual with `dim_airport[latitude]`/`[longitude]`. (If Azure Maps is greyed out: **File → Options → Preview/Security → enable Azure Maps**.)

### 2.5 Save
- Save as `powerbi/flight_delay_dashboard.pbix` (this path is gitignored for the
  data but you may commit the `.pbix` if it's small enough; otherwise keep it
  outside git or use Git LFS).

### 2.6 Sanity check (from the spec §8)
On-Time % ≈ 0.80, Avg Arr Delay single digits, `dim_airport` = 70 rows,
`fact_predictions` 70 airports, Page-3 slicers update all cards, Page-4 MAE ≈ 10 min rising with horizon.

---

## Step 3 — Wire the live dashboard (cloud DirectLake)  *(the "live" half of hybrid)*

Only needed for the live page narrative; Page 3 already works offline from `fact_predictions`.

1. **Re-import** the updated `fabric/notebooks/nb_inference.ipynb` into the Fabric
   workspace and **Run all once**. This refreshes `predictions_latest`/`history`
   with the corrected `pct_arr_delayed_15` (sigmoid) and the new
   `sched_arr_count`/`sched_dep_count`/`expected_delayed_flights` columns.
   *No wheel rebuild needed* (the notebook's own imports are unchanged).
2. **Redeploy** the `GetFlightPredictions` Azure Function so the API serves the
   corrected long schema (`func azure functionapp publish <app>` or via the portal).
3. **Upload the reference tables** to OneLake so the live page can compute
   "traffic vs usual": load `outputs/powerbi/dim_airport.csv` and
   `agg_baseline_volume.csv` into `FlightData_Lakehouse` as Delta tables (drop the
   CSVs in `Files/` and load-to-table, or a small one-cell notebook).
4. Build a **DirectLake semantic model** on the lakehouse including
   `predictions_latest` + the two reference tables; relate on `airport_code`;
   recreate the Page-3 visuals against it.

---

## Step 4 — Verify end-to-end
- [ ] Local report opens, all 4 pages render, slicers work.
- [ ] Numbers match the sanity ranges (§2.6).
- [ ] (If doing Step 3) Live page updates after a notebook run; `predictions_latest`
      has the 3 new columns; API returns JSON (no 404).
- [ ] Re-run `uv run python -m pytest -q` → all green.

---

## Step 5 — (Optional) Part 2: end-user web/API
Deferred per scope. When ready (see `docs/powerbi-report-spec.md §7`): a small
**FastAPI** app reading `outputs/powerbi/fact_predictions.parquet` (or the OneLake
table), reusing `src/inference/predictor.py`. Endpoints: `GET /flight-delay`,
`GET /airport-traffic`. Add `fastapi`/`uvicorn` as a new optional extra in
`pyproject.toml`. Say the word and I'll scaffold it.

---

## Housekeeping
- **PR / merge:** `feat/tfm-powerbi` is pushed. Open a PR:
  `https://github.com/gonzalonao/flight-delay-propagation/pull/new/feat/tfm-powerbi`
  (or `gh pr create`). Merge into `main` when the report is built.
- **Parked recall work:** unrelated; lives on `feat/improve-recall` and is
  documented in `docs/recall-improvement-status.md` (resume anytime).
- **Untracked by design:** `outputs/powerbi/*`
  (regenerable data — gitignored), `uv.lock`.

## File map (what does what)
| Path | Role |
|---|---|
| `scripts/export_powerbi.py` | Builds the star-schema tables for Power BI |
| `scripts/predict.py` | Local batch inference CLI |
| `src/inference/predictor.py` | Shared forward-pass + long-schema assembler |
| `docs/powerbi-report-spec.md` | Data model + DAX + page-by-page visual spec |
| `powerbi/theme.json` | Power BI theme |
| `fabric/notebooks/nb_inference.ipynb` | Live hourly inference (enriched) |
| `azure_functions/.../GetFlightPredictions` | Predictions REST endpoint (schema fixed) |
| `configs/default.yaml` → `powerbi:` | Export knobs (out_dir, seats_per_flight, split) |
