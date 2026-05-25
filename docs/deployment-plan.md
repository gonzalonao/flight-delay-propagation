# Production Deployment Plan: Flight Delay Propagation ML System

> **Scope: demo-only TFM deployment.** Built on **Azure for Students** (no GPU
> SKUs). The monthly retraining pipeline is shipped as code-as-documentation
> and is intentionally **not enabled** — the architecture is defensible at
> thesis defense without actually running it. Everything else (ingestion,
> inference, serving, dashboard) is wired live.

---

## 🚦 Resume here (session handoff — last updated 2026-05-25)

**Working branch:** `claude/analyze-model-format-w89p1`
**Current phase:** **Phase 3 — Inference pipeline.**

### Phase 3 steps (in order)

- ✅ **Step 1**: Build project wheel locally:
  ```
  cd C:\Users\gonza\dev\flight-delay-propagation
  pip install build
  python -m build --wheel
  ```
  Output: `dist/flight_delay_propagation-0.1.0-py3-none-any.whl`

- ✅ **Step 2**: Upload wheel to OneLake — Fabric portal → `FlightData_Lakehouse` → Files → create folder `packages/` → upload the `.whl` file.

- ✅ **Step 3**: Create Fabric custom environment `env_inference` — Fabric workspace → New item → Environment. Add libraries: `torch>=2.1`, `torch-geometric>=2.4`, `pyarrow>=14`, `azure-storage-file-datalake>=12`, `azure-identity>=1.15`, `deltalake>=0.14`. Publish the environment (takes ~10 min). *(deltalake no longer used at runtime — Spark `saveAsTable` writes the Delta tables; lib can stay or be dropped on next env rebuild.)*

- ✅ **Step 4**: Import `fabric/notebooks/nb_inference.ipynb` into `TFM_Flight_Prediction` workspace → New item → Import notebook.

- ✅ **Step 5**: In the notebook settings — attach to `env_inference` environment + add `FlightData_Lakehouse` as the default lakehouse.

- ✅ **Step 6**: Run `nb_inference.ipynb` manually (Run all). Verified 2026-05-25:
  - Cell 13 prints `350 rows` in `predictions_latest` (70 airports × 5 horizons)
  - `airport_code` has 70 distinct values
  - `predicted_arr_delay_min` values in a plausible range

- ⏳ **Step 7** *(in progress)*: Create `pl_hourly_predict` pipeline in Fabric Data Factory — single Notebook Activity pointing at `nb_inference`, timeout 20 min, no retry.

- ⏳ **Step 8**: Add schedule trigger `hourly_at_15` — Fixed, hourly, at minute `:15` UTC, start `2026-05-25T00:15:00Z`. **Enable only after two consecutive manual runs of `pl_hourly_predict` pass.**

### Phase 2 — DONE ✅

- ✅ `nb_backfill_buffer.ipynb` hostname filled in; FUNCTION_KEY pasted at runtime
- ✅ `pl_fake_ingestion.json` hostname + startTime filled in; blueprint committed
- ✅ **Step 1**: Default host key copied from Azure Portal
- ✅ **Step 2**: `nb_backfill_buffer` run — 168 calls completed
- ✅ **Step 3**: 168 partitions verified in `Files/live_feed/rolling_buffer/`
- ✅ **Step 4**: `pl_fake_ingestion` created in Fabric Data Factory UI (Web Activity + If Condition + HTTP connection `conn_flight_ingest_api`)
- ✅ **Step 5**: Schedule trigger `hourly_at_05` enabled (Fixed, hourly, minute=5, UTC)
- ✅ **Step 6**: Partition verified in `live_feed/rolling_buffer/` within 10 min of `:05` UTC

### What's done

| | |
|---|---|
| Fabric workspace `TFM_Flight_Prediction` + Lakehouse `FlightData_Lakehouse` | ✅ Provisioned (region: francecentral, F2 trial capacity) |
| `Combined_Flights_2022.parquet` uploaded to OneLake | ✅ at `Files/raw/historical_2022/` |
| 4 champion artifacts uploaded to OneLake | ✅ at `Files/models/champion/` |
| Code audit fixes | ✅ Committed `b449d08` |
| `scripts/extract_artifacts.py` | ✅ Committed `d9efdff` |
| Demo-scope plan rewrite | ✅ Committed `5aa1c6a` |
| `bootstrap.sh` filled with real OneLake names | ✅ Committed `ce3a5f3` |
| OneLake smoke test (DataLakeServiceClient) | ✅ Committed `09c9b8a` |
| `GetFlightData` refactored (adlfs → DataLakeServiceClient) | ✅ Download-then-filter with module-level cache |
| `requirements.txt` updated | ✅ `adlfs` replaced with `azure-storage-file-datalake` |
| Local `func start` test | ✅ 200 OK, 1148 rows, all 3 OneLake targets written |
| Azure Function App deployed | ✅ `func-flight-ingest` in `rg-tfm-flight` (francecentral, Consumption, Python 3.11, Linux) |
| Managed Identity → Fabric workspace Member | ✅ |
| Cloud endpoint verified | ✅ POST returned `status: ok` |

### Azure resource names (for reference)

| Resource | Name |
|---|---|
| Resource group | `rg-tfm-flight` |
| Storage account | `stflightfunc` |
| Function App (ingestion) | `func-flight-ingest` |
| Function App URL | `https://func-flight-ingest.azurewebsites.net/api/v1/flights/ingest` |

### What's next — Phase 2: Ingestion pipeline + backfill

1. Create `pl_fake_ingestion` pipeline in Fabric Data Factory — Web Activity calling `POST /v1/flights/ingest`, triggered hourly at :05 UTC.
2. Run `nb_backfill_buffer` notebook — calls GetFlightData 168 times (one per hour, 7 days back) to populate `rolling_buffer/` before inference starts.
3. Verify 168 partitions exist in `live_feed/rolling_buffer/`.

Then Phase 3 (inference pipeline), Phase 4 (Power BI), Phase 5 (Predictions API).

### Then Phase 2 → 5 in order (see "Implementation Sequence" below).

### Reference: memory files for fresh sessions

If starting a new Claude Code session, read these in order:

- `~/.claude/projects/G--My-Drive-Master-ESESA-TFM-flight-delay-propagation/memory/MEMORY.md` (index)
- `user_role.md` — author's stack and what to explain vs. assume
- `project_tfm_context.md` — demo-only scope, days of runway, retrain is scaffolding
- `infra_azure_for_students.md` — no GPU SKUs, francecentral region
- `infra_fabric_resources.md` — workspace + Lakehouse names + trained-config schema (hidden_dim=256, num_heads=8, weather v4)

---

## Context

The model (Seq2SeqGNN, Transformer encoder-decoder) is fully trained on the `feat/weather-integration` branch and outputs hourly delay predictions for 70 US airports across 5 horizons (1h, 2h, 4h, 6h, 8h ahead). The goal is to deploy this to a Microsoft Azure / Fabric pipeline that demonstrates how a production system would:
1. Simulate live data ingestion via a real HTTP API (backed by historical 2022 files)
2. Run hourly inference and expose results to Power BI
3. ~~Retrain the model monthly on accumulated data~~ — **scaffolded only, not enabled** (Student-sub GPU restriction; the trained champion is uploaded once and stays the champion).
4. Expose predictions via a second REST API and an interactive airport map

---

## Current Status

| Step | Status |
|---|---|
| Fabric workspace + Lakehouse created | ✅ Done |
| Azure ML workspace, compute, environment | ⏭️ Skipped (demo-only; retrain is scaffolding) |
| Upload `Combined_Flights_2022.parquet` to OneLake | ✅ Uploaded to `Files/raw/historical_2022/` |
| Trained model checkpoint (.pt) | ✅ Stripped via `prep_inference_checkpoint.py` |
| 3 deployment artifacts (airport_map.json, feature_stats.pt, metadata.json) | ✅ Generated via `extract_artifacts.py` |
| All 4 champion artifacts uploaded to OneLake | ✅ At `Files/models/champion/` |
| OneLake SDK choice verified | ✅ `azure-storage-file-datalake` confirmed |
| `GetFlightData` Function deployed to Azure | ✅ `func-flight-ingest` (rg-tfm-flight, francecentral) |
| Inference notebook + Power BI + Predictions API | ⏳ Pending |
| Monthly retraining pipeline | 📄 Scaffolding only — never enabled |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       MICROSOFT FABRIC WORKSPACE                         │
│                                                                          │
│  OneLake (ADLS Gen2)                                                     │
│  ├── raw/historical_2022/Combined_Flights_2022.parquet  (seed data)      │
│  ├── raw/historical_2018_2021/*.parquet                 (train data)     │
│  ├── live_feed/class_b/YYYY/MM/DD/HH/flights.parquet   (actuals)        │
│  ├── live_feed/class_a_schedule/YYYY/MM/DD/HH/         (schedule)       │
│  ├── live_feed/rolling_buffer/  (last 168 hours, for lag features)      │
│  ├── predictions/latest/        (Delta table — DirectLake source)       │
│  ├── predictions/history/       (Delta table — append-only audit)       │
│  └── models/champion/           (checkpoint + airport_map + stats)      │
│                                                                          │
│  Data Factory Pipelines                                                  │
│  ├── pl_fake_ingestion   — hourly :05, Web Activity → GetFlightData API │
│  ├── pl_hourly_predict   — hourly :15, Notebook activity                │
│  └── pl_monthly_retrain  — 1st of month 02:00 UTC, 3 activities        │
│                                                                          │
│  Power BI (DirectLake)                                                   │
│  ├── Page 1: Azure Maps airport heatmap (delay severity + slider)       │
│  └── Page 2: Per-airport 5-horizon bar chart                            │
└──────────────────────────────────────────────────────────────────────────┘
         │ Web Activity (hourly)          │ AzureML job (monthly)
         ▼                               ▼
┌─────────────────────────────┐  ┌──────────────────────────────────────┐
│  Azure Functions (App #1)   │  │  Azure Machine Learning              │
│  "Fake Flight Data API"     │  │  Compute: NC4as_T4_v3 (T4 GPU)      │
│                             │  │  Environment: flight-delay-prod:1    │
│  POST /v1/flights/ingest    │  │  Job: train.py --config production   │
│  ├── reads 2022 parquet     │  │  Output → models/challenger/         │
│  │   from OneLake via ADLS  │  └──────────────────────────────────────┘
│  ├── writes class_b to      │
│  │   live_feed/class_b/     │
│  ├── writes class_a to      │
│  │   live_feed/class_a_sched│
│  ├── writes rolling_buffer  │
│  ├── trims buffer >168h     │
│  └── returns status JSON    │
└─────────────────────────────┘
         │ HTTP (GET)
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Azure Functions (App #2) + API Management (Consumption)        │
│  "Predictions API"                                              │
│  GET /predictions?airport=ATL&horizon=2                         │
│  Reads predictions/latest Delta table via Managed Identity      │
│  APIM: rate-limit 60/min, 55s response cache                   │
└─────────────────────────────────────────────────────────────────┘
```

### Why Two Separate Function Apps

The ingestion API (`GetFlightData`) and the predictions API (`GetFlightPredictions`) have different runtime requirements:

- **GetFlightData** needs `pyarrow` + `adlfs` + ADLS write access — heavyweight, runs once per hour, memory-intensive (reads a large parquet)
- **GetFlightPredictions** needs `deltalake` + ADLS read access — lightweight, serves live user traffic

Keeping them in separate Function Apps lets you scale, redeploy, and set resource limits independently.

---

## Files to Create

### Azure Functions — App 1: Fake Flight Data API

| Path | Purpose |
|---|---|
| `azure_functions/flight_data_api/GetFlightData/__init__.py` | `POST /v1/flights/ingest?timestamp=<ISO>` — reads 2022 parquet from OneLake via ADLS SDK, writes class_b + class_a + rolling_buffer, trims buffer, returns status JSON |
| `azure_functions/flight_data_api/GetFlightData/function.json` | HTTP trigger, POST only, authLevel: function |
| `azure_functions/flight_data_api/host.json` | Runtime config, Python 3.10, logging |
| `azure_functions/flight_data_api/requirements.txt` | `pyarrow>=14.0`, `adlfs>=2023.9`, `azure-storage-file-datalake>=12.0`, `azure-identity>=1.15`, `pandas>=2.0` |

### Azure Functions — App 2: Predictions API

| Path | Purpose |
|---|---|
| `azure_functions/predictions_api/GetFlightPredictions/__init__.py` | `GET /predictions?airport=ATL&horizon=2` — reads Delta table via `deltalake` + Managed Identity, returns JSON |
| `azure_functions/predictions_api/GetFlightPredictions/function.json` | HTTP trigger, GET only, authLevel: function |
| `azure_functions/predictions_api/host.json` | Runtime config |
| `azure_functions/predictions_api/requirements.txt` | `deltalake>=0.14`, `azure-storage-file-datalake>=12.0`, `azure-identity>=1.15` |

### Fabric Notebooks

| Path | Purpose |
|---|---|
| `fabric/notebooks/nb_backfill_buffer.ipynb` | One-time: calls `GetFlightData` in a loop for the past 168 hours to populate rolling_buffer before predictions start — **already written** |
| `fabric/notebooks/nb_inference.ipynb` | Hourly: load last 6h + next-8h schedule, build graph sequence, run Seq2SeqGNN, write Delta |
| `fabric/notebooks/nb_submit_aml_job.ipynb` | Monthly: submit AzureML training job via `azure-ai-ml` SDK, write run_id to log table |
| `fabric/notebooks/nb_champion_challenger.ipynb` | Monthly: compare new model MAE vs champion; promote if ≥2% improvement, copy all 4 artifacts atomically |

> `nb_ingest_live_feed.ipynb` has been superseded by the `GetFlightData` Azure Function. The Fabric pipeline calls the Function via a Web Activity — no notebook needed for ingestion.

### Fabric Pipeline Definitions (JSON)

| Path | Trigger | Key activity |
|---|---|---|
| `fabric/pipelines/pl_fake_ingestion.json` | Every hour at :05 | **Web Activity** → `POST /v1/flights/ingest` |
| `fabric/pipelines/pl_hourly_predict.json` | Every hour at :15 | Notebook Activity → `nb_inference` |
| `fabric/pipelines/pl_monthly_retrain.json` | 1st of month at 02:00 UTC | submit → Until poll → champion/challenger |

### Azure ML

| Path | Purpose |
|---|---|
| `aml/compute/training_cluster.yml` | NC4as_T4_v3 cluster, min=0/max=1, 120s idle scale-down — **already written** |
| `aml/environments/flight-delay-prod.yml` + `Dockerfile` | PyTorch 2.1+cu118, PyG 2.4, matching CUDA wheels — **already written** |
| `aml/jobs/train_job.yml` | CLI v2 job YAML; mounts OneLake historical + live_feed; registers output to model registry — **already written** |

### Config & APIM

| Path | Purpose |
|---|---|
| `configs/production.yaml` | Production training config — **already written** |
| `apim/api_definition.yaml` | OpenAPI 3.0 spec for predictions API; `GET /predictions` with airport + horizon params |

### Deployment

| Path | Purpose |
|---|---|
| `deploy/bootstrap.sh` | One-time setup script — **already written**, covers AzureML + OneLake upload |
| `deploy/prep_inference_checkpoint.py` | Strip optimizer state from checkpoint — **already written** |

### Power BI

| Path | Purpose |
|---|---|
| `powerbi/flight_delay_dashboard.pbix` | Page 1: Azure Maps heatmap. Page 2: Per-airport 5-horizon bar chart, 15-min reference line |

---

## Files Already Modified (Done)

| File | Change |
|---|---|
| `src/utils/io.py` | Added `load_checkpoint_inference_only()` with `weights_only=True` |
| `src/data/graph_builder.py` | Added `__all__`; `build_graph_dataset` now returns 3-tuple `(graphs, airport_map, norm_stats)` |
| `scripts/train.py` | Saves `airport_map.json`, `feature_stats.pt`, `metadata.json` automatically after GNN training |
| `scripts/evaluate.py` | Updated to unpack 3-tuple from `build_graph_dataset` |
| `pyproject.toml` | Added `[deploy]` optional extra for Azure SDK deps |

---

## Pipeline Details

### pl_fake_ingestion — Web Activity calling GetFlightData

```
Trigger: every hour at :05 UTC
Timeout: 20 minutes
On failure: alert + leave live_feed unchanged (inference will reuse last hour)

Activity: WebActivity "call_flight_data_api"
  Method: POST
  URL: https://<function-app>.azurewebsites.net/api/v1/flights/ingest
  Headers: { "x-functions-key": "@{linkedService().functionKey}" }
  Body: { "timestamp": "@{formatDateTime(pipeline().TriggerTime, 'yyyy-MM-ddTHH:00:00Z')}" }

On success: check response body status == "ok"
  → pipeline succeeds, inference pipeline can proceed
On failure / status != "ok":
  → pipeline fails, alert fires, last hour's data remains in rolling_buffer
```

### GetFlightData Function Logic

```python
# POST /v1/flights/ingest?timestamp=2026-05-19T14:00:00Z
# (or timestamp in request body JSON)

now_ts   = parse_timestamp(request)          # e.g. 2026-05-19T14:00:00Z
equiv_dt = now_ts.replace(year=2022)         # → 2022-05-19T14:00:00Z

# Read from OneLake via ADLS Gen2 + Managed Identity
credential = DefaultAzureCredential()
fs = adlfs.AzureBlobFileSystem(account_name=ONELAKE_ACCOUNT, credential=credential)

table_b = pq.read_table(SOURCE_PARQUET, filesystem=fs, filters=[
    ("Month", "=", equiv_dt.month),
    ("DayofMonth", "=", equiv_dt.day),
    ("CRSDepTime", ">=", equiv_dt.hour * 100),
    ("CRSDepTime", "<",  (equiv_dt.hour + 1) * 100),
], columns=CLASS_B_COLS)

# Write class_b, class_a schedule (+8h), rolling_buffer to OneLake
# Trim rolling_buffer partitions older than 168 hours
# Return: {"status": "ok", "rows_ingested": 2314, "timestamp": "...", "equivalent_2022": "..."}
```

### pl_hourly_predict Logic (nb_inference.ipynb)

Python kernel notebook (not Spark — PyTorch + PyG not available in Fabric Spark by default):

1. Load last 6 class_b partitions + next-8h class_a partitions from rolling_buffer
2. Call `clean_flights`, `fill_delay_nulls`, `encode_time` — **do NOT call `filter_top_airports`** (use frozen `airport_map.json`)
3. Filter to `airport_map.keys()` only
4. Call `create_temporal_graphs(df, airport_map, ...)` to build 6-snapshot sequence
5. Apply frozen normalization stats (`feature_stats.pt`)
6. `build_model(config, input_dim=55, edge_dim=5)` — config must have `loss: multi_task` to get `output_channels=3`
7. `load_checkpoint_inference_only(checkpoint_path, model)` (safe, `weights_only=True`)
8. `model.eval(); torch.no_grad(); preds = model(sequence)` → `[70, 5, 3]`; use channel 0 (ArrDelay)
9. Write 70-row DataFrame to `predictions/latest` (overwrite) and `predictions/history` (append)

### pl_monthly_retrain Logic

```
[1] nb_submit_aml_job      → submits train_job.yml via azure-ai-ml SDK
[2] Until(poll every 5min) → waits for AzureML run completion (max 6h)
[3] nb_champion_challenger → downloads challenger MAE, compares composite MAE
                             if (champion_mae - challenger_mae) / champion_mae >= 0.02:
                               atomic copy of .pt + airport_map.json + feature_stats.pt + metadata.json
                               to models/champion/
```

### Backfill Flow (one-time, before first prediction)

`nb_backfill_buffer` calls `GetFlightData` 168 times in a loop, once per hour going back 7 days. This means the backfill uses the same code path as the live pipeline — not a separate notebook that reads parquets directly. The rolling_buffer is fully populated before `pl_hourly_predict` is enabled.

---

## End Products

### 1. Power BI Dashboard (DirectLake, hourly auto-update)
- Page 1: Azure Maps visual — airport bubbles, color = mean predicted ArrDelay, size = scheduled departure count, horizon slicer
- Page 2: Selected airport drill-down — 5-horizon grouped bar chart, 15-min delay threshold reference line

### 2. Predictions REST API (Azure Functions App #2 + APIM)
- `GET /predictions?airport=ATL&horizon=2`
- Response: `{"airport":"ATL","horizon":2,"predicted_arr_delay_min":14.3,"prediction_ts":"2026-05-19T15:00Z"}`
- APIM: 60 req/min rate limit, 55s response cache, OpenAPI schema

### 3. Interactive Airport Map
- Azure Maps custom visual in Power BI, color-encodes delay severity across all 70 airports
- Clicking an airport opens the Page 2 drill-down

---

## Monthly Cost Estimate (France Central, Azure for Students + Fabric trial)

| Service | SKU / Usage | Monthly Cost |
|---|---|---|
| Microsoft Fabric | F2 capacity via **60-day free trial** | $0 |
| OneLake storage | ~10 GB LRS during demo period | <$1 (covered by Student credit) |
| ~~Azure ML compute~~ | Not provisioned | $0 |
| Function App #1 (GetFlightData) | Consumption, 720 calls/month, ~2s each | $0 (free tier) |
| Function App #2 (GetFlightPredictions) | Consumption, ~5K calls/month | $0 (free tier) |
| Function App storage accounts (×2) | 2 × ~1 GB LRS | <$0.10 |
| Azure API Management | Consumption tier, ~5K calls/month | $0 (free tier) |
| **Total (demo period)** | | **≈ $0** (within $100 Student credit + Fabric trial) |

After the Fabric 60-day trial ends, F2 pay-as-you-go is ~$365/mo on PAYG subs.
For TFM defense, time the demo window so the trial is still active, then tear
down the F2 capacity (the Lakehouse data persists in storage either way).

---

## Technical Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **GetFlightData cold start + large parquet read** | High | 7 GB parquet on Consumption plan can take 5–15s on cold start. Pre-partition the 2022 parquet by month (12 files × ~600 MB) so filter pushdown only scans 1 file. Set Function timeout to 300s in host.json. |
| **PyTorch not supported in Fabric Spark** | Critical | Use Python kernel notebooks (not Spark) for inference. Install PyTorch via Fabric custom environment YAML. |
| **Lag feature bootstrap (cold start)** | High | Run `nb_backfill_buffer` (calls GetFlightData 168×) before enabling `pl_hourly_predict`. Set `WARM_UP` flag in predictions table when <24 buffer partitions exist. |
| **Fixed airport_map must be frozen** | High | Save `airport_map.json` from training run. Inference notebook skips `filter_top_airports`; filters directly to frozen keys. |
| **Normalization stats must travel with checkpoint** | High | Champion promotion copies all 4 artifacts atomically (`.pt`, `airport_map.json`, `feature_stats.pt`, `metadata.json`). |
| **`output_channels=3` config dependency** | Medium | `configs/production.yaml` must specify `loss: multi_task`. Inference uses only channel 0 (ArrDelay). |
| **Managed Identity RBAC on OneLake** | Medium | Function App's system-assigned MI needs `Storage Blob Data Contributor` on the Lakehouse ADLS Gen2 endpoint — grant this in Azure Portal before deploying. |
| **`torch.load(weights_only=False)` deprecated** | Medium | Strip optimizer state at deploy time; inference notebook uses `load_checkpoint_inference_only` with `weights_only=True`. |
| **Seq2SeqGNN forward() expects `list[Data]`** | Medium | Inference notebook must pass exactly 6 snapshots. If <6 hours available, skip inference and set `WARM_UP` flag. |

---

## Implementation Sequence

| Phase | Status | Deliverable |
|---|---|---|
| **0a — Fabric Lakehouse** | ✅ Done | Lakehouse created |
| **0b — Azure ML setup** | ⏭️ Skipped | Not needed for demo (no GPU on Student sub, retrain never runs) |
| **0c — Upload data** | ⏳ Pending | `Combined_Flights_2022.parquet` (only) in OneLake — 2018–2021 not needed since we never retrain |
| **0d — Checkpoint prep** | ⏳ Pending | `extract_artifacts.py` → `prep_inference_checkpoint.py` → 4 champion artifacts uploaded to OneLake |
| **1 — GetFlightData API** | ✅ Done | `func-flight-ingest` deployed, Managed Identity granted, cloud POST returns `status: ok` |
| **2 — Ingestion pipeline** | ✅ Done | `pl_fake_ingestion` live at `:05` UTC; 168-partition backfill verified; rolling_buffer populated |
| **3 — Inference pipeline** | ⏳ In progress | `nb_inference` runs end-to-end in Fabric (350 rows in `predictions_latest`); `pl_hourly_predict` pipeline + schedule pending |
| **4 — Power BI** | ⏳ Pending | DirectLake semantic model connected; Azure Maps visual + drill-down page published |
| **5 — Predictions API** | ⏳ Pending | `GetFlightPredictions` Function + APIM deployed; `GET /predictions` returns JSON |
| **6 — Retraining pipeline** | 📄 Scaffolding only | `aml/`, `nb_submit_aml_job`, `nb_champion_challenger`, `pl_monthly_retrain` exist as code but are **not** deployed or triggered. Defended at the thesis as the production-extension path. |

**Demo target: ~3–4 days of focused work** (phases 0c → 5).

---

## Verification

1. **GetFlightData API**: `curl -X POST "https://<fn>.azurewebsites.net/api/v1/flights/ingest?timestamp=2026-05-19T14:00:00Z" -H "x-functions-key: <key>"` → `{"status":"ok","rows_ingested":~2000}`; check OneLake for `live_feed/class_b/2026/05/19/14/flights.parquet`
2. **Ingestion pipeline**: After enabling `pl_fake_ingestion`, verify a new partition appears in `live_feed/rolling_buffer/` within 10 minutes past each hour
3. **Backfill**: Run `nb_backfill_buffer` once; verify 168 partitions exist in `rolling_buffer/`
4. **Inference**: After enabling `pl_hourly_predict`, check `predictions/latest` — exactly 70 rows, 5 prediction columns, updated within 15 min past each hour
5. **Predictions API**: `curl "https://<apim-url>/predictions?airport=ATL&horizon=2"` → JSON with `predicted_arr_delay_min` between -10 and 120
6. **Retraining**: Submit AzureML job manually; verify challenger checkpoint in `models/challenger/`; verify `nb_champion_challenger` promotes or rejects correctly
7. **Power BI**: Map loads, bubbles colored, horizon slicer works, timestamp reflects last inference run
