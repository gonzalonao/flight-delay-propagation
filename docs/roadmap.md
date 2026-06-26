# Roadmap / Future Work

Planned improvements, ordered roughly by impact. This is an honest list:
several items address gaps that exist in the current demo, not just
nice-to-haves.

## 1. Serve real weather to the live model *(headline)*

The champion is **trained with weather** (ERA5 reanalysis via Open-Meteo —
wind, precipitation/cloud, and a bucketed WMO category per airport-hour, see
[`src/data/weather.py`](../src/data/weather.py) and the weather block in
[`docs/ablation-log.md`](ablation-log.md)). But the **live inference path
does not feed weather**: the Fabric notebook
[`fabric/notebooks/nb_inference.ipynb`](../fabric/notebooks/nb_inference.ipynb)
zero-fills the weather feature columns, so production predictions currently
treat weather as "no signal".

Two things need to happen to close this:

1. **Wire a live/forecast weather source into inference.** The training
   features come from ERA5 *reanalysis*, which lags real time by several
   days — unusable for live serving. The fix is to call the Open-Meteo
   **Forecast** API (same schema, same `WeatherLookups` machinery in
   `src/data/graph_builder.py`) for the input window and the target
   horizons, instead of zero-filling.
2. **Fix the feature-dim mismatch.** The uploaded champion artifacts pair a
   109-dim checkpoint with 55-dim `feature_stats` (no-weather). The
   notebook currently pads the missing columns with mean=0/std=1, which is
   why zero-filling is "safe" but lossy. Regenerate `feature_stats.pt`
   (and `airport_map.json`/`metadata.json`) from a weather-enabled run via
   [`scripts/extract_artifacts.py`](../scripts/extract_artifacts.py) so the
   served features line up with what the model was trained on.

## 2. Quantify weather's contribution

Run the weather ablation (block H on/off, and per-group: wind /
precip_cloud / category) and report the per-horizon MAE delta. Hypothesis:
the biggest gains are at the 4/6/8 h horizons, where convective weather
(thunderstorms → NAS ground stops/holds) drives propagating delay. The
toggles already exist (`weather.params` in the configs); this is an
experiment to run and write up, not new code.

## 3. Enable the champion/challenger retraining pipeline

The AzureML retraining pipeline under [`aml/`](../aml/) is currently
code-as-documentation (intentionally not enabled for the demo). Turning it
on — monthly retrain, evaluate challenger vs. champion on a held-out
temporal split, promote on improvement — would make the deployed model
self-updating as new BTS data lands.

## 4. End-user serving API (Part 2)

A small **FastAPI** service over
[`src/inference/predictor.py`](../src/inference/predictor.py), reading
either `outputs/powerbi/fact_predictions.parquet` or the OneLake table.
Endpoints: `GET /flight-delay?airport=&horizon=`,
`GET /airport-traffic?airport=`. Add `fastapi`/`uvicorn` as an optional
extra in `pyproject.toml`. This complements the Power BI dashboard with a
programmatic interface.

## 5. Surface per-horizon results in the repo

The headline result (MAE ≈ 10 min averaged over horizons) lives in the
README; the per-horizon breakdown and the full baseline ladder (DenseNN →
LSTM → GCN → GAT → Transformer) live in the thesis. Populate
[`notebooks/03_results.ipynb`](../notebooks/03_results.ipynb) with the
comparison table, the MAE-vs-horizon curve, and a predicted-vs-actual
calibration plot so the evidence is reproducible directly from the repo.

## Smaller items

- Mixed-precision training (bf16 autocast + GradScaler) and LR warmup →
  cosine annealing (the scheduler hook exists; warmup is not wired).
- Per-horizon validation logging during training (today only the average
  val loss is logged).
- Optional MkDocs Material site published to GitHub Pages from `docs/`.
