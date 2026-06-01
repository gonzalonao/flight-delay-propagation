# Flight-Level Signal Study — quantifying what airport-hour aggregation throws away

> **Goal.** Before changing the model architecture, *measure* how much predictive
> signal is lost when flights are aggregated to per-(airport, hour) nodes (as the
> GNN does). Hypothesis (raised in review): per-flight information — especially
> **airline** and **aircraft-rotation state** (a late inbound aircraft forces a
> late departure) — carries large signal that averaging destroys.
>
> Reproduce: `uv run python scripts/explore_flight_level.py` (full data) or
> `--sample-train 400000` (fast). Report JSON in `outputs/research/flight_level*/`.

## Method

- **Data / split — matched to the GNN** so results are comparable: 2018–2019,
  top-70 airports (both endpoints), temporal split `train < 2019-07-01`,
  `test ≥ 2019-10-01`. ~6.66 M training flights, ~1.38 M test flights.
- **Model.** `HistGradientBoostingRegressor` (sklearn, native categorical +
  NaN), regressing `ArrDelay` in minutes. Same model across all feature sets.
- **Ablation (nested feature sets)** to isolate the *marginal* value of each
  signal block:
  - **A** schedule + route (`Distance`, `CRSElapsedTime`, `dep_hour`, `Month`,
    `DayOfWeek`, `Origin`, `Dest`)
  - **B** A + airline (`Operating_Airline`)
  - **C** A + rotation (`prev_arr_delay`, `sched_turnaround_min`,
    `turnaround_slack`, `is_first_leg`, `prev_delayed15`, `rotation_continuity`)
  - **D** all
- **Multi-threshold readout.** Recall/precision/F1 derived from the regression at
  ≥15/30/45/60 min (the agreed "regression → threshold at inference" approach).
- **No-leakage discipline.** Only pre-departure/scheduled fields + the *prior*
  leg's outcome. Same-flight post-departure columns (`DepDelay`, `DepTime`,
  `WheelsOff`, `TaxiOut`, `AirTime`, `ActualElapsedTime`, `ArrTime`, …) are
  **excluded** — they would leak the target.

### Rotation features (the key construction)
For each `Tail_Number`, flights are ordered by scheduled departure; the **prior
leg of the same aircraft** gives:
- `prev_arr_delay` — inbound leg's arrival delay (min).
- `sched_turnaround_min` — scheduled minutes between inbound arrival and this
  departure.
- `turnaround_slack` = `sched_turnaround_min − prev_arr_delay` (negative ⇒ the
  inbound delay eats the buffer ⇒ likely late departure).
- `is_first_leg`, `prev_delayed15`, `rotation_continuity` (prior dest == current
  origin; 0 ⇒ data gap/repositioning → rotation features invalidated).

## Results (full data, 6.66 M train / 1.38 M test)

| Feature set | MAE (min) | RMSE | recall@15 | recall@30 | recall@60 |
|---|---|---|---|---|---|
| **A** schedule + route | 22.18 | 44.60 | 0.156 | 0.036 | 0.001 |
| **B** + airline | 22.15 | 44.52 | 0.180 | 0.042 | 0.003 |
| **C** + rotation | 17.50 | 38.42 | 0.453 | 0.412 | 0.359 |
| **D** all | **17.41** | **38.12** | **0.455** | **0.422** | **0.370** |
| *GNN baseline (airport-hour agg)* | *~10.0* | *—* | *0.352* | *—* | *—* |

**Permutation importance (model D, MAE increase when shuffled):**

| Feature | ΔMAE |
|---|---|
| `turnaround_slack` | **+5.72** |
| `dep_hour` | +1.31 |
| `Operating_Airline` | +1.13 |
| `Origin` | +1.08 |
| `prev_arr_delay` | +0.85 |
| `CRSElapsedTime` | +0.66 |
| `Dest` | +0.55 |
| `Distance` | +0.55 |
| `sched_turnaround_min` | +0.49 |
| `rotation_continuity` | +0.20 |

## Findings

1. **Aircraft rotation is the dominant lost signal.** Adding rotation (A→C)
   **nearly triples recall@15 (0.156 → 0.453)** and cuts MAE ~21%.
   `turnaround_slack` is the single most important feature by ~5× — this is the
   "late aircraft" propagation mechanism (BTS attributes ~30–40% of delay minutes
   to it), and airport-hour averaging erases it.
2. **Airline identity matters, modestly.** A→B lifts recall@15 0.156 → 0.180;
   `Operating_Airline` ranks #3 in importance on full data. Real, but the
   *specific aircraft's* state dwarfs the carrier's average profile.
3. **Big delays are only recoverable with rotation.** Schedule-only is blind to
   ≥30/≥60-min delays (recall 0.036 / 0.001); rotation recovers them
   (0.422 / 0.370). For the multi-threshold goal this is the whole story.
4. **Per-flight recall@15 (0.455) already beats the GNN (0.352)** on the harder,
   more useful per-flight task — with an off-the-shelf GBM and no tuning.

## Caveats (read honestly)

- **Do not compare per-flight MAE (17.4) to the GNN's ~10.** The GNN predicts a
  *smoothed airport-hour mean*; averaging cancels noise so its MAE is
  mechanically lower. Per-flight prediction is intrinsically harder (RMSE ~38
  from fat tails). The fair, comparable metric is **delay-classification
  recall/precision**, where the per-flight model already wins.
- **`prev_arr_delay` uses the inbound aircraft's *actual* arrival.** Legitimate
  and observable in real-time ops (the inbound flight has landed before the
  outbound pushes back), but **optimistic for long horizons** — production would
  substitute the inbound leg's *predicted* delay, chaining predictions along the
  rotation. Expect some recall give-back at h=4/6/8.
- This is a single GBM, untuned; numbers are a lower bound on achievable signal.

## Recommendation / next steps

The evidence supports the chosen direction (per-flight target + rotation
features). Proposed path:

1. **Productionize the rotation feature builder** as a reusable, leak-safe module
   (a `Tail_Number`-keyed rotation chain), separate from the off-limits
   `src/data/graph_builder.py`.
2. **Train a per-flight delay model** (start with HistGBM; LightGBM if we add the
   dep) as the primary "will my flight be late" predictor, with the
   regression→multi-threshold readout. This also feeds the Part-2 product.
3. **Feed rotation-derived aggregates into the GNN** as node features (e.g.
   per-airport-hour mean/max inbound-delay pressure, tight-turnaround counts) to
   test whether the network model also lifts — cheap, keeps the spatial model.
4. **Horizon-honest evaluation**: re-run with the inbound delay replaced by a
   predicted/observed-so-far value to measure the realistic (non-optimistic)
   recall at each horizon.
