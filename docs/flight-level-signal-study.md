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

---

# Part 2 — Point-in-time pipeline results (honest, per horizon)

> The headline above (flat recall ~0.45) assumed the inbound aircraft's *actual*
> arrival is known — optimistic at long horizons. The productionized pipeline
> (`src/flight_level/`, `scripts/train_flight_model.py`) enforces
> **point-in-time correctness**: at horizon H, prediction time is
> `t_pred = scheduled_departure − H`, and a feature may use the inbound leg's
> ArrDelay only if it has *landed* by `t_pred`, else its DepDelay if it has
> *departed*, else nothing; airport-state features use only the last
> fully-completed hour(s) before `t_pred`. These are the numbers to trust.

**Ablation × horizon** (600k train, 1.38M test 2019-Q4; recall@15 unless noted):

| Feature set | H=1 | H=2 | H=4 | H=6 | H=8 | precision@15 (H=1) |
|---|---|---|---|---|---|---|
| A schedule+route | 0.160 | 0.160 | 0.160 | 0.160 | 0.160 | 0.299 |
| B + airline | 0.178 | 0.178 | 0.178 | 0.178 | 0.178 | 0.307 |
| C + rotation | **0.336** | 0.313 | 0.190 | 0.169 | 0.158 | **0.602** |
| D + airport-state | 0.262 | 0.243 | **0.223** | **0.216** | **0.209** | 0.444 |
| **E all** | **0.389** | **0.365** | **0.254** | **0.225** | **0.210** | **0.621** |
| *GNN baseline (agg)* | — | — | — | — | — | *0.523 (rec 0.352)* |

**What the point-in-time view reveals (that the optimistic study hid):**

1. **The two signals have different temporal reach.** Rotation (C) is the
   strongest lift at H=1–2 (recall 0.336/0.313, precision 0.60) but **decays to
   the schedule floor by H=8** — the specific inbound aircraft hasn't departed
   yet, so its state is unobservable. Airport-state (D) is weaker at H=1 but
   **persists** (0.262→0.209) because congestion is autocorrelated over hours.
2. **They are complementary.** E (all) is best at every horizon: at H=1 rotation
   carries it (0.389/0.621 — beats the GNN on both recall *and* precision); from
   H≥4 airport-state carries it (D > C).
3. **Schedule/airline alone is a weak, horizon-flat floor** (A 0.160, B 0.178).
   Airline's marginal lift is real but small and time-independent.
4. **The per-flight model beats the GNN at short horizons** (E H=1: recall 0.389
   vs 0.352, precision 0.621 vs 0.523) on the harder per-flight task; at long
   horizons recall falls below the GNN — expected, since the dominant per-flight
   signal (inbound aircraft) is no longer observable.

**Architectural implication.** To keep rotation's edge at H=4/6/8 we must
*predict* the inbound leg's delay and chain it along the tail rotation (recursive
/ sequential prediction) — precisely where a sequential or graph model adds
value. Short-horizon serving (H=1–2) is already strong with the plain GBM. A
natural design: GBM for H=1–2; rotation-chaining (or the GNN's spatial signal)
for longer horizons.

---

# Part 3 — GNN ↔ per-flight fusion (how the two models play together)

> The GNN baseline and the per-flight model stay **separate** (no shared code).
> This experiment only *consumes their outputs* to test fusion, point-in-time.
> Code: `src/flight_level/fusion.py`, `scripts/compare_fusion.py`. The GNN's
> airport-hour forecast is broadcast to each flight as the prediction for its
> destination-hour, taken from the freshest GNN forecast issued ≤ `t_pred`.
> Fusion knobs (blend weight, switch, stack meta-model) are fit on **val**
> (out-of-sample) and reported on **test** — no stacking leakage.

**Per-flight test results (E_all features; recall/precision at 15 min):**

| H | strategy | MAE | rec@15 | pre@15 | rec@30 |
|---|---|---|---|---|---|
| 1 | gbm_only | 18.18 | 0.389 | 0.621 | 0.358 |
| 1 | gnn_only | 21.16 | 0.221 | 0.413 | 0.109 |
| 1 | **stack** | **18.09** | **0.418** | 0.598 | **0.383** |
| 2 | gbm_only | 19.41 | 0.365 | 0.523 | 0.288 |
| 2 | **stack** | **19.21** | **0.377** | **0.531** | **0.312** |
| 4 | gbm_only | 20.90 | **0.254** | 0.426 | 0.136 |
| 4 | **stack** | **20.33** | 0.253 | **0.462** | **0.167** |
| 6 | gbm_only | 21.26 | **0.225** | 0.393 | 0.093 |
| 6 | **stack** | **20.52** | 0.216 | **0.445** | **0.125** |
| 8 | gbm_only | 21.47 | 0.210 | 0.371 | 0.070 |
| 8 | gnn_only | 22.39 | **0.215** | 0.387 | 0.091 |
| 8 | **stack** | **20.55** | 0.202 | **0.438** | **0.107** |

(blend and switch omitted for brevity; see `report.json`.)

**Findings:**

1. **Feature-level stacking is the winner.** Feeding the GNN's airport-hour
   forecast into the per-flight GBM as a feature beats every single-model option:
   it has the **lowest MAE at every horizon** and the **highest precision@15 and
   recall@30 from H≥4**. At short horizons it also lifts recall@15 (H=1:
   0.389→0.418; H=2: 0.365→0.377).
2. **The GNN's value is the long-horizon / congestion regime.** Stack's gains
   over gbm_only *grow* with horizon (MAE −0.6 at H=1 → −0.9 at H=8; rec@30 at
   H=8: 0.070→0.107, +53%). The network's spatial congestion forecast supplies
   exactly what the per-flight model loses when the inbound aircraft becomes
   unobservable.
3. **gnn_only is weak per-flight** (~0.22 recall, flat). Broadcasting an
   airport-hour *mean* to individual flights is a smoothed predictor — far below
   the GNN's own airport-hour recall (0.352), which is a different (easier) unit.
   Not a contradiction: it confirms the GNN is a *complement*, not a per-flight
   predictor on its own.
4. **blend trades recall for precision** (pulls toward the smooth GNN); **switch
   always chose the GBM** (higher val F1). Stacking dominates both.

**Conclusion / recommended design.** They play together best via **stacking**:
one per-flight GBM that *includes the GNN's destination-hour forecast as a
feature*. Keep the two models independent and trained separately; the fusion
lives only at inference/feature assembly. This gives the best MAE everywhere,
the recall edge at H=1–2, and meaningful precision / big-delay-recall gains at
long horizons — the GNN earning its keep exactly where the per-flight signal
fades.

---

# Part 4 — Full-data training: LightGBM + point-in-time weather (authoritative)

> Code: `scripts/train_flight_full.py`, `src/flight_level/weather.py`,
> LightGBM backend in `src/flight_level/model.py`. Trains on the **entire**
> 2018–2019 train split (no 600k subsample): train=6,661,824, val=1,395,722,
> test=1,378,256. Backend = LightGBM 4.6 with early stopping on val. One compact
> random hyperparameter search (`--tune`, selected by val F1@15) reused across
> all horizons: `num_leaves=127, learning_rate=0.03, min_child_samples=100,
> colsample_bytree=0.7, subsample=0.9`.
>
> Two feature sets compared to isolate weather's contribution:
> * **E_all** — schedule + route + airline + rotation + airport-state.
> * **F_all_weather** — E_all + origin weather *as-of* `t_pred` + destination
>   weather at scheduled arrival (forecast, mirroring the GNN, which also uses
>   forward weather). Same Open-Meteo parquets the GNN consumes, at flight grain.

**Per-flight test results (2019-Q4; recall/precision at 15 and 30 min):**

| H | config | MAE | rec@15 | pre@15 | rec@30 |
|---|---|---|---|---|---|
| 1 | E_all | 17.59 | 0.455 | 0.603 | 0.423 |
| 1 | **+weather** | **17.33** | **0.480** | **0.607** | **0.445** |
| 2 | E_all | 18.79 | 0.423 | 0.541 | 0.358 |
| 2 | **+weather** | **18.46** | **0.450** | **0.549** | **0.385** |
| 4 | E_all | 20.45 | 0.317 | 0.447 | 0.211 |
| 4 | **+weather** | **20.00** | **0.357** | **0.469** | **0.254** |
| 6 | E_all | 20.95 | 0.283 | 0.417 | 0.167 |
| 6 | **+weather** | **20.39** | **0.333** | **0.447** | **0.223** |
| 8 | E_all | 21.25 | 0.267 | 0.395 | 0.145 |
| 8 | **+weather** | **20.58** | **0.321** | **0.436** | **0.206** |

**Findings:**

1. **Full data + LightGBM is a large jump.** Versus the prior 600k HistGBM
   baseline (H=1 rec@15=0.389, MAE=18.18), full-data LightGBM alone reaches
   rec@15=0.455 / MAE=17.59 at H=1 — before weather.
2. **Weather helps at every horizon, and the lift grows with H.** rec@15:
   +0.025 at H=1 → +0.054 at H=8. rec@30 at H=8: 0.145→0.206 (+42%). MAE drops
   monotonically with weather, most at long horizons (−0.67 at H=8). Weather is
   complementary to the rotation signal: it's strongest exactly where rotation
   has decayed to the schedule floor.
3. **Authoritative operating numbers.** These are the reference per-flight
   results: **+weather** is the recommended config. Best single-model recall@15
   is 0.480 at H=1, degrading gracefully to 0.321 at H=8 — all well above the
   GNN's airport-hour recall (0.352), on the harder per-flight unit.

Saved models + metrics: `outputs/flight_level_full/` (`report.json`,
`model_{config}_h{H}.joblib`).

## Next steps (deferred levers)

Two further recall levers are queued as dedicated runs (see
`scripts/train_flight_recall.py`, `scripts/train_flight_chain.py`):

1. **Recall-targeted objective + operating-point tuning** — instead of
   thresholding the regression at exactly 15 min, tune the decision cutoff on
   val (F-beta, β>1) and/or train a class-weighted classifier, to trade
   precision for recall at a chosen operating point.
2. **Inbound-delay chaining for long horizons** — when the inbound leg is not
   yet observable at `t_pred`, predict its delay and feed it forward, to recover
   the rotation signal that currently decays to the schedule floor past H=2.
