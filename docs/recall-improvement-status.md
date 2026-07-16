# Recall Improvement — Status & Continuation

> Paused doc. Captures everything needed to resume the recall work on
> `seq2seq_gnn` later without re-deriving context. Work lives on branch
> **`feat/improve-recall`** (off the clean base `feat/phase4-powerbi` @ `5d9aa58`),
> **not** `exercise/leak-rolling` (separate exercise — do not touch).

## Goal

Raise **both** recall families the pipeline reports, per-horizon and averaged:
- **Derived recall** — ArrDelay regression thresholded at 15 min
  (`src/evaluation/metrics.py`).
- **BCE-head recall** — `bce_recall` from the `pct_arr_delayed_15` classifier head.

All levers must stay legitimate: **no edits to `src/data/graph_builder.py`**
(planted leak surface). Work is confined to config + losses + metrics + trainer.

## Reference checkpoint & baseline

- Clean checkpoint: `outputs/runs/20260514-213242/seq2seq_gnn_large.pt`
  (epoch 97/100), produced by `configs/weekend/seq2seq_gnn_large.yaml`
  (hidden_dim **256**, num_heads **8** — the "large" variant, NOT base 128/4).
- Snapshot cache lives at `data/processed/snapshots` (`graph.cache_dir`), so the
  ~70-min build happens once and is reused; features never change across recall
  levers.

Baseline (test split, from logs `20260514-213242`):

| | Derived (regr@15) | BCE head (@0.5) |
|---|---|---|
| Precision | 0.523 | 0.685 |
| Recall | **0.352** | **0.112** |
| F1 | 0.420 | 0.193 |

Regression avg: MAE 10.02, RMSE 18.26, R² 0.25. Both heads are precision-biased
→ classic class-imbalance + MSE-shrinkage signature.

### Why epoch count / LR are NOT recall levers

val_loss fell monotonically to ep97, no overfitting, cosine LR fully annealed by
ep100 — training is converged. More epochs / different LR only move along the
**MSE frontier**, and MSE-optimal = conservative = *lower* recall. Recall is
bottlenecked by: (1) the objective rewards hedging, (2) checkpoint selected by
val MSE, (3) `bce_pos_weight=null`, (4) untuned thresholds.
Efficiency note: train experiments at **~75 epochs** (cosine T_max=75) to save
~4h/run; keep 100 only for the final model.

## What's already built (committed as `3de4112`, NOT pushed)

Infrastructure + the free Phase-1 sweep. Defaults preserve baseline numbers
exactly (pred=15, bce=0.5, focal gamma=None).

- **`src/evaluation/metrics.py`** — decoupled the regression *decision* threshold
  (`pred_threshold`, tunable) from the *label* threshold (`label_threshold`,
  fixed business truth = 15). Added sweep/pick helpers
  (`compute_classification_sweep`, `pick_pred_threshold`, `compute_bce_sweep`,
  `pick_bce_threshold`) and a `collect_sequence_predictions` forward-once
  collector. Threaded `pred_threshold`/`bce_threshold` through
  `compute_unified_metrics` and all four `evaluate_*` functions.
- **`scripts/sweep_thresholds.py`** (new) — Phase-1 workhorse. One forward pass
  per split (val + test), then in-memory grid sweep. Picks `t*`/`b*` and the
  OR-ensemble pair on **val** (max recall s.t. precision ≥ floor), reports on
  **test**. ASCII-only logging (Windows cp1252 console).
- **`scripts/train.py`** — reads `evaluation.pred_threshold` /
  `evaluation.bce_threshold`; passes focal params to `MultiTaskLoss`.
- **`scripts/evaluate.py`** — `--pred-threshold` / `--bce-threshold` CLI overrides.
- **`src/training/losses.py`** — optional focal BCE
  (`bce_focal_gamma`, `bce_focal_alpha`); `None` ⇒ plain BCE.
- **`configs/recall/default.yaml`** — self-contained merge base (= large config +
  `pred_threshold: null`, `bce_threshold: 0.5`).
- **`configs/recall/reweight.yaml`** — Phase-2 deltas.
- **`configs/recall/focal.yaml`** — Phase-3 deltas.

Tests: 18/18 loss+metrics pass. (The 8 `test_weather.py` failures were a
pre-existing missing-`import numpy` bug — fix now in the working tree, see git
status below.)

## Phase 1 results (free, no retrain) — `outputs/phase1_sweep.json`, `logs/phase1_sweep.log`

Operating points picked on val, precision floor 0.50, reported on test:

| Variant | Recall | Precision | F1 | Δ recall |
|---|---|---|---|---|
| Baseline regr@15 | 0.3584 | 0.5174 | 0.4224 | — |
| **E1a** regr pred 15→13 | **0.4070** | 0.4802 | 0.4398 | **+4.9 pp** |
| E1b BCE 0.5→0.45 | 0.1456 | 0.6038 | 0.2339 | (bce head) |
| E1c OR-ensemble (regr≥13 OR σ(bce)≥0.35) | 0.4086 | 0.4796 | 0.4404 | +5.0 pp |

**Conclusions:** lowering the regression cutoff to ~13 recovers ~+5 pp recall with
precision still above the 0.85×baseline floor and F1 slightly up; MAE unchanged.
The BCE head stays collapsed (`bce_pos_weight=null`), so the OR-ensemble adds
essentially nothing over E1a. **The free threshold lever is exhausted at ~+5 pp.**
Bigger gains require a retrain (Phase 2).

## Next steps (deferred — require explicit go-ahead)

1. **(Optional, free)** Re-run the sweep at a lower precision floor to trade more
   precision for recall:
   ```
   uv run python scripts/sweep_thresholds.py \
     --checkpoint outputs/runs/20260514-213242/seq2seq_gnn_large.pt \
     --config configs/weekend/seq2seq_gnn_large.yaml \
     --min-precision 0.40 --out outputs/phase1_sweep_p40.json
   ```
2. **Phase 2 — reweight retrain (~16h).** Configs written, not launched:
   ```
   uv run python scripts/train.py --config configs/recall/reweight.yaml
   ```
   `delay_weight 2.0→3.5`, `bce_weight 0.5→1.0`, `bce_pos_weight null→3`. After
   training, re-run the Phase-1 sweep on the new checkpoint; if it wins, freeze
   `evaluation.pred_threshold` / `bce_threshold` in the config.
3. **Phase 3 — only if Phase 2 underdelivers.** Focal BCE
   (`configs/recall/focal.yaml`), plus the **deferred, NOT-built** infra:
   checkpoint selection by val F1/recall@floor in
   `src/training/graph_trainer.py` (currently selects by val MSE, which favors
   the low-recall hedger); sweep `bce_pos_weight ∈ {2,4,6}`.

## Success criteria (per experiment)

- `test.recall` (or `bce_recall`) **≥ +5 pp absolute** over baseline.
- `test.precision` **≥ 0.85 × baseline** (or above the chosen floor).
- `test.f1` down by **≤ 2 pp**; `test.mae` up by **≤ ~10%**.

## Guardrails

- Confirm branch ≠ `exercise/*` before every run (`git rev-parse --abbrev-ref HEAD`).
- `git diff src/data/graph_builder.py` must be empty.
- train↔val recall gap < ~15 pp (sudden widening = leak amplification → revert).
- Run sweeps/eval with `PYTHONIOENCODING=utf-8` (Windows cp1252 console can't
  encode →/≥/box-drawing chars).
- Keep `EXERCISE_LEAKS.md` untracked.
