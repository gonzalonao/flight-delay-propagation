# Weekend Training Queue

`scripts/run_weekend.ps1` orchestrates 6 unattended training runs and hibernates the PC when finished. Built for an idle weekend.

## What runs

| # | Config | Model | Capacity |
|---|---|---|---|
| 1 | `configs/weekend/multi_horizon_gat_small.yaml`  | `multi_horizon_gat`  | hidden=64, heads=4, layers=3 |
| 2 | `configs/weekend/multi_horizon_gat_large.yaml`  | `multi_horizon_gat`  | hidden=128, heads=8, layers=3 |
| 3 | `configs/weekend/spatiotemporal_gnn_small.yaml` | `spatiotemporal_gnn` | gnn=64, lstm=128, heads=4 |
| 4 | `configs/weekend/spatiotemporal_gnn_large.yaml` | `spatiotemporal_gnn` | gnn=128, lstm=256, heads=8 |
| 5 | `configs/weekend/seq2seq_gnn_small.yaml`        | `seq2seq_gnn`        | hidden=128, heads=4 |
| 6 | `configs/weekend/seq2seq_gnn_large.yaml`        | `seq2seq_gnn`        | hidden=256, heads=8 |

All 6 share:
- `weather.enabled: true` (Block H, ERA5 via Open-Meteo)
- `graph.temporal_window_hours: 1`
- `loss: multi_task` (Huber + auxiliary DepDelay + BCE)
- `years: [2018, 2019]`, full data, seed 42

## Before you leave

1. **Activate the venv** in the same PowerShell window you'll launch from (otherwise `python` won't resolve to the project Python).
2. **Smoke-run** with a 1-epoch override. Create `configs/local.yaml` (gitignored — `load_config` merges it as the final override layer) containing:
   ```yaml
   training:
     epochs: 1
     patience: 1
   ```
   Then:
   ```powershell
   python scripts\train.py --config configs\weekend\multi_horizon_gat_small.yaml
   ```
   This validates that weather download/cache, snapshot-cache rebuild with `temporal_window_hours=1`, checkpoint write, and overall pipeline work end-to-end. **Delete `configs/local.yaml` afterwards** or your weekend run will only do 1 epoch.

3. **Verify hibernation is enabled** (Administrator PowerShell, one-time):
   ```powershell
   powercfg /a
   ```
   If "Hibernation has not been enabled":
   ```powershell
   powercfg /h on
   ```

4. **Dry-run** to confirm the script parses and the run directory is writable:
   ```powershell
   .\scripts\run_weekend.ps1 -DryRun
   ```
   You should see the 6 commands printed and the run/log directories created.

   > **Execution policy**: if PowerShell refuses with `cannot be loaded ... not digitally signed`, launch via:
   > ```powershell
   > powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_weekend.ps1 -DryRun
   > ```
   > Use the same `powershell.exe -ExecutionPolicy Bypass -File ...` pattern for the real run too. (Or, one-time per user: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and answer Yes.)

5. **Launch the real run**:
   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_weekend.ps1
   ```
   Walk away.

## Outputs

For a run with `RUN_ID = 20260516-180000`:

```
outputs/runs/20260516-180000/
├── multi_horizon_gat_small.pt
├── multi_horizon_gat_large.pt
├── spatiotemporal_gnn_small.pt
├── spatiotemporal_gnn_large.pt
├── seq2seq_gnn_small.pt
├── seq2seq_gnn_large.pt
├── summary.txt          # human-readable
└── summary.json         # pandas-friendly for 03_results.ipynb

logs/20260516-180000/
├── preflight.log
├── _runner.log
├── multi_horizon_gat_small_train.log
├── multi_horizon_gat_small_eval.log
└── ... (one train+eval pair per run)
```

Each `*_eval.log` contains the per-horizon MAE/RMSE/MAPE/R² block printed by `evaluate.py`.

## Failure handling

A failure in one run is logged and the queue continues with the next. The summary records the status of each run as one of:

- `PASS` — train + evaluate both exited 0
- `FAIL_TRAIN` — train.py exited non-zero (see `*_train.log`)
- `FAIL_NO_CKPT` — train.py succeeded but `outputs/best_<model>.pt` is missing
- `FAIL_EVAL` — evaluate.py exited non-zero (see `*_eval.log`)
- `FAIL_EXCEPTION` — PowerShell-level exception (rare)

## Aborting

There is no `shutdown /a` for hibernate — it's immediate. If you need to intervene before hibernate fires:

- **During training**: `Ctrl-C` in the PowerShell window kills the current Python process. The runner's `try/catch` will record a `FAIL_EXCEPTION` and move to the next run.
- **Before hibernate**: kill the PowerShell process via Task Manager. The summary is written *before* hibernate is called, so it's already on disk.

To stop the queue without hibernating, set the env var before launching:

```powershell
$env:NO_HIBERNATE = 1
.\scripts\run_weekend.ps1
```

Or pass the switch:

```powershell
.\scripts\run_weekend.ps1 -NoHibernate
```

## Monday verification

1. Resume from hibernate.
2. Open `outputs/runs/<latest>/summary.txt` for a quick PASS/FAIL overview.
3. Open `outputs/runs/<latest>/summary.json` from `notebooks/03_results.ipynb` to build a DataFrame (`pd.read_json(...)`) for cross-run comparison.
4. Spot-check `logs/<latest>/<name>_eval.log` for the per-horizon MAE block of the best-looking run.

## Runtime expectations

- **First run** is the slowest: cold-fetches Open-Meteo weather for 70 airports × 2 years and rebuilds the snapshot cache for the new `temporal_window_hours=1` key.
- **Subsequent 5 runs** reuse the snapshot + weather cache, so they're GPU-bound.
- Rough estimate per run on RTX 5070: 1.5–4 h with `epochs=100` + cosine + patience 20–25.
- **Total: 12–24 h.** Fits comfortably in a weekend.

## Caveats

- The script writes to `outputs/best_<model_name>.pt` during training (existing trainer convention), then *moves* the file into the run directory. This means a stray `best_multi_horizon_gat.pt` left over from a previous interactive session may be moved away by the first run — that's intentional but worth knowing.
- If a `large` variant OOMs on your GPU, the queue continues to the next run; you'll see `FAIL_TRAIN` in the summary.
- The script is Windows-PowerShell-only (uses `shutdown.exe /h`, `Start-Process`, etc.). For Linux/macOS, port to a shell script with `tee` and `systemctl suspend-then-hibernate`.
