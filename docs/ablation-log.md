# Ablation Log

> One bullet = one atomic change. Toggle individually to isolate its effect on test MAE per horizon.
>
> **Format:** `- [ ]` pending / reverted, `- [x]` applied as the current `default.yaml` baseline. Append `// note` with the latest measured delta from the most recent ablation run (e.g. `// h=8 MAE 12.4 → 11.1`).
>
> The numbered "Suggested ablation order" at the bottom is the recommended sequence to toggle items when the model is underperforming, ordered by expected effect size.

## Data scope
- [x] Expand loaded columns: schedule-side (CRSArrTime, CRSElapsedTime) + actuals (DepTime, ArrTime, WheelsOff/On, TaxiOut/In, AirTime, ActualElapsedTime, DepDel15, ArrDel15)  // applied in commit A; load_parquet now ignores absent columns
- [x] Add BTS cause columns: CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay  // applied in commit A; fill_delay_nulls fills NaN→0 (NaN ≡ ArrDelay<15)
- [x] Top airports: 30 → 70  // applied in commit B (config-only)
- [x] min_route_flights: 50 → 30  // applied in commit B (config-only)
- [x] Years: single year (sampled 10-30%) → 2018 + 2019, full  // applied in commit B (sample_frac=null)
- [x] Temporal split: month-based → date-based chronological cutoff  // applied in commit C; train < 2019-07-01 < val < 2019-10-01 ≤ test
- [x] Snapshot caching to disk under data/processed/snapshots/  // applied in commit G; key is SHA-256 over schema+config+airports+temporal range of the df

## Node features (historical, from completed flights in input window)
- [x] Replace `avg/std DepDelay outgoing` primacy with `avg/std/p75/p90 ArrDelay incoming` as headline  // applied in commit E (compute_node_features_rich, block A)
- [x] Add taxi-out / taxi-in averages  // applied in commit E (col 8, blends TaxiOut+TaxiIn when both exist)
- [x] Add cancellation & diversion counts  // applied in commit E (cols 11, 12)
- [x] Add 5 BTS delay-cause means (carrier/weather/NAS/security/late_aircraft)  // applied in commit E (cols 13–17)
- [x] Add lag features: arr_delay at t-1, t-3, t-6, t-24  // applied in commit E (via _build_history_lookups + _series_lookup)
- [x] Add rolling 6h mean & std of arr delay  // applied in commit E (same precompute, over the hourly series)
- [x] Add cyclic calendar: hour_sin/cos, dow_sin/cos, month_sin/cos  // applied in commit E (broadcast to all nodes)

## Node features (exogenous future, from published schedule for target window)
- [x] scheduled_arrivals_count per horizon  // applied in commit E (block G, col_offset+0)
- [x] scheduled_departures_count per horizon  // applied in commit E (block G, col_offset+1)
- [x] scheduled_arrivals_from_top10 per horizon (hub concentration)  // applied in commit E (block G, col_offset+2)
- [x] scheduled_mean_distance_in per horizon  // applied in commit E (block G, col_offset+3)
- [x] hour-of-day at target window cyclic encoding per horizon  // applied in commit E (col_offset+4, sin only to avoid inflating the dim)

## Weather features (exogenous, ERA5 reanalysis via Open-Meteo) — block H, optional
> Global activation: ``weather.enabled: true`` in the config. Each group can be
> toggled individually via ``weather.params``. The on-disk cache
> (``data/processed/weather/open_meteo/``) avoids hitting the network between runs.
>
> Baseline note: ``default.yaml`` ships with ``weather.enabled: false`` so the
> first network-free run still works. The **deployed champion**
> (`configs/weekend/seq2seq_gnn_large.yaml`) sets ``enabled: true`` with
> ``params: [wind, precip_cloud, category]``.
- [ ] Activate weather block entirely (`weather.enabled: true`)  // off in default.yaml; bumps feature dim 55 → 109 with H=5
- [ ] Wind block: mean wind + max gust (`weather.params: [wind]`)  // block H, cols 0–1 (hist) and 0–1 (fut)
- [ ] Precip + cloud block (`weather.params: [precip_cloud]`)  // block H, cols 2–3
- [ ] Weather category one-hot (`weather.params: [category]`)  // block H, cols 4–8 (clear/fog/rain/snow/thunder)
- [ ] Future-window weather as "perfect-forecast proxy" (observation at T+h)  // current baseline — switch later to noisy forecast for realism
- [ ] Historical-only weather (drop future block)  // conservative leakage stance for sensitivity check
- [ ] Swap Open-Meteo → Iowa State ASOS/METAR  // higher fidelity, requires a `metar` parser; future ablation

## Edge features (replace single static weight)
- [x] flight_count_norm (current weight, retained)  // applied in commit F (col 0, static)
- [x] mean_air_time_norm  // applied in commit F (col 1, static)
- [x] recent_route_delay (rolling 6h on this OD pair)  // applied in commit F (col 3, dynamic via route_recent_arr_delay)
- [x] scheduled_flights_next_h_norm (per-snapshot)  // applied in commit F (col 4, dynamic via route_sched_count)
- [x] mean_scheduled_distance_norm  // applied in commit F (col 2, static)

## Targets
- [x] Primary target: DepDelay → ArrDelay  // applied in commit D; grouped by Dest, target window keyed by arr_timestamp (CRSArrTime)
- [x] Add classification head: pct_arr_delayed_15  // applied in the W2 commit; channel 2 of the multi-task target, BCEWithLogits via MultiTaskLoss; bce_* metrics in evaluate_multi_horizon_*
- [x] Add auxiliary regression head: DepDelay (multi-task, λ=0.3)  // applied in the W2 commit; channel 1, weighted Huber identical to channel 0

## Horizons & loss
- [x] Horizons: [1,2,3,4,5] → [1,2,4,6,8]  // applied in commit H; change across 5 configs (default + 4 per-model)
- [x] Horizon weights: [5,4,3,2,1] → [3,3,4,5,5] (favor business horizons)  // applied in commit I; 3 multi-horizon configs (gat, spatiotemporal, seq2seq)
- [x] Regression loss: MSE → Huber (δ=10)  // applied in commit I; WeightedHuberLoss via the shared base _WeightedRegressionLossBase; WeightedMSELoss kept for ablation
- [x] WeightedMSE → multi-task Huber + 0.3·Huber_aux + 0.5·BCE  // applied in the W2 commit; new `multi_task` loss enabled via `training.loss`; weights via `training.multi_task.{main,aux,bce}_weight`; the factory builds models with `output_channels=3` automatically

## Architecture — applies per-model (toggle independently)
- [x] GATConv → GATv2Conv with edge_dim=5  // applied in commit F (multi_horizon_gat + spatiotemporal/seq2seq via GATEncoder)
- [ ] BatchNorm → LayerNorm
- [ ] Residual connections in GAT encoder (`x = x + gat(x)`)
- [ ] Activation ELU → GELU
- [ ] Hidden dim 64 → 128 (encoder)

### SpatioTemporalGNN-specific
- [ ] LSTM 1-layer → 2-layer
- [ ] Sinusoidal positional encoding on LSTM input

### Seq2SeqGNN redesign (replaces autoregressive decoder)
- [x] LSTM temporal encoder → 2-layer Transformer encoder (d_model=128, nhead=4)  // applied in the W3 commit; pre-norm, batch_first, GELU, dim_feedforward=2*d
- [x] Autoregressive LSTM decoder → horizon-query MultiheadAttention decoder  // applied in the W3 commit; cross-attn with residual + LayerNorm over the queries
- [x] 5 separate Linear heads → single MLP applied to 5 attended queries  // applied in the W3 commit; head = Linear→GELU→Dropout→Linear(1)
- [x] Drop teacher forcing entirely (no train/eval distribution shift)  // applied in the W3 commit; the forward ignores ``y`` entirely
- [x] Add learnable horizon-query embeddings Q ∈ R^{5×128}  // applied in the W3 commit; BERT-style init std=0.02

## Training infrastructure
- [x] CUDA 12.8 wheels + RTX 5070 (Blackwell, sm_120) enabled  // device-selection fix + docs/gpu-setup.md; select_device fails fast with `device: cuda`
- [ ] Mixed precision (bf16 autocast + GradScaler)
- [ ] LR warmup → cosine annealing
- [ ] Per-horizon validation logging (not just average)
- [ ] Gradient clip 1.0 (existing — confirm retained)

## Code hygiene (no accuracy impact, affects iteration speed)
- [x] Single model factory in src/models/factory.py  // applied in commit 2172eec; removes the detected divergence (evaluate.py did not propagate ``activation`` to DenseNN)
- [x] Single results reporter in src/evaluation/reporting.py  // applied in commit 2172eec; train.py + evaluate.py import log_test_results / log_multi_horizon_results
- [x] BaseTrainer + 2 subclasses (collapse 3 trainer classes)  // applied in W4.3; the train/val/early-stop/checkpoint/scheduler cycle lives in src/training/base_trainer.py; Trainer, GraphTrainer and SequenceGraphTrainer only contribute _forward_batch + _align_for_val + public-API wrappers
- [ ] Remove abandoned LSTM stub from configs and README
- [ ] Lightweight per-model dataclass for config validation

## Verification
- [x] Leakage unit test (tests/test_data/test_leakage.py)  // applied in commit G; sentinel ArrDelay=999 checks nodes+edges; revealed that window_df needed an arr_timestamp<T filter
- [x] Per-model smoke test on mock data (tests/test_integration.py)  // applied in commits 2172eec + 9abed7b; 5 models, synthetic data, exposed a GATEncoder bug with num_layers=1
- [ ] Populate notebooks/03_results.ipynb with comparison table & MAE-vs-horizon plot

## Suggested ablation order (when accuracy underwhelms)
1. Toggle target ArrDelay vs DepDelay — biggest expected effect on the baseline-vs-GNN gap
2. Toggle exogenous future features as a group — biggest expected effect on h=6/h=8 horizons
3. **Toggle weather block (H) as a group** — expected high impact on h=4/6/8; thunderstorms drive NAS holds
4. Toggle BTS cause columns + lags — affects all horizons
5. Toggle Seq2SeqGNN redesign as a whole vs old architecture (same features)
6. Toggle edge features (GATv2 with edge_dim vs without) — affects GAT/Seq2Seq only
7. Toggle weather sub-groups individually (wind / precip_cloud / category) — finer attribution within block H
8. Toggle multi-task auxiliary DepDelay head — small regularization effect
9. Toggle Huber vs MSE loss — small, helps the tail
