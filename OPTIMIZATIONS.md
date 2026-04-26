# OPTIMIZATIONS — Ablation Log

> One bullet = one atomic change. Toggle individually to isolate its effect on test MAE per horizon.
>
> **Format:** `- [ ]` pending / reverted, `- [x]` applied as current default. Append `// note` with the latest measured delta from the most recent ablation run (e.g. `// h=8 MAE 12.4 → 11.1`).
>
> The numbered "Suggested ablation order" at the bottom is the recommended sequence to toggle items when the model is underperforming, ordered by expected effect size.

## Data scope
- [x] Expand loaded columns: schedule-side (CRSArrTime, CRSElapsedTime) + actuals (DepTime, ArrTime, WheelsOff/On, TaxiOut/In, AirTime, ActualElapsedTime, DepDel15, ArrDel15)  // applied in commit A; load_parquet ahora ignora columnas ausentes
- [x] Add BTS cause columns: CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay  // applied in commit A; fill_delay_nulls rellena NaN→0 (NaN ≡ ArrDelay<15)
- [x] Top airports: 30 → 70  // applied in commit B (config-only)
- [x] min_route_flights: 50 → 30  // applied in commit B (config-only)
- [x] Years: single year (sampled 10-30%) → 2018 + 2019, full  // applied in commit B (sample_frac=null)
- [x] Temporal split: month-based → date-based chronological cutoff  // applied in commit C; train < 2019-07-01 < val < 2019-10-01 ≤ test
- [x] Snapshot caching to disk under data/processed/snapshots/  // applied in commit G; key SHA-256 sobre schema+config+airports+rango temporal del df

## Node features (historical, from completed flights in input window)
- [x] Replace `avg/std DepDelay outgoing` primacy with `avg/std/p75/p90 ArrDelay incoming` as headline  // applied in commit E (compute_node_features_rich, bloque A)
- [x] Add taxi-out / taxi-in averages  // applied in commit E (col 8, mezcla TaxiOut+TaxiIn cuando ambos existen)
- [x] Add cancellation & diversion counts  // applied in commit E (cols 11, 12)
- [x] Add 5 BTS delay-cause means (carrier/weather/NAS/security/late_aircraft)  // applied in commit E (cols 13–17)
- [x] Add lag features: arr_delay at t-1, t-3, t-6, t-24  // applied in commit E (vía _build_history_lookups + _series_lookup)
- [x] Add rolling 6h mean & std of arr delay  // applied in commit E (mismo precomputado, sobre la serie horaria)
- [x] Add cyclic calendar: hour_sin/cos, dow_sin/cos, month_sin/cos  // applied in commit E (broadcast a todos los nodos)

## Node features (exogenous future, from published schedule for target window)
- [x] scheduled_arrivals_count per horizon  // applied in commit E (bloque G, col_offset+0)
- [x] scheduled_departures_count per horizon  // applied in commit E (bloque G, col_offset+1)
- [x] scheduled_arrivals_from_top10 per horizon (hub concentration)  // applied in commit E (bloque G, col_offset+2)
- [x] scheduled_mean_distance_in per horizon  // applied in commit E (bloque G, col_offset+3)
- [x] hour-of-day at target window cyclic encoding per horizon  // applied in commit E (col_offset+4, solo sin para no inflar)

## Edge features (replace single static weight)
- [x] flight_count_norm (current weight, retained)  // applied in commit F (col 0 estática)
- [x] mean_air_time_norm  // applied in commit F (col 1 estática)
- [x] recent_route_delay (rolling 6h on this OD pair)  // applied in commit F (col 3 dinámica vía route_recent_arr_delay)
- [x] scheduled_flights_next_h_norm (per-snapshot)  // applied in commit F (col 4 dinámica vía route_sched_count)
- [x] mean_scheduled_distance_norm  // applied in commit F (col 2 estática)

## Targets
- [x] Primary target: DepDelay → ArrDelay  // applied in commit D; agrupado por Dest, ventana del target keada por arr_timestamp (CRSArrTime)
- [ ] Add classification head: pct_arr_delayed_15
- [ ] Add auxiliary regression head: DepDelay (multi-task, λ=0.3)

## Horizons & loss
- [x] Horizons: [1,2,3,4,5] → [1,2,4,6,8]  // applied in commit H; cambio en 5 configs (default + 4 per-model)
- [x] Horizon weights: [5,4,3,2,1] → [3,3,4,5,5] (favor business horizons)  // applied in commit I; 3 configs multi-horizonte (gat, spatiotemporal, seq2seq)
- [x] Regression loss: MSE → Huber (δ=10)  // applied in commit I; WeightedHuberLoss via base compartida _WeightedRegressionLossBase; WeightedMSELoss mantenido para ablación
- [ ] WeightedMSE → multi-task Huber + 0.3·Huber_aux + 0.5·BCE

## Architecture — applies per-model (toggle independently)
- [x] GATConv → GATv2Conv with edge_dim=5  // applied in commit F (multi_horizon_gat + spatiotemporal/seq2seq vía GATEncoder)
- [ ] BatchNorm → LayerNorm
- [ ] Residual connections in GAT encoder (`x = x + gat(x)`)
- [ ] Activation ELU → GELU
- [ ] Hidden dim 64 → 128 (encoder)

### SpatioTemporalGNN-specific
- [ ] LSTM 1-layer → 2-layer
- [ ] Sinusoidal positional encoding on LSTM input

### Seq2SeqGNN redesign (replaces autoregressive decoder)
- [x] LSTM temporal encoder → 2-layer Transformer encoder (d_model=128, nhead=4)  // applied in W3 commit; pre-norm, batch_first, GELU, dim_feedforward=2*d
- [x] Autoregressive LSTM decoder → horizon-query MultiheadAttention decoder  // applied in W3 commit; cross-attn con residual + LayerNorm sobre las queries
- [x] 5 separate Linear heads → single MLP applied to 5 attended queries  // applied in W3 commit; head = Linear→GELU→Dropout→Linear(1)
- [x] Drop teacher forcing entirely (no train/eval distribution shift)  // applied in W3 commit; el forward ignora ``y`` por completo
- [x] Add learnable horizon-query embeddings Q ∈ R^{5×128}  // applied in W3 commit; init BERT-style std=0.02

## Training infrastructure
- [ ] CUDA 12.8 wheels + RTX 5070 enabled
- [ ] Mixed precision (bf16 autocast + GradScaler)
- [ ] LR warmup → cosine annealing
- [ ] Per-horizon validation logging (not just average)
- [ ] Gradient clip 1.0 (existing — confirm retained)

## Code hygiene (no accuracy impact, affects iteration speed)
- [x] Single model factory in src/models/factory.py  // applied in commit 2172eec; elimina divergencia detectada (evaluate.py no propagaba ``activation`` a DenseNN)
- [x] Single results reporter in src/evaluation/reporting.py  // applied in commit 2172eec; train.py + evaluate.py importan log_test_results / log_multi_horizon_results
- [ ] BaseTrainer + 2 subclasses (collapse 3 trainer classes)
- [ ] Remove abandoned LSTM stub from configs and README
- [ ] Lightweight per-model dataclass for config validation

## Verification
- [x] Leakage unit test (tests/test_data/test_leakage.py)  // applied in commit G; sentinel ArrDelay=999 verifica nodos+edges; reveló que window_df necesitaba filtro arr_timestamp<T
- [x] Per-model smoke test on mock data (tests/test_integration.py)  // applied in commits 2172eec + 9abed7b; 5 modelos, datos sintéticos, expuso bug de GATEncoder con num_layers=1
- [ ] Populated notebooks/03_results.ipynb with comparison table & MAE-vs-horizon plot

## Suggested ablation order (when accuracy underwhelms)
1. Toggle target ArrDelay vs DepDelay — biggest expected effect on baseline-vs-GNN gap
2. Toggle exogenous future features as a group — biggest expected effect on h=6/h=8 horizons
3. Toggle BTS cause columns + lags — affects all horizons
4. Toggle Seq2SeqGNN redesign as a whole vs old architecture (same features)
5. Toggle edge features (GATv2 with edge_dim vs without) — affects GAT/Seq2Seq only
6. Toggle multi-task auxiliary DepDelay head — small regularization effect
7. Toggle Huber vs MSE loss — small, helps tail
