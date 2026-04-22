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
- [ ] Temporal split: month-based → date-based chronological cutoff
- [ ] Snapshot caching to disk under data/processed/snapshots/

## Node features (historical, from completed flights in input window)
- [ ] Replace `avg/std DepDelay outgoing` primacy with `avg/std/p75/p90 ArrDelay incoming` as headline
- [ ] Add taxi-out / taxi-in averages
- [ ] Add cancellation & diversion counts
- [ ] Add 5 BTS delay-cause means (carrier/weather/NAS/security/late_aircraft)
- [ ] Add lag features: arr_delay at t-1, t-3, t-6, t-24
- [ ] Add rolling 6h mean & std of arr delay
- [ ] Add cyclic calendar: hour_sin/cos, dow_sin/cos, month_sin/cos

## Node features (exogenous future, from published schedule for target window)
- [ ] scheduled_arrivals_count per horizon
- [ ] scheduled_departures_count per horizon
- [ ] scheduled_arrivals_from_top10 per horizon (hub concentration)
- [ ] scheduled_mean_distance_in per horizon
- [ ] hour-of-day at target window cyclic encoding per horizon

## Edge features (replace single static weight)
- [ ] flight_count_norm (current weight, retained)
- [ ] mean_air_time_norm
- [ ] recent_route_delay (rolling 6h on this OD pair)
- [ ] scheduled_flights_next_h_norm (per-snapshot)
- [ ] mean_scheduled_distance_norm

## Targets
- [ ] Primary target: DepDelay → ArrDelay
- [ ] Add classification head: pct_arr_delayed_15
- [ ] Add auxiliary regression head: DepDelay (multi-task, λ=0.3)

## Horizons & loss
- [ ] Horizons: [1,2,3,4,5] → [1,2,4,6,8]
- [ ] Horizon weights: [5,4,3,2,1] → [3,3,4,5,5] (favor business horizons)
- [ ] Regression loss: MSE → Huber (δ=10)
- [ ] WeightedMSE → multi-task Huber + 0.3·Huber_aux + 0.5·BCE

## Architecture — applies per-model (toggle independently)
- [ ] GATConv → GATv2Conv with edge_dim=5
- [ ] BatchNorm → LayerNorm
- [ ] Residual connections in GAT encoder (`x = x + gat(x)`)
- [ ] Activation ELU → GELU
- [ ] Hidden dim 64 → 128 (encoder)

### SpatioTemporalGNN-specific
- [ ] LSTM 1-layer → 2-layer
- [ ] Sinusoidal positional encoding on LSTM input

### Seq2SeqGNN redesign (replaces autoregressive decoder)
- [ ] LSTM temporal encoder → 2-layer Transformer encoder (d_model=128, nhead=4)
- [ ] Autoregressive LSTM decoder → horizon-query MultiheadAttention decoder
- [ ] 5 separate Linear heads → single MLP applied to 5 attended queries
- [ ] Drop teacher forcing entirely (no train/eval distribution shift)
- [ ] Add learnable horizon-query embeddings Q ∈ R^{5×128}

## Training infrastructure
- [ ] CUDA 12.8 wheels + RTX 5070 enabled
- [ ] Mixed precision (bf16 autocast + GradScaler)
- [ ] LR warmup → cosine annealing
- [ ] Per-horizon validation logging (not just average)
- [ ] Gradient clip 1.0 (existing — confirm retained)

## Code hygiene (no accuracy impact, affects iteration speed)
- [ ] Single model factory in src/models/factory.py
- [ ] Single results reporter in src/evaluation/reporting.py
- [ ] BaseTrainer + 2 subclasses (collapse 3 trainer classes)
- [ ] Remove abandoned LSTM stub from configs and README
- [ ] Lightweight per-model dataclass for config validation

## Verification
- [ ] Leakage unit test (tests/test_data/test_leakage.py)
- [ ] Per-model smoke test on mock data (tests/test_integration.py)
- [ ] Populated notebooks/03_results.ipynb with comparison table & MAE-vs-horizon plot

## Suggested ablation order (when accuracy underwhelms)
1. Toggle target ArrDelay vs DepDelay — biggest expected effect on baseline-vs-GNN gap
2. Toggle exogenous future features as a group — biggest expected effect on h=6/h=8 horizons
3. Toggle BTS cause columns + lags — affects all horizons
4. Toggle Seq2SeqGNN redesign as a whole vs old architecture (same features)
5. Toggle edge features (GATv2 with edge_dim vs without) — affects GAT/Seq2Seq only
6. Toggle multi-task auxiliary DepDelay head — small regularization effect
7. Toggle Huber vs MSE loss — small, helps tail
