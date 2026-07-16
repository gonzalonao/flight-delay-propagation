# Archivo — línea exploratoria per-vuelo (gradient boosting / LightGBM)

> **No forma parte del sistema implementado del TFM.** Este material se conserva
> aquí por si la vía per-vuelo se retoma en el futuro. La memoria final describe
> únicamente la vía **GNN espacio-temporal** (ver [`../../tfm-diagramas.md`](../../tfm-diagramas.md)).

## Diagramas archivados

| Archivo | Descripción |
|---|---|
| `01_pipeline_general.mmd` / `png/` | Versión **anterior** del flujo de extremo a extremo, con dos vías (GNN + per-vuelo) y fusión. Sustituida en el set principal por una versión solo-GNN. |
| `07_track_per_vuelo.mmd` / `png/` | Vía per-vuelo *point-in-time*: familias de features (schedule, rotación, estado del aeropuerto, meteo), guardián anti-fuga, regresor de *gradient boosting* y lectura multi-umbral. |
| `08_fusion_gnn_per_vuelo.mmd` / `png/` | Fusión GNN ↔ per-vuelo (puente de granularidad + estrategias de *stacking*). |

Re-renderizado (si se retoma):
`npx -p @mermaid-js/mermaid-cli mmdc -i <archivo>.mmd -o png/<archivo>.png -b "#0d1117" -s 3`

## Fragmentos de código asociados (referencia)

Código exploratorio en `src/flight_level/` (presente en el repositorio pero no
desplegado). Se conservan aquí las referencias por si se documenta esta línea.

- **Feature de rotación de aeronave** — `src/flight_level/features.py:142-185`
  (`add_rotation_asof`): observabilidad *as-of* `t_pred` y `turnaround_slack`,
  la señal más importante del estudio.
- **Guardián anti-fuga** — `src/flight_level/features.py:68-73` y `258-262`
  (`BLOCKED_SELF_COLUMNS`, `assert_no_leakage`).
- **Regresión → umbral en inferencia** — `src/flight_level/model.py:110-135`
  (`FlightDelayModel.evaluate`): recall/precision/F1 a 15/30/45/60 min.
- **Puente de granularidad de la fusión** — `src/flight_level/fusion.py:55-73`
  (`attach_gnn_feature`): elige el pronóstico GNN más fresco admisible ≤ `t_pred`.

Resultados completos del estudio: [`../../flight-level-signal-study.md`](../../flight-level-signal-study.md).
