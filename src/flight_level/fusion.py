"""Fusión GNN (aeropuerto-hora) ↔ modelo per-vuelo — utilidades de comparación.

⛔ Rama `explore/flight-level-signal`: no fusionar a main (ver BRANCH_NOTES.md).

El GNN baseline y el modelo per-vuelo viven separados; aquí NO se modifica ninguno.
Esta capa solo **consume sus salidas** para evaluar, point-in-time, cómo juegan
juntos:

* **Bridge de granularidad.** El GNN predice retraso medio por (aeropuerto, hora).
  Para un vuelo F que llega a ``Dest`` en la hora ``T``, su feature-GNN es el
  pronóstico del GNN para ``(Dest, T)`` *emitido en el último instante ≤ t_pred*,
  donde ``t_pred = salida_programada − horizonte``. Como ``prediction_ts = T − h``,
  basta elegir el menor horizonte GNN ``h`` con ``h ≥ lead`` (``lead = T − t_pred``):
  así ``prediction_ts ≤ t_pred`` (sin fuga) y se usa el pronóstico más fresco.

* **Estrategias** evaluadas a nivel per-vuelo (mismo test que el modelo per-vuelo):
  ``gbm_only``, ``gnn_only``, ``blend`` (peso w* ajustado en val), ``switch``
  (elige por horizonte el mejor en val) y ``stack`` (meta-GBM sobre las dos
  predicciones, ajustado en val). Los ajustes se hacen en **val** (out-of-sample)
  y se reportan en **test** para evitar fuga de stacking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

GNN_HORIZONS = (1, 2, 4, 6, 8)


def load_gnn_predictions(path: str) -> pd.DataFrame:
    """Carga el frame largo del GNN (salida de scripts/predict.py)."""
    df = pd.read_parquet(
        path,
        columns=["airport_code", "horizon_h", "predicted_arr_delay_min",
                 "pct_arr_delayed_15", "target_ts", "prediction_ts"],
    )
    df["target_ts"] = pd.to_datetime(df["target_ts"])
    df["prediction_ts"] = pd.to_datetime(df["prediction_ts"])
    return df


def attach_gnn_feature(
    work: pd.DataFrame,
    gnn: pd.DataFrame,
    horizon_h: int,
    gnn_horizons: tuple[int, ...] = GNN_HORIZONS,
) -> pd.DataFrame:
    """Une, por fila y point-in-time, el pronóstico del GNN para la llegada de F.

    ``work`` debe traer ``dep_dt``, ``arr_dt`` y ``Dest`` (de ``build_work_frame``).
    Añade ``gnn_pred_arr_delay``, ``gnn_pred_pct15`` y ``gnn_lead_h``.
    """
    t_pred = work["dep_dt"] - pd.Timedelta(hours=horizon_h)
    target_hour = work["arr_dt"].dt.floor("h")
    lead = (target_hour - t_pred).dt.total_seconds() / 3600.0

    gh = np.array(sorted(gnn_horizons))
    # menor horizonte GNN >= lead (clip al máximo si lead supera el alcance GNN).
    idx = np.clip(np.searchsorted(gh, lead.to_numpy(), side="left"), 0, len(gh) - 1)
    chosen = gh[idx]

    key = pd.DataFrame({
        "airport_code": work["Dest"].astype(str).to_numpy(),
        "target_ts": target_hour.to_numpy(),
        "horizon_h": chosen.astype(int),
    })
    merged = key.merge(
        gnn[["airport_code", "target_ts", "horizon_h",
             "predicted_arr_delay_min", "pct_arr_delayed_15"]],
        on=["airport_code", "target_ts", "horizon_h"], how="left",
    )
    out = work.copy()
    out["gnn_pred_arr_delay"] = merged["predicted_arr_delay_min"].to_numpy()
    out["gnn_pred_pct15"] = merged["pct_arr_delayed_15"].to_numpy()
    out["gnn_lead_h"] = lead.to_numpy()
    return out


def f1_at(y_true: np.ndarray, y_pred: np.ndarray, thr: int = 15) -> float:
    """F1 de la clasificación derivada al umbral ``thr`` (objetivo de selección)."""
    t, p = y_true >= thr, y_pred >= thr
    rec = recall_score(t, p, zero_division=0)
    prec = precision_score(t, p, zero_division=0)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def best_blend_weight(
    y_val: np.ndarray, gbm_val: np.ndarray, gnn_val: np.ndarray,
    grid: np.ndarray | None = None, thr: int = 15,
) -> float:
    """w* que maximiza F1@thr en validación para ``w·gbm + (1−w)·gnn``."""
    grid = np.linspace(0.0, 1.0, 11) if grid is None else grid
    scores = [(w, f1_at(y_val, w * gbm_val + (1 - w) * gnn_val, thr)) for w in grid]
    return max(scores, key=lambda t: t[1])[0]
