"""Forward-pass + ensamblado de predicciones para modelos de secuencia.

El modelo ``seq2seq_gnn`` (y demás ``SEQUENCE_MODELS``) consume una
secuencia de ``input_window`` grafos y emite, para el último instante de la
ventana (``current_end``), una predicción por aeropuerto y horizonte con
``NUM_TARGET_CHANNELS`` canales:

    canal 0 (``TARGET_CHANNEL_ARR_DELAY``)   -> ArrDelay en minutos (regresión)
    canal 2 (``TARGET_CHANNEL_PCT_DELAYED``) -> logit de pct_arr_delayed_15

Este módulo convierte esa salida en un DataFrame "long" con el **mismo
esquema** que la tabla ``predictions_latest`` que escribe el notebook de
Fabric, de modo que el dashboard live (DirectLake) y el export local
consumen filas idénticas:

    airport_code, horizon_h, predicted_arr_delay_min,
    pct_arr_delayed_15, target_ts, prediction_ts

Cuando los grafos traen ``y`` (ventana histórica) se añaden además columnas
de verdad-terreno (``actual_arr_delay_min``, ``actual_arr_del15``,
``pct_target``) para alimentar la página de precisión del informe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.graph_builder import (
    NUM_TARGET_CHANNELS,
    TARGET_CHANNEL_ARR_DELAY,
    TARGET_CHANNEL_PCT_DELAYED,
)

# Esquema canónico (idéntico a predictions_latest en OneLake).
PREDICTION_COLUMNS = [
    "airport_code",
    "horizon_h",
    "predicted_arr_delay_min",
    "pct_arr_delayed_15",
    "target_ts",
    "prediction_ts",
]

# Columnas extra disponibles cuando hay verdad-terreno (export local).
ACTUAL_COLUMNS = ["actual_arr_delay_min", "actual_arr_del15", "pct_target"]


def invert_airport_map(airport_map: dict[str, int]) -> dict[int, str]:
    """Invierte ``{IATA: idx}`` -> ``{idx: IATA}`` (orden de nodos del grafo)."""
    return {int(idx): iata for iata, idx in airport_map.items()}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _move_sequence_to_device(
    sequence: list[Data], device: torch.device
) -> list[Data]:
    """Clona y mueve a ``device`` los tensores de cada grafo de la secuencia.

    Réplica de la lógica de ``evaluate_multi_horizon_sequence_model`` para
    mantener paridad exacta con la evaluación. Los atributos no-tensoriales
    (``timestamp``) se preservan por el ``clone()``.
    """
    seq_on_device: list[Data] = []
    for g in sequence:
        g_dev = g.clone()
        g_dev.x = g.x.to(device)
        g_dev.edge_index = g.edge_index.to(device)
        if g.edge_attr is not None:
            g_dev.edge_attr = g.edge_attr.to(device)
        if g.y is not None:
            g_dev.y = g.y.to(device)
        if getattr(g, "active_mask", None) is not None:
            g_dev.active_mask = g.active_mask.to(device)
        seq_on_device.append(g_dev)
    return seq_on_device


@torch.no_grad()
def predict_sequence(
    model: torch.nn.Module,
    sequence: list[Data],
    device: torch.device,
) -> np.ndarray:
    """Forward de una sola secuencia.

    Returns:
        Array NumPy con la salida del modelo: ``[num_nodes, num_horizons, C]``
        (multi-tarea) o ``[num_nodes, num_horizons]`` (sólo regresión).
    """
    model.eval()
    seq_dev = _move_sequence_to_device(sequence, device)
    return model(seq_dev).cpu().numpy()


def _reference_ts(graph: Data, override: pd.Timestamp | None) -> pd.Timestamp:
    """Instante ``current_end`` (prediction_ts) a usar para esta secuencia.

    Si se pasa ``override`` tiene prioridad (uso live de una sola secuencia,
    donde "ahora" es explícito). Si no, se usa el ``timestamp`` del grafo
    objetivo (ruta batch); como último recurso, la hora UTC actual.
    """
    if override is not None:
        return pd.Timestamp(override)
    ts = getattr(graph, "timestamp", None)
    if ts is not None:
        return pd.Timestamp(ts)
    return pd.Timestamp.utcnow()


def _frame_from_sequence(
    preds: np.ndarray,
    target_graph: Data,
    prediction_horizons: list[int],
    idx_to_iata: dict[int, str],
    prediction_ts: pd.Timestamp,
    include_actuals: bool,
) -> pd.DataFrame:
    """Ensambla las filas long de UNA secuencia de forma vectorizada."""
    has_channels = preds.ndim == 3
    has_pct = has_channels and preds.shape[-1] >= NUM_TARGET_CHANNELS

    mask = getattr(target_graph, "active_mask", None)
    if mask is not None:
        active_idx = np.nonzero(mask.cpu().numpy())[0]
    else:
        active_idx = np.arange(preds.shape[0])

    # Sólo nodos con IATA conocido (defensivo frente a mapas parciales).
    active_idx = np.array(
        [i for i in active_idx if int(i) in idx_to_iata], dtype=int
    )
    if active_idx.size == 0:
        cols = PREDICTION_COLUMNS + (ACTUAL_COLUMNS if include_actuals else [])
        return pd.DataFrame(columns=cols)

    iatas = np.array([idx_to_iata[int(i)] for i in active_idx])
    horizons = np.asarray(prediction_horizons, dtype=int)
    k, n_h = active_idx.size, horizons.size

    if has_channels:
        arr = preds[active_idx][:, :, TARGET_CHANNEL_ARR_DELAY]
    else:
        arr = preds[active_idx]
    if has_pct:
        pct = _sigmoid(preds[active_idx][:, :, TARGET_CHANNEL_PCT_DELAYED])
    else:
        pct = np.full((k, n_h), np.nan, dtype=float)

    # target_ts por horizonte (convención cloud: prediction_ts + h horas).
    ts_by_h = {
        int(h): (prediction_ts + pd.Timedelta(hours=int(h))).isoformat()
        for h in horizons
    }

    df = pd.DataFrame(
        {
            "airport_code": np.repeat(iatas, n_h),
            "horizon_h": np.tile(horizons, k),
            "predicted_arr_delay_min": arr.reshape(-1).astype(float),
            "pct_arr_delayed_15": pct.reshape(-1).astype(float),
        }
    )
    df["target_ts"] = df["horizon_h"].map(ts_by_h)
    df["prediction_ts"] = prediction_ts.isoformat()

    if include_actuals and getattr(target_graph, "y", None) is not None:
        targets = target_graph.y.cpu().numpy()
        target_has_channels = targets.ndim == 3
        if target_has_channels:
            actual = targets[active_idx][:, :, TARGET_CHANNEL_ARR_DELAY]
            if targets.shape[-1] >= NUM_TARGET_CHANNELS:
                pct_t = targets[active_idx][:, :, TARGET_CHANNEL_PCT_DELAYED]
            else:
                pct_t = np.full((k, n_h), np.nan, dtype=float)
        else:
            actual = targets[active_idx]
            pct_t = np.full((k, n_h), np.nan, dtype=float)
        actual_flat = actual.reshape(-1).astype(float)
        df["actual_arr_delay_min"] = actual_flat
        df["actual_arr_del15"] = (actual_flat >= 15.0).astype(int)
        df["pct_target"] = pct_t.reshape(-1).astype(float)

    return df[PREDICTION_COLUMNS + (ACTUAL_COLUMNS if "actual_arr_delay_min" in df else [])]


@torch.no_grad()
def predict_to_frame(
    model: torch.nn.Module,
    sequences: list[list[Data]],
    prediction_horizons: list[int],
    idx_to_iata: dict[int, str],
    device: torch.device,
    *,
    include_actuals: bool = True,
    reference_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Ejecuta el modelo sobre ``sequences`` y devuelve el frame long canónico.

    Args:
        model: Modelo de secuencia (``seq2seq_gnn``) ya en ``device``.
        sequences: Lista de secuencias (``list[list[Data]]``), p.ej. la salida
            de ``create_temporal_sequences`` sobre el split deseado. Para la
            inferencia live es una lista con una sola secuencia.
        prediction_horizons: Horizontes en horas (e.g. ``[1, 2, 4, 6, 8]``).
        idx_to_iata: Mapeo índice-de-nodo -> código IATA
            (usa :func:`invert_airport_map`).
        device: Dispositivo de cómputo.
        include_actuals: Si los grafos traen ``y``, añade columnas de
            verdad-terreno (para la página de precisión del informe).
        reference_ts: Si se indica, SUSTITUYE ``prediction_ts`` en TODAS las
            secuencias (uso live de una sola secuencia, donde "ahora" es
            explícito). Si es None se usa el ``timestamp`` de cada grafo.

    Returns:
        ``pd.DataFrame`` con las columnas de :data:`PREDICTION_COLUMNS`
        (y :data:`ACTUAL_COLUMNS` si aplica), ordenado por
        ``prediction_ts, airport_code, horizon_h``.
    """
    model.eval()
    frames: list[pd.DataFrame] = []
    for sequence in sequences:
        if not sequence:
            continue
        target_graph = sequence[-1]
        prediction_ts = _reference_ts(target_graph, reference_ts)
        preds = predict_sequence(model, sequence, device)
        frames.append(
            _frame_from_sequence(
                preds,
                target_graph,
                prediction_horizons,
                idx_to_iata,
                prediction_ts,
                include_actuals,
            )
        )

    if not frames:
        cols = PREDICTION_COLUMNS + (ACTUAL_COLUMNS if include_actuals else [])
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        ["prediction_ts", "airport_code", "horizon_h"]
    ).reset_index(drop=True)
