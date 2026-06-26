"""Forward pass + prediction assembly for sequence models.

The ``seq2seq_gnn`` model (and the rest of ``SEQUENCE_MODELS``) consumes a
sequence of ``input_window`` graphs and emits, for the last instant of the
window (``current_end``), one prediction per airport and horizon with
``NUM_TARGET_CHANNELS`` channels:

    channel 0 (``TARGET_CHANNEL_ARR_DELAY``)   -> ArrDelay in minutes (regression)
    channel 2 (``TARGET_CHANNEL_PCT_DELAYED``) -> logit of pct_arr_delayed_15

This module converts that output into a "long" DataFrame with the **same
schema** as the ``predictions_latest`` table that the Fabric notebook
writes, so that the live dashboard (DirectLake) and the local export
consume identical rows:

    airport_code, horizon_h, predicted_arr_delay_min,
    pct_arr_delayed_15, target_ts, prediction_ts

When the graphs carry ``y`` (historical window), ground-truth columns are
also added (``actual_arr_delay_min``, ``actual_arr_del15``, ``pct_target``)
to feed the report's accuracy page.
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

# Canonical schema (identical to predictions_latest in OneLake).
PREDICTION_COLUMNS = [
    "airport_code",
    "horizon_h",
    "predicted_arr_delay_min",
    "pct_arr_delayed_15",
    "target_ts",
    "prediction_ts",
]

# Extra columns available when ground truth is present (local export).
ACTUAL_COLUMNS = ["actual_arr_delay_min", "actual_arr_del15", "pct_target"]


def invert_airport_map(airport_map: dict[str, int]) -> dict[int, str]:
    """Invert ``{IATA: idx}`` -> ``{idx: IATA}`` (graph node order)."""
    return {int(idx): iata for iata, idx in airport_map.items()}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _move_sequence_to_device(sequence: list[Data], device: torch.device) -> list[Data]:
    """Clone and move to ``device`` the tensors of each graph in the sequence.

    Replica of the logic in ``evaluate_multi_horizon_sequence_model`` to
    keep exact parity with evaluation. Non-tensor attributes
    (``timestamp``) are preserved by the ``clone()``.
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
    """Forward pass of a single sequence.

    Returns:
        NumPy array with the model output: ``[num_nodes, num_horizons, C]``
        (multi-task) or ``[num_nodes, num_horizons]`` (regression only).
    """
    model.eval()
    seq_dev = _move_sequence_to_device(sequence, device)
    return model(seq_dev).cpu().numpy()


def _reference_ts(graph: Data, override: pd.Timestamp | None) -> pd.Timestamp:
    """``current_end`` instant (prediction_ts) to use for this sequence.

    If ``override`` is passed it takes priority (live use of a single
    sequence, where "now" is explicit). Otherwise, the target graph's
    ``timestamp`` is used (batch path); as a last resort, the current UTC
    time.
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
    """Assemble the long rows of ONE sequence in a vectorized way."""
    has_channels = preds.ndim == 3
    has_pct = has_channels and preds.shape[-1] >= NUM_TARGET_CHANNELS

    mask = getattr(target_graph, "active_mask", None)
    if mask is not None:
        active_idx = np.nonzero(mask.cpu().numpy())[0]
    else:
        active_idx = np.arange(preds.shape[0])

    # Only nodes with a known IATA (defensive against partial maps).
    active_idx = np.array([i for i in active_idx if int(i) in idx_to_iata], dtype=int)
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

    # target_ts per horizon (cloud convention: prediction_ts + h hours).
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

    return df[
        PREDICTION_COLUMNS + (ACTUAL_COLUMNS if "actual_arr_delay_min" in df else [])
    ]


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
    """Run the model over ``sequences`` and return the canonical long frame.

    Args:
        model: Sequence model (``seq2seq_gnn``) already on ``device``.
        sequences: List of sequences (``list[list[Data]]``), e.g. the output
            of ``create_temporal_sequences`` over the desired split. For live
            inference it is a list with a single sequence.
        prediction_horizons: Horizons in hours (e.g. ``[1, 2, 4, 6, 8]``).
        idx_to_iata: Node-index -> IATA-code mapping
            (use :func:`invert_airport_map`).
        device: Compute device.
        include_actuals: If the graphs carry ``y``, add ground-truth columns
            (for the report's accuracy page).
        reference_ts: If provided, REPLACES ``prediction_ts`` in ALL
            sequences (live use of a single sequence, where "now" is
            explicit). If None, each graph's ``timestamp`` is used.

    Returns:
        ``pd.DataFrame`` with the columns of :data:`PREDICTION_COLUMNS`
        (and :data:`ACTUAL_COLUMNS` if applicable), sorted by
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
    return out.sort_values(["prediction_ts", "airport_code", "horizon_h"]).reset_index(
        drop=True
    )
