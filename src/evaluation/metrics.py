"""Unified evaluation metrics for delay prediction models.

All models (tabular and GNN) produce regression predictions (delay in
minutes) and are evaluated with the same set of metrics:
- Regression: MAE, RMSE, MAPE, R²
- Derived classification: Accuracy, Precision, Recall, F1
  (applying a threshold over the continuous ArrDelay prediction)

From W2 onward, the multi-task models (``output_channels=3``) also produce
a **logits** channel for ``pct_arr_delayed_15``. When that channel is
present, additional classification metrics with the ``bce_`` prefix
(``bce_accuracy``, ``bce_precision``, etc.) are computed directly from the
BCE head (sigmoid + threshold 0.5). The two metric families coexist to
allow an apples-to-apples comparison between the regressor-derived decision
and that of the dedicated classifier.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.graph_builder import (
    NUM_TARGET_CHANNELS,
    TARGET_CHANNEL_ARR_DELAY,
    TARGET_CHANNEL_PCT_DELAYED,
)


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        predictions: Model predictions.
        targets: Ground-truth values.

    Returns:
        MAE in the same units as the data (minutes).
    """
    return float(np.mean(np.abs(predictions - targets)))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        predictions: Model predictions.
        targets: Ground-truth values.

    Returns:
        RMSE in the same units as the data (minutes).
    """
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_mape(
    predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1.0
) -> float:
    """Mean Absolute Percentage Error.

    Uses epsilon to avoid division by zero for targets near 0.

    Args:
        predictions: Model predictions.
        targets: Ground-truth values.
        epsilon: Minimum denominator value.

    Returns:
        MAPE as a percentage (0-100+).
    """
    denominator = np.maximum(np.abs(targets), epsilon)
    return float(np.mean(np.abs(predictions - targets) / denominator) * 100)


def compute_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Coefficient of determination R².

    R² = 1 means perfect prediction.
    R² = 0 means the model is as good as predicting the mean.
    R² < 0 means the model is worse than predicting the mean.

    Args:
        predictions: Model predictions.
        targets: Ground-truth values.

    Returns:
        R² value.
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Compute all regression metrics.

    Args:
        predictions: Model predictions.
        targets: Ground-truth values.

    Returns:
        Dictionary with mae, rmse, mape, r2.
    """
    return {
        "mae": compute_mae(predictions, targets),
        "rmse": compute_rmse(predictions, targets),
        "mape": compute_mape(predictions, targets),
        "r2": compute_r2(predictions, targets),
    }


def compute_classification_metrics(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 15.0
) -> dict[str, float]:
    """Compute binary classification metrics derived from regression.

    Converts continuous predictions and targets (delay minutes) to binary
    classes using a delay threshold.

    Args:
        predictions: Continuous predictions (delay minutes).
        targets: Continuous ground-truth values (delay minutes).
        threshold: Threshold in minutes to consider "delayed".

    Returns:
        Dictionary with accuracy, precision, recall, f1.
    """
    pred_labels = (predictions >= threshold).astype(int)
    target_labels = (targets >= threshold).astype(int)

    tp = np.sum((pred_labels == 1) & (target_labels == 1))
    fp = np.sum((pred_labels == 1) & (target_labels == 0))
    fn = np.sum((pred_labels == 0) & (target_labels == 1))
    tn = np.sum((pred_labels == 0) & (target_labels == 0))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_bce_classification_metrics(
    pct_logits: np.ndarray,
    pct_targets: np.ndarray,
    bce_threshold: float = 0.5,
    target_threshold: float = 0.5,
) -> dict[str, float]:
    """Binary classification metrics derived from the W2 BCE head.

    The model emits ``logits`` for ``pct_arr_delayed_15`` (logit ∈ ℝ); we
    apply sigmoid and compare against ``bce_threshold`` (0.5 by default) to
    obtain the predicted label. The target is the real fraction of delayed
    flights in the window ∈ [0, 1]; it is binarized against
    ``target_threshold`` (0.5 by default — "more than half delayed" counts
    as a delayed window). Lowering ``bce_threshold`` is the natural lever
    to raise recall at the expense of precision.

    Args:
        pct_logits: Logits ``[N]`` of the pct channel (pre-sigmoid).
        pct_targets: Targets ``[N]`` ∈ [0, 1] of the same channel.
        bce_threshold: Minimum probability to predict "delayed".
        target_threshold: Minimum target fraction that counts as a positive
            label.

    Returns:
        Dict with ``bce_accuracy``, ``bce_precision``, ``bce_recall``,
        ``bce_f1``.
    """
    probs = 1.0 / (1.0 + np.exp(-pct_logits))
    pred_labels = (probs >= bce_threshold).astype(int)
    target_labels = (pct_targets >= target_threshold).astype(int)

    tp = int(np.sum((pred_labels == 1) & (target_labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (target_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (target_labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (target_labels == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "bce_accuracy": float(accuracy),
        "bce_precision": float(precision),
        "bce_recall": float(recall),
        "bce_f1": float(f1),
    }


def compute_unified_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    delay_threshold: float = 15.0,
    pct_logits: np.ndarray | None = None,
    pct_targets: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute unified metrics: regression + derived classification.

    Standard function for evaluating all of the project's models. Combines
    regression metrics (MAE, RMSE, MAPE, R²) with classification metrics
    (Accuracy, Precision, Recall, F1) derived from applying a delay
    threshold to the continuous predictions.

    If ``pct_logits`` and ``pct_targets`` are passed (W2 multi-task path),
    the ``bce_*`` metrics from the dedicated BCE head are also added,
    allowing comparison of the thresholded-from-regression decision with
    that of the explicitly trained classifier.

    Args:
        predictions: Continuous ArrDelay predictions (minutes).
        targets: Ground-truth ArrDelay values (minutes).
        delay_threshold: Threshold in minutes for the derived classification.
        pct_logits: Logits of the pct_arr_delayed channel (optional).
        pct_targets: Targets ∈ [0, 1] of the pct channel (optional).

    Returns:
        Dictionary with all metrics. If the ``pct_*`` args are present it
        also includes ``bce_accuracy/precision/recall/f1``.
    """
    regression = compute_all_metrics(predictions, targets)
    classification = compute_classification_metrics(
        predictions, targets, threshold=delay_threshold
    )
    out = {**regression, **classification}

    if pct_logits is not None and pct_targets is not None:
        out.update(compute_bce_classification_metrics(pct_logits, pct_targets))

    return out


@torch.no_grad()
def evaluate_multi_horizon_graph_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    prediction_horizons: list[int],
    delay_threshold: float = 15.0,
) -> dict[str, dict[str, float]]:
    """Evaluate a multi-horizon GNN model over temporal graphs.

    Computes unified metrics for each prediction horizon and a global
    average.

    Args:
        model: Multi-horizon GNN model in eval mode.
        graphs: List of PyG graphs with targets [num_nodes, num_horizons].
        device: Device (cpu/cuda).
        prediction_horizons: List of horizons (e.g., [1, 2, 3, 4, 5]).
        delay_threshold: Threshold in minutes for the derived classification.

    Returns:
        Dictionary with per-horizon metrics and the average:
        {"horizon_1h": {...}, "horizon_2h": {...}, ..., "average": {...}}
    """
    model.eval()
    num_horizons = len(prediction_horizons)

    # Accumulate predictions and targets per horizon. ``pct_*`` is only
    # populated if the model emits the third channel (W2 multi-task).
    all_preds: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_logits: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    has_pct_head = False

    for graph in graphs:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None
        mask = graph.active_mask

        # Output [num_nodes, num_horizons] or [num_nodes, num_horizons, C]
        preds = model(x, edge_index, edge_attr=edge_attr).cpu().numpy()
        targets_np = graph.y.numpy()
        mask_np = mask.numpy()

        # Detect whether the model emits the pct head (channel 2). We assume
        # that if the last dim matches NUM_TARGET_CHANNELS we are in
        # multi-task; the dimensions branch here, not in the model.
        pred_has_channels = preds.ndim == 3
        target_has_channels = targets_np.ndim == 3

        for h_idx in range(num_horizons):
            if pred_has_channels:
                all_preds[h_idx].append(preds[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY])
            else:
                all_preds[h_idx].append(preds[mask_np, h_idx])

            if target_has_channels:
                all_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY]
                )
            else:
                all_targets[h_idx].append(targets_np[mask_np, h_idx])

            if (
                pred_has_channels
                and target_has_channels
                and preds.shape[-1] >= NUM_TARGET_CHANNELS
            ):
                has_pct_head = True
                all_pct_logits[h_idx].append(
                    preds[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )
                all_pct_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )

    if not all_preds[0]:
        empty = {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        result = {}
        for h in prediction_horizons:
            result[f"horizon_{h}h"] = empty.copy()
        result["average"] = empty.copy()
        return result

    # Compute per-horizon metrics
    result: dict[str, dict[str, float]] = {}
    all_metrics_for_avg: list[dict[str, float]] = []

    for h_idx, h in enumerate(prediction_horizons):
        preds_h = np.concatenate(all_preds[h_idx])
        targets_h = np.concatenate(all_targets[h_idx])
        if has_pct_head:
            pct_logits_h = np.concatenate(all_pct_logits[h_idx])
            pct_targets_h = np.concatenate(all_pct_targets[h_idx])
            metrics_h = compute_unified_metrics(
                preds_h,
                targets_h,
                delay_threshold,
                pct_logits=pct_logits_h,
                pct_targets=pct_targets_h,
            )
        else:
            metrics_h = compute_unified_metrics(
                preds_h,
                targets_h,
                delay_threshold,
            )
        result[f"horizon_{h}h"] = metrics_h
        all_metrics_for_avg.append(metrics_h)

    # Average of the metrics
    avg_metrics: dict[str, float] = {}
    for key in all_metrics_for_avg[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics_for_avg]))
    result["average"] = avg_metrics

    return result


@torch.no_grad()
def evaluate_multi_horizon_sequence_model(
    model: torch.nn.Module,
    sequences: list[list[Data]],
    device: torch.device,
    prediction_horizons: list[int],
    delay_threshold: float = 15.0,
) -> dict[str, dict[str, float]]:
    """Evaluate a multi-horizon sequence model (SpatioTemporalGNN).

    Same as ``evaluate_multi_horizon_graph_model`` but operates over
    sequences of graphs instead of individual graphs. The targets and mask
    are taken from the last graph of each sequence.

    Args:
        model: Model that accepts ``list[Data]`` (already on device).
        sequences: List of sequences, each one a ``list[Data]``.
        device: Device (cpu/cuda).
        prediction_horizons: List of horizons (e.g., [1, 2, 3, 4, 5]).
        delay_threshold: Threshold in minutes for the derived classification.

    Returns:
        Dictionary with per-horizon metrics and the average:
        {"horizon_1h": {...}, ..., "average": {...}}
    """
    model.eval()
    num_horizons = len(prediction_horizons)

    all_preds: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_logits: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_pct_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    has_pct_head = False

    for sequence in sequences:
        # Move each graph in the sequence to device
        seq_on_device = []
        for g in sequence:
            g_dev = g.clone()
            g_dev.x = g.x.to(device)
            g_dev.edge_index = g.edge_index.to(device)
            if g.edge_attr is not None:
                g_dev.edge_attr = g.edge_attr.to(device)
            if g.y is not None:
                g_dev.y = g.y.to(device)
            if g.active_mask is not None:
                g_dev.active_mask = g.active_mask.to(device)
            seq_on_device.append(g_dev)

        preds = model(seq_on_device).cpu().numpy()

        last_graph = sequence[-1]
        targets_np = last_graph.y.numpy()
        mask_np = last_graph.active_mask.numpy()

        pred_has_channels = preds.ndim == 3
        target_has_channels = targets_np.ndim == 3

        for h_idx in range(num_horizons):
            if pred_has_channels:
                all_preds[h_idx].append(preds[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY])
            else:
                all_preds[h_idx].append(preds[mask_np, h_idx])

            if target_has_channels:
                all_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_ARR_DELAY]
                )
            else:
                all_targets[h_idx].append(targets_np[mask_np, h_idx])

            if (
                pred_has_channels
                and target_has_channels
                and preds.shape[-1] >= NUM_TARGET_CHANNELS
            ):
                has_pct_head = True
                all_pct_logits[h_idx].append(
                    preds[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )
                all_pct_targets[h_idx].append(
                    targets_np[mask_np, h_idx, TARGET_CHANNEL_PCT_DELAYED]
                )

    if not all_preds[0]:
        empty = {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        result = {}
        for h in prediction_horizons:
            result[f"horizon_{h}h"] = empty.copy()
        result["average"] = empty.copy()
        return result

    result: dict[str, dict[str, float]] = {}
    all_metrics_for_avg: list[dict[str, float]] = []

    for h_idx, h in enumerate(prediction_horizons):
        preds_h = np.concatenate(all_preds[h_idx])
        targets_h = np.concatenate(all_targets[h_idx])
        if has_pct_head:
            pct_logits_h = np.concatenate(all_pct_logits[h_idx])
            pct_targets_h = np.concatenate(all_pct_targets[h_idx])
            metrics_h = compute_unified_metrics(
                preds_h,
                targets_h,
                delay_threshold,
                pct_logits=pct_logits_h,
                pct_targets=pct_targets_h,
            )
        else:
            metrics_h = compute_unified_metrics(
                preds_h,
                targets_h,
                delay_threshold,
            )
        result[f"horizon_{h}h"] = metrics_h
        all_metrics_for_avg.append(metrics_h)

    avg_metrics: dict[str, float] = {}
    for key in all_metrics_for_avg[0]:
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics_for_avg]))
    result["average"] = avg_metrics

    return result


@torch.no_grad()
def evaluate_graph_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    delay_threshold: float = 15.0,
) -> dict[str, float]:
    """Evaluate a GNN model over a list of temporal graphs.

    The model produces continuous predictions (delay minutes) and is
    evaluated with unified regression + classification metrics.

    Args:
        model: GNN model in eval mode.
        graphs: List of PyG graphs with active_mask.
        device: Device (cpu/cuda).
        delay_threshold: Threshold in minutes for the derived classification.

    Returns:
        Dictionary with unified metrics (regression + classification).
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for graph in graphs:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = graph.edge_attr.to(device) if graph.edge_attr is not None else None
        mask = graph.active_mask

        # Direct model output (regression, no sigmoid)
        preds = model(x, edge_index, edge_attr=edge_attr).squeeze(-1)
        preds = preds.cpu().numpy()

        all_predictions.append(preds[mask.numpy()])
        all_targets.append(graph.y[mask].numpy())

    if not all_predictions:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return compute_unified_metrics(predictions, targets, delay_threshold)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    delay_threshold: float = 15.0,
) -> dict[str, float]:
    """Evaluate a tabular model over a full dataloader.

    Computes unified regression + derived-classification metrics.

    Args:
        model: PyTorch model in eval mode.
        dataloader: DataLoader with test data.
        device: Device (cpu/cuda).
        delay_threshold: Threshold in minutes for the derived classification.

    Returns:
        Dictionary with unified metrics (regression + classification).
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for features, targets in dataloader:
        features = features.to(device)
        predictions = model(features).squeeze(-1).cpu().numpy()
        all_predictions.append(predictions)
        all_targets.append(targets.numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return compute_unified_metrics(predictions, targets, delay_threshold)
