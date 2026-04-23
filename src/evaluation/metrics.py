"""Métricas de evaluación unificadas para modelos de predicción de retrasos.

Todos los modelos (tabulares y GNN) producen predicciones de regresión
(retraso en minutos) y se evalúan con el mismo conjunto de métricas:
- Regresión: MAE, RMSE, MAPE, R²
- Clasificación derivada: Accuracy, Precision, Recall, F1
  (aplicando un umbral sobre la predicción continua)
"""

import numpy as np
import torch
from torch_geometric.data import Data


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        MAE en las mismas unidades que los datos (minutos).
    """
    return float(np.mean(np.abs(predictions - targets)))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        RMSE en las mismas unidades que los datos (minutos).
    """
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def compute_mape(
    predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1.0
) -> float:
    """Mean Absolute Percentage Error.

    Usa epsilon para evitar división por cero en targets cercanos a 0.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.
        epsilon: Valor mínimo del denominador.

    Returns:
        MAPE como porcentaje (0-100+).
    """
    denominator = np.maximum(np.abs(targets), epsilon)
    return float(np.mean(np.abs(predictions - targets) / denominator) * 100)


def compute_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Coeficiente de determinación R².

    R² = 1 significa predicción perfecta.
    R² = 0 significa que el modelo es igual de bueno que predecir la media.
    R² < 0 significa que el modelo es peor que predecir la media.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        Valor R².
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Calcula todas las métricas de regresión.

    Args:
        predictions: Predicciones del modelo.
        targets: Valores reales.

    Returns:
        Diccionario con mae, rmse, mape, r2.
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
    """Calcula métricas de clasificación binaria derivadas de regresión.

    Convierte predicciones y targets continuos (minutos de retraso)
    a clases binarias usando un umbral de retraso.

    Args:
        predictions: Predicciones continuas (minutos de retraso).
        targets: Valores reales continuos (minutos de retraso).
        threshold: Umbral en minutos para considerar "retrasado".

    Returns:
        Diccionario con accuracy, precision, recall, f1.
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


def compute_unified_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    delay_threshold: float = 15.0,
) -> dict[str, float]:
    """Calcula métricas unificadas: regresión + clasificación derivada.

    Función estándar para evaluar todos los modelos del proyecto.
    Combina métricas de regresión (MAE, RMSE, MAPE, R²) con métricas
    de clasificación (Accuracy, Precision, Recall, F1) derivadas
    de aplicar un umbral de retraso a las predicciones continuas.

    Args:
        predictions: Predicciones continuas (minutos de retraso).
        targets: Valores reales continuos (minutos de retraso).
        delay_threshold: Umbral en minutos para clasificación binaria.

    Returns:
        Diccionario con todas las métricas (regresión + clasificación).
    """
    regression = compute_all_metrics(predictions, targets)
    classification = compute_classification_metrics(
        predictions, targets, threshold=delay_threshold
    )
    return {**regression, **classification}


@torch.no_grad()
def evaluate_multi_horizon_graph_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    prediction_horizons: list[int],
    delay_threshold: float = 15.0,
) -> dict[str, dict[str, float]]:
    """Evalúa un modelo GNN multi-horizonte sobre grafos temporales.

    Calcula métricas unificadas para cada horizonte de predicción
    y un promedio global.

    Args:
        model: Modelo GNN multi-horizonte en modo eval.
        graphs: Lista de grafos PyG con targets [num_nodes, num_horizons].
        device: Dispositivo (cpu/cuda).
        prediction_horizons: Lista de horizontes (e.g., [1, 2, 3, 4, 5]).
        delay_threshold: Umbral en minutos para clasificación derivada.

    Returns:
        Diccionario con métricas por horizonte y promedio:
        {"horizon_1h": {...}, "horizon_2h": {...}, ..., "average": {...}}
    """
    model.eval()
    num_horizons = len(prediction_horizons)

    # Acumular predicciones y targets por horizonte
    all_preds: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]

    for graph in graphs:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = (
            graph.edge_attr.to(device) if graph.edge_attr is not None else None
        )
        mask = graph.active_mask

        # Salida [num_nodes, num_horizons]
        preds = model(x, edge_index, edge_attr=edge_attr)
        preds = preds.cpu().numpy()

        targets_np = graph.y.numpy()
        mask_np = mask.numpy()

        for h_idx in range(num_horizons):
            all_preds[h_idx].append(preds[mask_np, h_idx])
            all_targets[h_idx].append(targets_np[mask_np, h_idx])

    if not all_preds[0]:
        empty = {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }
        result = {}
        for h in prediction_horizons:
            result[f"horizon_{h}h"] = empty.copy()
        result["average"] = empty.copy()
        return result

    # Calcular métricas por horizonte
    result: dict[str, dict[str, float]] = {}
    all_metrics_for_avg: list[dict[str, float]] = []

    for h_idx, h in enumerate(prediction_horizons):
        preds_h = np.concatenate(all_preds[h_idx])
        targets_h = np.concatenate(all_targets[h_idx])
        metrics_h = compute_unified_metrics(preds_h, targets_h, delay_threshold)
        result[f"horizon_{h}h"] = metrics_h
        all_metrics_for_avg.append(metrics_h)

    # Promedio de métricas
    avg_metrics: dict[str, float] = {}
    for key in all_metrics_for_avg[0]:
        avg_metrics[key] = float(
            np.mean([m[key] for m in all_metrics_for_avg])
        )
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
    """Evalúa un modelo de secuencias multi-horizonte (SpatioTemporalGNN).

    Igual que ``evaluate_multi_horizon_graph_model`` pero opera sobre
    secuencias de grafos en vez de grafos individuales. Los targets y la
    máscara se toman del último grafo de cada secuencia.

    Args:
        model: Modelo que acepta ``list[Data]`` (ya en device).
        sequences: Lista de secuencias, cada una es ``list[Data]``.
        device: Dispositivo (cpu/cuda).
        prediction_horizons: Lista de horizontes (e.g., [1, 2, 3, 4, 5]).
        delay_threshold: Umbral en minutos para clasificación derivada.

    Returns:
        Diccionario con métricas por horizonte y promedio:
        {"horizon_1h": {...}, ..., "average": {...}}
    """
    model.eval()
    num_horizons = len(prediction_horizons)

    all_preds: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]
    all_targets: list[list[np.ndarray]] = [[] for _ in range(num_horizons)]

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

        for h_idx in range(num_horizons):
            all_preds[h_idx].append(preds[mask_np, h_idx])
            all_targets[h_idx].append(targets_np[mask_np, h_idx])

    if not all_preds[0]:
        empty = {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
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
        metrics_h = compute_unified_metrics(preds_h, targets_h, delay_threshold)
        result[f"horizon_{h}h"] = metrics_h
        all_metrics_for_avg.append(metrics_h)

    avg_metrics: dict[str, float] = {}
    for key in all_metrics_for_avg[0]:
        avg_metrics[key] = float(
            np.mean([m[key] for m in all_metrics_for_avg])
        )
    result["average"] = avg_metrics

    return result


@torch.no_grad()
def evaluate_graph_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    delay_threshold: float = 15.0,
) -> dict[str, float]:
    """Evalúa un modelo GNN sobre una lista de grafos temporales.

    El modelo produce predicciones continuas (minutos de retraso)
    y se evalúa con métricas unificadas de regresión + clasificación.

    Args:
        model: Modelo GNN en modo eval.
        graphs: Lista de grafos PyG con active_mask.
        device: Dispositivo (cpu/cuda).
        delay_threshold: Umbral en minutos para clasificación derivada.

    Returns:
        Diccionario con métricas unificadas (regresión + clasificación).
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for graph in graphs:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_attr = (
            graph.edge_attr.to(device) if graph.edge_attr is not None else None
        )
        mask = graph.active_mask

        # Salida directa del modelo (regresión, sin sigmoid)
        preds = model(x, edge_index, edge_attr=edge_attr).squeeze(-1)
        preds = preds.cpu().numpy()

        all_predictions.append(preds[mask.numpy()])
        all_targets.append(graph.y[mask].numpy())

    if not all_predictions:
        return {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
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
    """Evalúa un modelo tabular sobre un dataloader completo.

    Calcula métricas unificadas de regresión + clasificación derivada.

    Args:
        model: Modelo de PyTorch en modo eval.
        dataloader: DataLoader con datos de test.
        device: Dispositivo (cpu/cuda).
        delay_threshold: Umbral en minutos para clasificación derivada.

    Returns:
        Diccionario con métricas unificadas (regresión + clasificación).
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
