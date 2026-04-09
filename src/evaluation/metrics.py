"""Métricas de evaluación para modelos de predicción de retrasos.

Proporciona funciones para calcular MAE, RMSE, MAPE y R² (regresión),
así como accuracy, precision, recall y F1 (clasificación binaria).
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
    """Calcula todas las métricas a la vez.

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
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calcula métricas de clasificación binaria.

    Args:
        predictions: Probabilidades predichas (después de sigmoid).
        targets: Valores reales binarios (0 o 1).
        threshold: Umbral para convertir probabilidades a clases.

    Returns:
        Diccionario con accuracy, precision, recall, f1.
    """
    pred_labels = (predictions >= threshold).astype(int)
    target_labels = targets.astype(int)

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


@torch.no_grad()
def evaluate_graph_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
) -> dict[str, float]:
    """Evalúa un modelo GNN sobre una lista de grafos temporales.

    Args:
        model: Modelo GNN en modo eval.
        graphs: Lista de grafos PyG con active_mask.
        device: Dispositivo (cpu/cuda).

    Returns:
        Diccionario con métricas de clasificación.
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for graph in graphs:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_weight = None
        if graph.edge_attr is not None:
            edge_weight = graph.edge_attr.squeeze(-1).to(device)
        mask = graph.active_mask

        logits = model(x, edge_index, edge_weight=edge_weight).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_predictions.append(probs[mask.numpy()])
        all_targets.append(graph.y[mask].numpy())

    if not all_predictions:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return compute_classification_metrics(predictions, targets)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evalúa un modelo sobre un dataloader completo.

    Args:
        model: Modelo de PyTorch en modo eval.
        dataloader: DataLoader con datos de test.
        device: Dispositivo (cpu/cuda).

    Returns:
        Diccionario con todas las métricas.
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

    return compute_all_metrics(predictions, targets)
