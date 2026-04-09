"""Métricas de evaluación para modelos de predicción de retrasos.

Proporciona funciones para calcular MAE, RMSE, MAPE y R², tanto
para predicciones simples como multi-horizonte.
"""

import numpy as np
import torch


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
