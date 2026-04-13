"""Funciones de pérdida para modelos de predicción de retrasos.

Incluye pérdidas estándar y variantes ponderadas para el problema
de predicción multi-horizonte. Funciona con tensores 1D [N] (single-
horizon) y 2D [N, H] (multi-horizonte).
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE ponderado que penaliza más los errores en retrasos significativos.

    Asigna mayor peso a nodos donde el target real supera un umbral
    de retraso, compensando el desbalance entre aeropuertos retrasados
    y puntuales. Compatible con single-horizon [N] y multi-horizonte
    [N, H].

    Args:
        high_delay_threshold: Umbral en minutos para considerar retraso alto.
        high_delay_weight: Peso multiplicador para muestras con retraso alto.
    """

    def __init__(
        self,
        high_delay_threshold: float = 15.0,
        high_delay_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.threshold = high_delay_threshold
        self.high_weight = high_delay_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calcula la pérdida MSE ponderada.

        Args:
            predictions: Predicciones [N] o [N, H].
            targets: Valores reales [N] o [N, H].

        Returns:
            Pérdida escalar.
        """
        se = (predictions - targets) ** 2

        # Asignar pesos: más peso a retrasos altos en el target
        weights = torch.ones_like(targets)
        delayed_mask = targets.abs() >= self.threshold
        weights[delayed_mask] = self.high_weight

        return (se * weights).mean()
