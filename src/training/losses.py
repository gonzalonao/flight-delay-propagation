"""Funciones de pérdida para modelos de predicción de retrasos.

Incluye pérdidas estándar y variantes ponderadas para el problema
de predicción multi-horizonte.
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE ponderado que penaliza más los errores en ciertos rangos de retraso.

    Útil para dar más importancia a retrasos significativos (>15 min)
    frente a vuelos puntuales.

    Args:
        high_delay_threshold: Umbral en minutos para considerar retraso alto.
        high_delay_weight: Peso extra para muestras con retraso alto.
    """

    def __init__(
        self,
        high_delay_threshold: float = 15.0,
        high_delay_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.threshold = high_delay_threshold
        self.high_weight = high_delay_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calcula la pérdida MSE ponderada.

        Args:
            predictions: Predicciones del modelo.
            targets: Valores reales.

        Returns:
            Pérdida escalar.
        """
        se = (predictions - targets) ** 2

        # Asignar pesos: más peso a retrasos altos
        weights = torch.ones_like(targets)
        weights[targets.abs() > self.threshold] = self.high_weight

        return (se * weights).mean()
