"""Funciones de pérdida para modelos de predicción de retrasos.

Incluye pérdidas estándar y variantes ponderadas para el problema
de predicción multi-horizonte. Funciona con tensores 1D [N] (single-
horizon) y 2D [N, H] (multi-horizonte).
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE ponderado que penaliza más los errores en retrasos significativos.

    Combina dos tipos de ponderación:
    1. **Delay weighting**: nodos con delay ≥ umbral reciben más peso,
       compensando el desbalance entre aeropuertos retrasados y puntuales.
    2. **Horizon weighting** (solo multi-horizonte): horizontes cercanos
       reciben más peso que los lejanos, priorizando predicciones a corto
       plazo que son más precisas y accionables.

    Compatible con single-horizon [N] y multi-horizonte [N, H].

    Args:
        high_delay_threshold: Umbral en minutos para considerar retraso alto.
        high_delay_weight: Peso multiplicador para muestras con retraso alto.
        horizon_weights: Pesos por horizonte (e.g., [5, 4, 3, 2, 1]).
            Si None, todos los horizontes pesan igual.
    """

    def __init__(
        self,
        high_delay_threshold: float = 15.0,
        high_delay_weight: float = 2.0,
        horizon_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.threshold = high_delay_threshold
        self.high_weight = high_delay_weight

        if horizon_weights is not None:
            # Normalizar para que sumen 1 * num_horizons (media ponderada)
            hw = torch.tensor(horizon_weights, dtype=torch.float32)
            hw = hw * len(horizon_weights) / hw.sum()
            self.register_buffer("horizon_weights", hw)
        else:
            self.horizon_weights = None

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

        # Peso por delay: más peso a retrasos altos
        weights = torch.ones_like(targets)
        delayed_mask = targets.abs() >= self.threshold
        weights[delayed_mask] = self.high_weight

        # Peso por horizonte: más peso a horizontes cercanos
        if self.horizon_weights is not None and targets.dim() == 2:
            # horizon_weights shape [H] -> broadcast a [1, H]
            hw = self.horizon_weights.to(targets.device)
            weights = weights * hw.unsqueeze(0)

        return (se * weights).mean()
