"""Funciones de pérdida para modelos de predicción de retrasos.

Incluye pérdidas estándar y variantes ponderadas para el problema
de predicción multi-horizonte. Funciona con tensores 1D [N] (single-
horizon) y 2D [N, H] (multi-horizonte).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _WeightedRegressionLossBase(nn.Module):
    """Base común para pérdidas ponderadas por delay y horizonte.

    Combina dos tipos de ponderación sobre el error puntual:

    1. **Delay weighting**: muestras con ``|target| ≥ threshold`` reciben
       ``high_delay_weight``× más peso, compensando el desbalance entre
       aeropuertos retrasados y puntuales.
    2. **Horizon weighting** (solo multi-horizonte, tensores 2D): cada
       horizonte recibe su propio peso, permitiendo priorizar los
       horizontes de negocio (4-8 h) sobre los cortos (1-2 h).

    Las subclases sólo definen el *error puntual* (MSE, Huber, etc.) vía
    ``_pointwise_loss``; el esquema de ponderación y promedio es común.

    Compatible con single-horizon [N] y multi-horizonte [N, H].

    Args:
        high_delay_threshold: Umbral en minutos para considerar retraso alto.
        high_delay_weight: Peso multiplicador para muestras con retraso alto.
        horizon_weights: Pesos por horizonte (e.g., ``[3, 3, 4, 5, 5]``).
            Se renormalizan para preservar la escala absoluta de la pérdida.
            Si ``None``, todos los horizontes pesan igual.
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
            # Normalizar para que sumen len(horizon_weights): así la media
            # ponderada mantiene escala comparable a la media no ponderada
            # (útil para comparar curvas de loss entre configuraciones).
            hw = torch.tensor(horizon_weights, dtype=torch.float32)
            hw = hw * len(horizon_weights) / hw.sum()
            self.register_buffer("horizon_weights", hw)
        else:
            self.horizon_weights = None

    def _pointwise_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Error puntual sin reducir (mismo shape que ``targets``)."""
        raise NotImplementedError

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Aplica ponderación por delay y horizonte sobre el error puntual."""
        err = self._pointwise_loss(predictions, targets)

        # Peso por delay: más peso a retrasos altos.
        weights = torch.ones_like(targets)
        delayed_mask = targets.abs() >= self.threshold
        weights[delayed_mask] = self.high_weight

        # Peso por horizonte: sólo aplica a tensores 2D [N, H].
        if self.horizon_weights is not None and targets.dim() == 2:
            hw = self.horizon_weights.to(targets.device)
            weights = weights * hw.unsqueeze(0)  # [1, H] broadcast

        return (err * weights).mean()


class WeightedMSELoss(_WeightedRegressionLossBase):
    """MSE ponderado (delay + horizonte). Mantenido para compatibilidad.

    La pérdida por defecto recomendada desde commit I es ``WeightedHuberLoss``
    (más robusta a la cola larga de retrasos), pero se mantiene la variante
    MSE para ablación y retrocompatibilidad con configs antiguos.
    """

    def _pointwise_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return (predictions - targets) ** 2


class WeightedHuberLoss(_WeightedRegressionLossBase):
    """Huber ponderado (delay + horizonte), robusto a outliers.

    Los retrasos tienen una distribución muy asimétrica: la mayoría son
    <30 min pero la cola llega a varias horas. MSE amplifica esos outliers
    (el gradiente crece linealmente con el error) y termina sobreajustando
    las colas; Huber es cuadrático para errores pequeños y lineal para
    grandes, estabilizando el gradiente sin perder sensibilidad en el
    régimen relevante.

    Args:
        delta: Punto de transición cuadrática→lineal (en las mismas
            unidades que el target, minutos). δ=10 significa que errores
            de ±10 min se castigan como MSE y errores mayores crecen
            linealmente.
    """

    def __init__(
        self,
        high_delay_threshold: float = 15.0,
        high_delay_weight: float = 2.0,
        horizon_weights: list[float] | None = None,
        delta: float = 10.0,
    ) -> None:
        super().__init__(
            high_delay_threshold=high_delay_threshold,
            high_delay_weight=high_delay_weight,
            horizon_weights=horizon_weights,
        )
        self.delta = delta

    def _pointwise_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.huber_loss(
            predictions, targets, reduction="none", delta=self.delta
        )
