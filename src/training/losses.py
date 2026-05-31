"""Funciones de pérdida para modelos de predicción de retrasos.

Incluye pérdidas estándar y variantes ponderadas para el problema
de predicción multi-horizonte. Funciona con tensores 1D [N]
(single-horizon), 2D [N, H] (multi-horizonte single-task) y 3D [N, H, C]
(multi-horizonte multi-task introducido en W2 — canales arr_delay,
dep_delay_aux, pct_arr_delayed_15_logit).

Las pérdidas single-task (``WeightedMSELoss``, ``WeightedHuberLoss``)
toleran que el target llegue con un canal extra y se quedan con el
canal 0 (ArrDelay). Esto evita que actualizar el shape del target en el
graph_builder rompa modelos que aún tienen ``output_channels=1``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.graph_builder import (
    NUM_TARGET_CHANNELS,
    TARGET_CHANNEL_ARR_DELAY,
    TARGET_CHANNEL_DEP_DELAY,
    TARGET_CHANNEL_PCT_DELAYED,
)


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
        """Aplica ponderación por delay y horizonte sobre el error puntual.

        Si ``targets`` tiene 3 dimensiones ``[N, H, C]`` (formato W2
        multi-task) y ``predictions`` tiene 2 ``[N, H]``, se reduce el
        target al canal 0 (ArrDelay). Esto permite que un modelo
        single-task siga entrenándose sobre datos multi-canal sin que el
        usuario tenga que pre-recortarlos.
        """
        if predictions.dim() == 2 and targets.dim() == 3:
            targets = targets[..., TARGET_CHANNEL_ARR_DELAY]

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


class MultiTaskLoss(nn.Module):
    """Pérdida combinada para los 3 canales del head multi-tarea de W2.

    Espera ``predictions`` y ``targets`` de shape ``[N, H, 3]`` con el
    contrato de canales documentado en ``src.data.graph_builder``:

      0) arr_delay (min)               — regresión primaria, Huber ponderado
      1) dep_delay (min)               — regresión auxiliar, Huber ponderado
      2) pct_arr_delayed_15 ∈ [0, 1]   — clasificación, BCEWithLogits

    Las ponderaciones por horizonte se aplican a las tres tareas para
    favorecer los horizontes de negocio (4-8 h). El umbral de retraso
    alto se aplica sólo a las dos pérdidas de regresión — para BCE no
    tiene sentido (el target ya es la fracción de delayed).

    Cada componente se loguea como atributo ``last_components`` tras el
    forward, útil para introspección en el trainer / TensorBoard.

    Args:
        main_weight: Peso de la pérdida de ArrDelay (canal 0).
        aux_weight: Peso de la pérdida de DepDelay auxiliar (canal 1).
        bce_weight: Peso de la pérdida BCE (canal 2).
        high_delay_threshold: Umbral en min para regresiones ponderadas.
        high_delay_weight: Multiplicador para muestras retrasadas.
        horizon_weights: Pesos por horizonte (mismo shape para los 3 heads).
        huber_delta: Delta de Huber para las dos regresiones.
        bce_pos_weight: Peso de la clase positiva en BCE — útil cuando el
            target binario está desbalanceado (mayoría de aeropuertos
            puntuales). ``None`` desactiva el rebalanceo. Si se da, se
            usa como ``pos_weight`` de ``BCEWithLogitsLoss``.
        bce_focal_gamma: Si se da, aplica modulación focal a la BCE: cada
            término se pondera por ``(1 - p_t)**gamma`` donde ``p_t`` es la
            probabilidad asignada a la clase verdadera. Concentra el gradiente
            en los positivos difíciles (raros) — complementa a
            ``bce_pos_weight``. ``None`` ⇒ BCE estándar (sin cambio de
            comportamiento). Valores típicos 1.5–2.0.
        bce_focal_alpha: Balanceo focal opcional ∈ (0, 1): pondera positivos
            por ``alpha`` y negativos por ``1 - alpha``. ``None`` ⇒ sin
            balanceo de clase en el término focal. Sólo tiene efecto si
            ``bce_focal_gamma`` está activo.

    Raises:
        ValueError: Si todos los pesos son cero (la pérdida sería trivialmente 0).
    """

    def __init__(
        self,
        main_weight: float = 1.0,
        aux_weight: float = 0.3,
        bce_weight: float = 0.5,
        high_delay_threshold: float = 15.0,
        high_delay_weight: float = 2.0,
        horizon_weights: list[float] | None = None,
        huber_delta: float = 10.0,
        bce_pos_weight: float | None = None,
        bce_focal_gamma: float | None = None,
        bce_focal_alpha: float | None = None,
    ) -> None:
        super().__init__()
        if main_weight + aux_weight + bce_weight <= 0:
            raise ValueError(
                "MultiTaskLoss: al menos uno de "
                "(main_weight, aux_weight, bce_weight) debe ser > 0."
            )

        self.main_weight = float(main_weight)
        self.aux_weight = float(aux_weight)
        self.bce_weight = float(bce_weight)
        self.delta = huber_delta
        self.threshold = high_delay_threshold
        self.high_weight = high_delay_weight
        self.focal_gamma = (
            float(bce_focal_gamma) if bce_focal_gamma is not None else None
        )
        self.focal_alpha = (
            float(bce_focal_alpha) if bce_focal_alpha is not None else None
        )

        if horizon_weights is not None:
            hw = torch.tensor(horizon_weights, dtype=torch.float32)
            hw = hw * len(horizon_weights) / hw.sum()
            self.register_buffer("horizon_weights", hw)
        else:
            self.horizon_weights = None

        if bce_pos_weight is not None:
            self.register_buffer(
                "bce_pos_weight", torch.tensor(float(bce_pos_weight))
            )
        else:
            self.bce_pos_weight = None

        # Snapshot del último forward para logging — no afecta al graph.
        self.last_components: dict[str, float] = {}

    def _weighted_huber(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Huber ponderado por delay alto + horizonte. Devuelve escalar."""
        err = F.huber_loss(pred, target, reduction="none", delta=self.delta)
        weights = torch.ones_like(target)
        weights[target.abs() >= self.threshold] = self.high_weight
        if self.horizon_weights is not None and target.dim() == 2:
            hw = self.horizon_weights.to(target.device)
            weights = weights * hw.unsqueeze(0)
        return (err * weights).mean()

    def _weighted_bce(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """BCEWithLogits ponderado por horizonte (target ya es ∈ [0, 1]).

        Con ``focal_gamma`` activo aplica la modulación focal
        ``(1 - p_t)**gamma`` (y opcionalmente el balanceo ``alpha``) sobre el
        término puntual antes de la ponderación por horizonte.
        """
        pos_weight = (
            self.bce_pos_weight if self.bce_pos_weight is not None else None
        )
        err = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight,
        )

        if self.focal_gamma is not None:
            # p_t = prob. asignada a la clase verdadera. Para targets blandos
            # ∈ [0, 1] (fracción delayed) interpolamos: p_t = t·p + (1-t)·(1-p).
            p = torch.sigmoid(logits)
            p_t = target * p + (1.0 - target) * (1.0 - p)
            focal = (1.0 - p_t).clamp(min=0.0, max=1.0) ** self.focal_gamma
            if self.focal_alpha is not None:
                alpha_t = (
                    target * self.focal_alpha
                    + (1.0 - target) * (1.0 - self.focal_alpha)
                )
                focal = focal * alpha_t
            err = err * focal

        if self.horizon_weights is not None and target.dim() == 2:
            hw = self.horizon_weights.to(target.device)
            err = err * hw.unsqueeze(0)
        return err.mean()

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Combina las 3 sub-pérdidas con pesos configurables."""
        if predictions.dim() != 3 or targets.dim() != 3:
            raise ValueError(
                "MultiTaskLoss espera tensores 3D [N, H, C]; "
                f"recibido predictions={tuple(predictions.shape)}, "
                f"targets={tuple(targets.shape)}."
            )
        if predictions.size(-1) < NUM_TARGET_CHANNELS or targets.size(-1) < NUM_TARGET_CHANNELS:
            raise ValueError(
                f"Last dim debe ser >= {NUM_TARGET_CHANNELS} (arr, dep, pct); "
                f"recibido pred={predictions.size(-1)}, "
                f"target={targets.size(-1)}."
            )

        arr_loss = self._weighted_huber(
            predictions[..., TARGET_CHANNEL_ARR_DELAY],
            targets[..., TARGET_CHANNEL_ARR_DELAY],
        )
        dep_loss = self._weighted_huber(
            predictions[..., TARGET_CHANNEL_DEP_DELAY],
            targets[..., TARGET_CHANNEL_DEP_DELAY],
        )
        bce_loss = self._weighted_bce(
            predictions[..., TARGET_CHANNEL_PCT_DELAYED],
            # El target del canal pct vive en [0, 1]; lo clampeamos por
            # seguridad numérica (no debería estar fuera de ese rango).
            targets[..., TARGET_CHANNEL_PCT_DELAYED].clamp(0.0, 1.0),
        )

        total = (
            self.main_weight * arr_loss
            + self.aux_weight * dep_loss
            + self.bce_weight * bce_loss
        )

        # Snapshot escalar para logging — no se usa en backward.
        self.last_components = {
            "arr_huber": float(arr_loss.detach()),
            "dep_huber": float(dep_loss.detach()),
            "pct_bce": float(bce_loss.detach()),
            "total": float(total.detach()),
        }

        return total
