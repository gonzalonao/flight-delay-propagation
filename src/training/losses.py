"""Loss functions for delay prediction models.

Includes standard losses and weighted variants for the multi-horizon
prediction problem. Works with 1D tensors [N] (single-horizon), 2D [N, H]
(multi-horizon single-task) and 3D [N, H, C] (multi-horizon multi-task
introduced in W2 — channels arr_delay, dep_delay_aux,
pct_arr_delayed_15_logit).

The single-task losses (``WeightedMSELoss``, ``WeightedHuberLoss``)
tolerate a target arriving with an extra channel and keep channel 0
(ArrDelay). This prevents updating the target shape in the graph_builder
from breaking models that still have ``output_channels=1``.
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
    """Shared base for losses weighted by delay and horizon.

    Combines two kinds of weighting over the pointwise error:

    1. **Delay weighting**: samples with ``|target| ≥ threshold`` receive
       ``high_delay_weight``× more weight, compensating for the imbalance
       between delayed and on-time airports.
    2. **Horizon weighting** (multi-horizon only, 2D tensors): each horizon
       receives its own weight, allowing the business horizons (4-8 h) to
       be prioritized over the short ones (1-2 h).

    Subclasses only define the *pointwise error* (MSE, Huber, etc.) via
    ``_pointwise_loss``; the weighting and averaging scheme is shared.

    Compatible with single-horizon [N] and multi-horizon [N, H].

    Args:
        high_delay_threshold: Threshold in minutes to consider a high delay.
        high_delay_weight: Multiplier weight for high-delay samples.
        horizon_weights: Per-horizon weights (e.g., ``[3, 3, 4, 5, 5]``).
            Renormalized to preserve the absolute scale of the loss. If
            ``None``, all horizons weigh the same.
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
            # Normalize so they sum to len(horizon_weights): this keeps the
            # weighted mean on a scale comparable to the unweighted mean
            # (useful for comparing loss curves across configurations).
            hw = torch.tensor(horizon_weights, dtype=torch.float32)
            hw = hw * len(horizon_weights) / hw.sum()
            self.register_buffer("horizon_weights", hw)
        else:
            self.horizon_weights = None

    def _pointwise_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Pointwise error without reduction (same shape as ``targets``)."""
        raise NotImplementedError

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Apply delay and horizon weighting over the pointwise error.

        If ``targets`` has 3 dimensions ``[N, H, C]`` (W2 multi-task
        format) and ``predictions`` has 2 ``[N, H]``, the target is reduced
        to channel 0 (ArrDelay). This lets a single-task model keep
        training on multi-channel data without the user having to pre-slice
        it.
        """
        if predictions.dim() == 2 and targets.dim() == 3:
            targets = targets[..., TARGET_CHANNEL_ARR_DELAY]

        err = self._pointwise_loss(predictions, targets)

        # Delay weight: more weight on high delays.
        weights = torch.ones_like(targets)
        delayed_mask = targets.abs() >= self.threshold
        weights[delayed_mask] = self.high_weight

        # Horizon weight: only applies to 2D tensors [N, H].
        if self.horizon_weights is not None and targets.dim() == 2:
            hw = self.horizon_weights.to(targets.device)
            weights = weights * hw.unsqueeze(0)  # [1, H] broadcast

        return (err * weights).mean()


class WeightedMSELoss(_WeightedRegressionLossBase):
    """Weighted MSE (delay + horizon). Kept for compatibility.

    The recommended default loss since commit I is ``WeightedHuberLoss``
    (more robust to the long tail of delays), but the MSE variant is kept
    for ablation and backward compatibility with old configs.
    """

    def _pointwise_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return (predictions - targets) ** 2


class WeightedHuberLoss(_WeightedRegressionLossBase):
    """Weighted Huber (delay + horizon), robust to outliers.

    Delays have a very skewed distribution: most are <30 min but the tail
    reaches several hours. MSE amplifies those outliers (the gradient grows
    linearly with the error) and ends up overfitting the tails; Huber is
    quadratic for small errors and linear for large ones, stabilizing the
    gradient without losing sensitivity in the relevant regime.

    Args:
        delta: Quadratic→linear transition point (in the same units as the
            target, minutes). δ=10 means errors of ±10 min are penalized
            as MSE and larger errors grow linearly.
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
        return F.huber_loss(predictions, targets, reduction="none", delta=self.delta)


class MultiTaskLoss(nn.Module):
    """Combined loss for the 3 channels of the W2 multi-task head.

    Expects ``predictions`` and ``targets`` of shape ``[N, H, 3]`` with the
    channel contract documented in ``src.data.graph_builder``:

      0) arr_delay (min)               — primary regression, weighted Huber
      1) dep_delay (min)               — auxiliary regression, weighted Huber
      2) pct_arr_delayed_15 ∈ [0, 1]   — classification, BCEWithLogits

    The horizon weights are applied to all three tasks to favor the
    business horizons (4-8 h). The high-delay threshold applies only to the
    two regression losses — for BCE it makes no sense (the target is
    already the fraction of delayed).

    Each component is logged as the ``last_components`` attribute after the
    forward, useful for introspection in the trainer / TensorBoard.

    Args:
        main_weight: Weight of the ArrDelay loss (channel 0).
        aux_weight: Weight of the auxiliary DepDelay loss (channel 1).
        bce_weight: Weight of the BCE loss (channel 2).
        high_delay_threshold: Threshold in min for the weighted regressions.
        high_delay_weight: Multiplier for delayed samples.
        horizon_weights: Per-horizon weights (same shape for the 3 heads).
        huber_delta: Huber delta for the two regressions.
        bce_pos_weight: Positive-class weight in BCE — useful when the
            binary target is imbalanced (majority of on-time airports).
            ``None`` disables the rebalancing. If given, it is used as the
            ``pos_weight`` of ``BCEWithLogitsLoss``.

    Raises:
        ValueError: If all weights are zero (the loss would be trivially 0).
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
    ) -> None:
        super().__init__()
        if main_weight + aux_weight + bce_weight <= 0:
            raise ValueError(
                "MultiTaskLoss: at least one of "
                "(main_weight, aux_weight, bce_weight) must be > 0."
            )

        self.main_weight = float(main_weight)
        self.aux_weight = float(aux_weight)
        self.bce_weight = float(bce_weight)
        self.delta = huber_delta
        self.threshold = high_delay_threshold
        self.high_weight = high_delay_weight

        if horizon_weights is not None:
            hw = torch.tensor(horizon_weights, dtype=torch.float32)
            hw = hw * len(horizon_weights) / hw.sum()
            self.register_buffer("horizon_weights", hw)
        else:
            self.horizon_weights = None

        if bce_pos_weight is not None:
            self.register_buffer("bce_pos_weight", torch.tensor(float(bce_pos_weight)))
        else:
            self.bce_pos_weight = None

        # Snapshot of the last forward for logging — does not affect the graph.
        self.last_components: dict[str, float] = {}

    def _weighted_huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Huber weighted by high delay + horizon. Returns a scalar."""
        err = F.huber_loss(pred, target, reduction="none", delta=self.delta)
        weights = torch.ones_like(target)
        weights[target.abs() >= self.threshold] = self.high_weight
        if self.horizon_weights is not None and target.dim() == 2:
            hw = self.horizon_weights.to(target.device)
            weights = weights * hw.unsqueeze(0)
        return (err * weights).mean()

    def _weighted_bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Horizon-weighted BCEWithLogits (target is already ∈ [0, 1])."""
        pos_weight = self.bce_pos_weight if self.bce_pos_weight is not None else None
        err = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=pos_weight,
        )
        if self.horizon_weights is not None and target.dim() == 2:
            hw = self.horizon_weights.to(target.device)
            err = err * hw.unsqueeze(0)
        return err.mean()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combine the 3 sub-losses with configurable weights."""
        if predictions.dim() != 3 or targets.dim() != 3:
            raise ValueError(
                "MultiTaskLoss expects 3D tensors [N, H, C]; "
                f"received predictions={tuple(predictions.shape)}, "
                f"targets={tuple(targets.shape)}."
            )
        if (
            predictions.size(-1) < NUM_TARGET_CHANNELS
            or targets.size(-1) < NUM_TARGET_CHANNELS
        ):
            raise ValueError(
                f"Last dim must be >= {NUM_TARGET_CHANNELS} (arr, dep, pct); "
                f"received pred={predictions.size(-1)}, "
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
            # The pct channel target lives in [0, 1]; we clamp it for
            # numerical safety (it should not be outside that range).
            targets[..., TARGET_CHANNEL_PCT_DELAYED].clamp(0.0, 1.0),
        )

        total = (
            self.main_weight * arr_loss
            + self.aux_weight * dep_loss
            + self.bce_weight * bce_loss
        )

        # Scalar snapshot for logging — not used in backward.
        self.last_components = {
            "arr_huber": float(arr_loss.detach()),
            "dep_huber": float(dep_loss.detach()),
            "pct_bce": float(bce_loss.detach()),
            "total": float(total.detach()),
        }

        return total
