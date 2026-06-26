"""Trainers for GNN models based on PyTorch Geometric.

Adapt :class:`BaseTrainer` to work with PyG graphs (Data objects),
handling ``edge_index``, ``edge_attr`` and active-node masks.

Includes:

- :class:`GraphTrainer`: for models that process individual graphs
  (BasicGCN, MultiHorizonGAT).
- :class:`SequenceGraphTrainer`: for models that process sequences of
  temporal graphs (SpatioTemporalGNN, Seq2SeqGNN).

After the W4.3 collapse both share ``BaseTrainer`` and only contribute the
logic specific to their batch type (a single graph vs a sequence).
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.data.graph_builder import TARGET_CHANNEL_ARR_DELAY
from src.training.base_trainer import BaseTrainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _align_for_val_channel0(
    predictions: torch.Tensor, targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align pred/target to the ArrDelay channel for validation.

    The standard val_criterion (MSE) compares ArrDelay minutes so that the
    ``val_loss`` is comparable across single-task and multi-task models
    (W2). When the model or target arrives in multi-channel format
    ``[..., 3]`` we extract channel 0 (ArrDelay). The other regression
    channel (DepDelay) and the BCE channel are evaluated via dedicated
    metrics (``compute_classification_metrics_bce`` etc.), not here.
    """
    if predictions.dim() == 3:
        predictions = predictions[..., TARGET_CHANNEL_ARR_DELAY]
    if targets.dim() == 3:
        targets = targets[..., TARGET_CHANNEL_ARR_DELAY]
    return predictions, targets


# Backward-compatible alias: there may be tests or external utilities still
# importing ``_align_for_val`` under its original name.
_align_for_val = _align_for_val_channel0


class GraphTrainer(BaseTrainer):
    """Trainer for GNN models over individual temporal graphs.

    Args:
        model: PyTorch GNN model.
        optimizer: Optimizer.
        criterion: Loss function for training (may be weighted).
        device: Device (cpu/cuda).
        scheduler: Learning rate scheduler (optional).
        gradient_clip: Maximum gradient value (0 = no clip).
        val_criterion: Loss function for validation. If ``None``, standard
            ``MSELoss`` is used so that ``val_loss`` is comparable across
            single-task and multi-task runs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        gradient_clip: float = 0.0,
        val_criterion: nn.Module | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            gradient_clip=gradient_clip,
            val_criterion=val_criterion if val_criterion is not None else nn.MSELoss(),
        )

    def _forward_batch(
        self, batch: Data,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move a graph to the device and run the forward pass.

        For single-horizon models that emit ``[N, 1]`` it applies
        ``squeeze(-1)`` to keep compatibility with the rest of the
        pipeline. Multi-horizon ``[N, H]`` and multi-task ``[N, H, 3]``
        pass through as-is.
        """
        graph = batch
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_attr = (
            graph.edge_attr.to(self.device) if graph.edge_attr is not None else None
        )
        y = graph.y.to(self.device)
        active_mask = graph.active_mask.to(self.device)

        logits = self.model(x, edge_index, edge_attr=edge_attr)
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(-1)

        return logits, y, active_mask

    def _align_for_val(
        self, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _align_for_val_channel0(predictions, targets)

    # Wrappers to preserve the public API (train_graphs/val_graphs).

    def fit(  # type: ignore[override]
        self,
        train_graphs: list[Data],
        val_graphs: list[Data],
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, list[float]]:
        return super().fit(
            train_data=train_graphs,
            val_data=val_graphs,
            epochs=epochs,
            patience=patience,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
        )

    def train_epoch(self, graphs: list[Data]) -> float:  # type: ignore[override]
        return super().train_epoch(graphs)

    def validate(self, graphs: list[Data]) -> float:  # type: ignore[override]
        return super().validate(graphs)


class SequenceGraphTrainer(BaseTrainer):
    """Trainer for models that process sequences of temporal graphs.

    Designed for SpatioTemporalGNN, Seq2SeqGNN and similar, where each
    sample is a sequence of ``input_window`` consecutive graphs. The target
    and mask are taken from the last graph in the sequence.

    Args:
        model: Model that accepts ``list[Data]`` in its ``forward()``.
        optimizer: Optimizer.
        criterion: Loss function for training.
        device: Device (cpu/cuda).
        scheduler: Learning rate scheduler (optional).
        gradient_clip: Maximum gradient value (0 = no clip).
        val_criterion: Loss function for validation. If ``None``, standard
            ``MSELoss`` is used for comparable metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        gradient_clip: float = 0.0,
        val_criterion: nn.Module | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            gradient_clip=gradient_clip,
            val_criterion=val_criterion if val_criterion is not None else nn.MSELoss(),
        )

    def _move_graph_to_device(self, graph: Data) -> Data:
        """Clone a graph and move all its tensors to the device."""
        graph = graph.clone()
        graph.x = graph.x.to(self.device)
        graph.edge_index = graph.edge_index.to(self.device)
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr.to(self.device)
        if graph.y is not None:
            graph.y = graph.y.to(self.device)
        if graph.active_mask is not None:
            graph.active_mask = graph.active_mask.to(self.device)
        return graph

    def _forward_batch(
        self, batch: list[Data],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move a sequence to the device and run the forward pass."""
        seq_on_device = [self._move_graph_to_device(g) for g in batch]
        predictions = self.model(seq_on_device)
        last_graph = seq_on_device[-1]
        return predictions, last_graph.y, last_graph.active_mask

    def _align_for_val(
        self, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _align_for_val_channel0(predictions, targets)

    # Wrappers to preserve the public API (train_sequences/val_sequences).

    def fit(  # type: ignore[override]
        self,
        train_sequences: list[list[Data]],
        val_sequences: list[list[Data]],
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, list[float]]:
        return super().fit(
            train_data=train_sequences,
            val_data=val_sequences,
            epochs=epochs,
            patience=patience,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
        )

    def train_epoch(self, sequences: list[list[Data]]) -> float:  # type: ignore[override]
        return super().train_epoch(sequences)

    def validate(self, sequences: list[list[Data]]) -> float:  # type: ignore[override]
        return super().validate(sequences)
