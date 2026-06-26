"""Reusable tabular training loop for dense models.

A thin wrapper over :class:`BaseTrainer` that only knows the DataLoader
contract ``(features, targets)``. All the epoch, early-stopping,
checkpoint and scheduler logic lives in the base class (W4.3).
"""

from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Trainer(BaseTrainer):
    """Tabular trainer for dense models (DenseNN, etc.).

    Args:
        model: PyTorch model that accepts a single feature tensor.
        optimizer: Optimizer.
        criterion: Loss function (also reused for validation, historical
            behavior).
        device: Device (cpu/cuda).
        scheduler: Learning rate scheduler (optional).
        gradient_clip: Maximum gradient value (0 = no clip).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        gradient_clip: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            gradient_clip=gradient_clip,
            val_criterion=criterion,
        )

    def _forward_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Move a tabular batch to the device and run the forward pass."""
        features, targets = batch
        features = features.to(self.device)
        targets = targets.to(self.device)
        predictions = self.model(features).squeeze(-1)
        return predictions, targets, None

    # Wrappers to preserve the historical public API (train_loader/val_loader).

    def fit(  # type: ignore[override]
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Train the model with tabular DataLoaders."""
        return super().fit(
            train_data=train_loader,
            val_data=val_loader,
            epochs=epochs,
            patience=patience,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
        )

    def train_epoch(self, dataloader: Iterable) -> float:  # type: ignore[override]
        return super().train_epoch(dataloader)

    def validate(self, dataloader: Iterable) -> float:  # type: ignore[override]
        return super().validate(dataloader)
