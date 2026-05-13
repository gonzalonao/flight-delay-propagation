"""Bucle de entrenamiento tabular reutilizable para modelos densos.

Wrapper fino sobre :class:`BaseTrainer` que solo conoce el contrato de
DataLoader ``(features, targets)``. Toda la lógica de epochs, early
stopping, checkpoint y scheduler vive en la clase base (W4.3).
"""

from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Trainer(BaseTrainer):
    """Entrenador tabular para modelos densos (DenseNN, etc.).

    Args:
        model: Modelo de PyTorch que acepta un único tensor de features.
        optimizer: Optimizador.
        criterion: Función de pérdida (se reutiliza también en validación,
            comportamiento histórico).
        device: Dispositivo (cpu/cuda).
        scheduler: Learning rate scheduler (opcional).
        gradient_clip: Valor máximo de gradiente (0 = sin clip).
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
        self, batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Mueve un batch tabular al dispositivo y ejecuta el forward."""
        features, targets = batch
        features = features.to(self.device)
        targets = targets.to(self.device)
        predictions = self.model(features).squeeze(-1)
        return predictions, targets, None

    # Wrappers para preservar la API pública histórica (train_loader/val_loader).

    def fit(  # type: ignore[override]
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Entrena el modelo con DataLoaders tabulares."""
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
