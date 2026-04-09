"""Callbacks para el bucle de entrenamiento.

Implementa Early Stopping y guardado de checkpoints del mejor modelo.
"""

import torch
import torch.nn as nn

from src.utils.io import save_checkpoint
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EarlyStopping:
    """Detiene el entrenamiento si la métrica de validación no mejora.

    Args:
        patience: Épocas a esperar sin mejora antes de detener.
        min_delta: Mejora mínima para considerar que hubo progreso.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def step(self, val_loss: float) -> bool:
        """Evalúa si debe detenerse el entrenamiento.

        Args:
            val_loss: Pérdida de validación actual.

        Returns:
            True si debe detenerse, False si debe continuar.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class ModelCheckpoint:
    """Guarda el modelo cuando la pérdida de validación mejora.

    Args:
        path: Ruta donde guardar el checkpoint.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.best_loss: float | None = None

    def step(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """Guarda el modelo si la pérdida mejora.

        Args:
            val_loss: Pérdida de validación actual.
            model: Modelo a guardar.
            optimizer: Optimizador a guardar.
            epoch: Época actual.
        """
        if self.path is None:
            return

        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss},
                path=self.path,
            )
            logger.info("  Mejor modelo guardado (val_loss=%.4f)", val_loss)
