"""Bucle de entrenamiento genérico reutilizable para todos los modelos.

Gestiona el ciclo train/validation, logging de métricas, early stopping
y guardado de checkpoints. Diseñado para funcionar con cualquier modelo
que acepte batches de (features, targets).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Trainer:
    """Entrenador genérico para modelos de PyTorch.

    Args:
        model: Modelo de PyTorch.
        optimizer: Optimizador.
        criterion: Función de pérdida.
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
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Ejecuta una época de entrenamiento.

        Args:
            dataloader: DataLoader con datos de entrenamiento.

        Returns:
            Pérdida promedio de la época.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for features, targets in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features).squeeze(-1)
            loss = self.criterion(predictions, targets)
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Ejecuta validación sin gradientes.

        Args:
            dataloader: DataLoader con datos de validación.

        Returns:
            Pérdida promedio de validación.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for features, targets in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(features).squeeze(-1)
            loss = self.criterion(predictions, targets)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
    ) -> dict[str, list[float]]:
        """Ejecuta el ciclo completo de entrenamiento con early stopping.

        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validación.
            epochs: Número máximo de épocas.
            patience: Épocas sin mejora antes de parar.
            checkpoint_path: Ruta para guardar el mejor modelo (opcional).

        Returns:
            Diccionario con historial de pérdidas {'train_loss', 'val_loss'}.
        """
        early_stopping = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(checkpoint_path) if checkpoint_path else None

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Época %d/%d | Train Loss: %.4f | Val Loss: %.4f | LR: %.2e",
                epoch, epochs, train_loss, val_loss,
                self.optimizer.param_groups[0]["lr"],
            )

            # Guardar mejor modelo
            if checkpoint is not None:
                checkpoint.step(val_loss, self.model, self.optimizer, epoch)

            # Early stopping
            if early_stopping.step(val_loss):
                logger.info("Early stopping en época %d", epoch)
                break

        return history
