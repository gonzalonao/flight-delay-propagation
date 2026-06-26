"""Callbacks for the training loop.

Implements Early Stopping and saving of the best-model checkpoint.
"""

import torch
import torch.nn as nn

from src.utils.io import save_checkpoint
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EarlyStopping:
    """Stop training if the validation metric does not improve.

    Args:
        patience: Epochs to wait without improvement before stopping.
        min_delta: Minimum improvement to consider that progress was made.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def step(self, val_loss: float) -> bool:
        """Evaluate whether training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if it should stop, False if it should continue.
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
    """Save the model when the validation loss improves.

    Args:
        path: Path where the checkpoint is saved.
        model_name: Logical model identifier (e.g.,
            ``"multi_horizon_gat"``). If provided, it is persisted in the
            checkpoint so that ``load_checkpoint`` can validate the
            architecture on load and give clear errors if the evaluation
            ``--config`` does not match.
    """

    def __init__(
        self,
        path: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.path = path
        self.model_name = model_name
        self.best_loss: float | None = None

    def step(
        self,
        val_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """Save the model if the loss improves.

        Args:
            val_loss: Current validation loss.
            model: Model to save.
            optimizer: Optimizer to save.
            epoch: Current epoch.
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
                model_name=self.model_name,
            )
            logger.info("  Best model saved (val_loss=%.4f)", val_loss)
