"""Training base shared by all of the project's trainers.

Concentrates the train/val/epochs/scheduler/early-stopping/checkpoint
cycle that used to be repeated across :class:`Trainer`,
:class:`GraphTrainer` and :class:`SequenceGraphTrainer`. Subclasses only
need to implement ``_forward_batch`` and, optionally, ``_align_for_val``.

Design (W4.3):

- ``_forward_batch(batch)`` returns ``(predictions, targets, mask | None)``.
  The mask is ``None`` for tabular data and a boolean tensor over the
  active nodes for graph data.
- ``_align_for_val(pred, target)`` is a hook subclasses override so that
  ``val_loss`` is comparable across single-task and multi-task runs (see
  W2): by default it is the identity.
- ``fit(train_data, val_data, ...)`` orchestrates the full cycle and
  delegates each step to ``train_epoch`` / ``validate``.

Subclasses:

- :class:`src.training.trainer.Trainer` for tabular data (DataLoader).
- :class:`src.training.graph_trainer.GraphTrainer` for graph snapshots.
- :class:`src.training.graph_trainer.SequenceGraphTrainer` for sequences
  of temporal graphs.
"""

from typing import Any, Iterable

import torch
import torch.nn as nn

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseTrainer:
    """Common template for all of the project's trainers.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        criterion: Loss function for training.
        device: Device (cpu/cuda).
        scheduler: Learning rate scheduler (optional).
        gradient_clip: Maximum gradient value (0 = no clip).
        val_criterion: Loss function for validation. If ``None``,
            ``criterion`` is used (historical tabular behavior). Graph
            subclasses pass ``MSELoss()`` so that val_loss is comparable
            across runs.
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
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_criterion = val_criterion if val_criterion is not None else criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

    # ------------------------------------------------------------------
    # Hooks that subclasses override
    # ------------------------------------------------------------------

    def _forward_batch(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Process a batch and return ``(pred, target, mask | None)``.

        Subclasses must implement this method to handle moving to the
        device, calling the model's ``forward`` and any shape
        normalization (for example ``squeeze(-1)``).

        If the subclass works with partially active graphs, it must return
        a boolean mask ``[N]``; the shared loop applies it. For tabular
        data, return ``None``.
        """
        raise NotImplementedError

    def _align_for_val(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook to align pred/target before the ``val_criterion``.

        Identity by default. Graph subclasses override it to extract the
        ArrDelay channel when the model emits multi-task outputs (see
        W2 / ``TARGET_CHANNEL_ARR_DELAY``).
        """
        return predictions, targets

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def _apply_mask(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Apply the active-node mask if present.

        Returns:
            Masked ``(pred, target)``, or ``None`` if the mask is present
            but empty (no active nodes in this batch).
        """
        if mask is None:
            return predictions, targets
        if mask.sum() == 0:
            return None
        return predictions[mask], targets[mask]

    def train_epoch(self, dataset: Iterable[Any]) -> float:
        """Run one training epoch.

        Args:
            dataset: Iterable of batches (DataLoader, list of graphs or
                list of sequences, depending on the subclass).

        Returns:
            Average loss of the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataset:
            predictions, targets, mask = self._forward_batch(batch)
            masked = self._apply_mask(predictions, targets, mask)
            if masked is None:
                continue
            pred_m, target_m = masked

            loss = self.criterion(pred_m, target_m)

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataset: Iterable[Any]) -> float:
        """Run validation without gradients.

        Returns:
            Average loss according to ``val_criterion``.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataset:
            predictions, targets, mask = self._forward_batch(batch)
            masked = self._apply_mask(predictions, targets, mask)
            if masked is None:
                continue
            pred_m, target_m = masked

            pred_v, target_v = self._align_for_val(pred_m, target_m)
            loss = self.val_criterion(pred_v, target_v)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_data: Iterable[Any],
        val_data: Iterable[Any],
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Full training cycle with early stopping and checkpointing.

        Args:
            train_data: Iterable of training batches.
            val_data: Iterable of validation batches.
            epochs: Maximum number of epochs.
            patience: Epochs without improvement before stopping.
            checkpoint_path: Path to save the best model (optional).
            model_name: Logical model name, persisted in the checkpoint so
                that ``load_checkpoint`` can validate the architecture.

        Returns:
            Dictionary with history ``{'train_loss', 'val_loss'}``.
        """
        early_stopping = EarlyStopping(patience=patience)
        checkpoint = (
            ModelCheckpoint(checkpoint_path, model_name=model_name)
            if checkpoint_path
            else None
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_data)
            val_loss = self.validate(val_data)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | LR: %.2e",
                epoch,
                epochs,
                train_loss,
                val_loss,
                self.optimizer.param_groups[0]["lr"],
            )

            if checkpoint is not None:
                checkpoint.step(val_loss, self.model, self.optimizer, epoch)

            if early_stopping.step(val_loss):
                logger.info("Early stopping at epoch %d", epoch)
                break

        return history
