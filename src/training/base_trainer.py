"""Base de entrenamiento compartida por todos los trainers del proyecto.

Concentra el ciclo train/val/epochs/scheduler/early-stopping/checkpoint
que antes se repetía en :class:`Trainer`, :class:`GraphTrainer` y
:class:`SequenceGraphTrainer`. Las subclases solo necesitan implementar
``_forward_batch`` y, opcionalmente, ``_align_for_val``.

Diseño (W4.3):

- ``_forward_batch(batch)`` devuelve ``(predictions, targets, mask | None)``.
  La máscara es ``None`` para datos tabulares y un tensor booleano sobre
  los nodos activos para datos de grafos.
- ``_align_for_val(pred, target)`` es un hook que las subclases sobreescriben
  para que el ``val_loss`` sea comparable entre runs single-task y
  multi-task (ver W2): por defecto es identidad.
- ``fit(train_data, val_data, ...)`` orquesta el ciclo completo y delega
  cada paso en ``train_epoch`` / ``validate``.

Subclases:

- :class:`src.training.trainer.Trainer` para datos tabulares (DataLoader).
- :class:`src.training.graph_trainer.GraphTrainer` para snapshots de grafos.
- :class:`src.training.graph_trainer.SequenceGraphTrainer` para secuencias
  de grafos temporales.
"""

from typing import Any, Iterable

import torch
import torch.nn as nn

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseTrainer:
    """Plantilla común para todos los entrenadores del proyecto.

    Args:
        model: Modelo de PyTorch.
        optimizer: Optimizador.
        criterion: Función de pérdida para entrenamiento.
        device: Dispositivo (cpu/cuda).
        scheduler: Learning rate scheduler (opcional).
        gradient_clip: Valor máximo de gradiente (0 = sin clip).
        val_criterion: Función de pérdida para validación. Si ``None``,
            se usa ``criterion`` (comportamiento tabular histórico). Las
            subclases de grafos pasan ``MSELoss()`` para que el val_loss
            sea comparable entre runs.
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
    # Hooks que las subclases sobreescriben
    # ------------------------------------------------------------------

    def _forward_batch(
        self, batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Procesa un batch y devuelve ``(pred, target, mask | None)``.

        Las subclases deben implementar este método para gestionar el
        movimiento al dispositivo, la llamada al ``forward`` del modelo y
        cualquier normalización de shapes (por ejemplo ``squeeze(-1)``).

        Si la subclase trabaja con grafos parcialmente activos, debe
        devolver una máscara booleana ``[N]``; el bucle común se encarga
        de aplicarla. Para datos tabulares, devolver ``None``.
        """
        raise NotImplementedError

    def _align_for_val(
        self, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook para alinear pred/target antes del ``val_criterion``.

        Por defecto, identidad. Las subclases de grafos lo sobreescriben
        para extraer el canal de ArrDelay cuando el modelo emite salidas
        multi-tarea (ver W2 / ``TARGET_CHANNEL_ARR_DELAY``).
        """
        return predictions, targets

    # ------------------------------------------------------------------
    # Lógica común
    # ------------------------------------------------------------------

    def _apply_mask(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Aplica la máscara de nodos activos si existe.

        Returns:
            ``(pred, target)`` enmascarados, o ``None`` si la máscara está
            presente pero vacía (no hay nodos activos en este batch).
        """
        if mask is None:
            return predictions, targets
        if mask.sum() == 0:
            return None
        return predictions[mask], targets[mask]

    def train_epoch(self, dataset: Iterable[Any]) -> float:
        """Ejecuta una época de entrenamiento.

        Args:
            dataset: Iterable de batches (DataLoader, lista de grafos o
                lista de secuencias, según la subclase).

        Returns:
            Pérdida promedio de la época.
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
                    self.model.parameters(), self.gradient_clip,
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataset: Iterable[Any]) -> float:
        """Ejecuta validación sin gradientes.

        Returns:
            Pérdida promedio según ``val_criterion``.
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
        """Ciclo completo de entrenamiento con early stopping y checkpoint.

        Args:
            train_data: Iterable de batches de entrenamiento.
            val_data: Iterable de batches de validación.
            epochs: Número máximo de épocas.
            patience: Épocas sin mejora antes de parar.
            checkpoint_path: Ruta para guardar el mejor modelo (opcional).
            model_name: Nombre lógico del modelo, persistido en el
                checkpoint para que ``load_checkpoint`` pueda validar la
                arquitectura.

        Returns:
            Diccionario con historial ``{'train_loss', 'val_loss'}``.
        """
        early_stopping = EarlyStopping(patience=patience)
        checkpoint = (
            ModelCheckpoint(checkpoint_path, model_name=model_name)
            if checkpoint_path else None
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_data)
            val_loss = self.validate(val_data)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
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

            if checkpoint is not None:
                checkpoint.step(val_loss, self.model, self.optimizer, epoch)

            if early_stopping.step(val_loss):
                logger.info("Early stopping en época %d", epoch)
                break

        return history
