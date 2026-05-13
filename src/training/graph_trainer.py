"""Entrenadores para modelos GNN basados en PyTorch Geometric.

Adaptan :class:`BaseTrainer` para trabajar con grafos PyG (Data objects),
manejando ``edge_index``, ``edge_attr`` y máscaras de nodos activos.

Incluye:

- :class:`GraphTrainer`: para modelos que procesan grafos individuales
  (BasicGCN, MultiHorizonGAT).
- :class:`SequenceGraphTrainer`: para modelos que procesan secuencias de
  grafos temporales (SpatioTemporalGNN, Seq2SeqGNN).

Tras el colapso W4.3 ambos comparten ``BaseTrainer`` y solo aportan
la lógica específica de su tipo de batch (un grafo vs una secuencia).
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
    """Alinea pred/target al canal de ArrDelay para validación.

    El val_criterion estándar (MSE) compara minutos de ArrDelay para que
    el ``val_loss`` sea comparable entre modelos single-task y multi-task
    (W2). Cuando el modelo o el target llegan en formato multi-canal
    ``[..., 3]`` extraemos el canal 0 (ArrDelay). El otro canal de
    regresión (DepDelay) y el canal BCE se evalúan vía métricas
    dedicadas (``compute_classification_metrics_bce`` etc.), no aquí.
    """
    if predictions.dim() == 3:
        predictions = predictions[..., TARGET_CHANNEL_ARR_DELAY]
    if targets.dim() == 3:
        targets = targets[..., TARGET_CHANNEL_ARR_DELAY]
    return predictions, targets


# Alias retro-compatible: hay tests u utilidades externas que pueden estar
# importando ``_align_for_val`` con su nombre original.
_align_for_val = _align_for_val_channel0


class GraphTrainer(BaseTrainer):
    """Entrenador para modelos GNN sobre grafos temporales individuales.

    Args:
        model: Modelo GNN de PyTorch.
        optimizer: Optimizador.
        criterion: Función de pérdida para entrenamiento (puede ser ponderada).
        device: Dispositivo (cpu/cuda).
        scheduler: Learning rate scheduler (opcional).
        gradient_clip: Valor máximo de gradiente (0 = sin clip).
        val_criterion: Función de pérdida para validación. Si ``None``, se
            usa ``MSELoss`` estándar para que ``val_loss`` sea comparable
            entre runs single-task y multi-task.
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
        """Mueve un grafo al dispositivo y ejecuta el forward.

        Para modelos single-horizon que emiten ``[N, 1]`` aplica
        ``squeeze(-1)`` para mantener la compatibilidad con el resto del
        pipeline. Multi-horizon ``[N, H]`` y multi-task ``[N, H, 3]``
        pasan tal cual.
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

    # Wrappers para preservar la API pública (train_graphs/val_graphs).

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
    """Entrenador para modelos que procesan secuencias de grafos temporales.

    Diseñado para SpatioTemporalGNN, Seq2SeqGNN y similares, donde cada
    muestra es una secuencia de ``input_window`` grafos consecutivos. El
    target y la máscara se toman del último grafo de la secuencia.

    Args:
        model: Modelo que acepta ``list[Data]`` en su ``forward()``.
        optimizer: Optimizador.
        criterion: Función de pérdida para entrenamiento.
        device: Dispositivo (cpu/cuda).
        scheduler: Learning rate scheduler (opcional).
        gradient_clip: Valor máximo de gradiente (0 = sin clip).
        val_criterion: Función de pérdida para validación. Si ``None``, se
            usa ``MSELoss`` estándar para métricas comparables.
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
        """Clona un grafo y mueve todos sus tensores al dispositivo."""
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
        """Mueve una secuencia al dispositivo y ejecuta el forward."""
        seq_on_device = [self._move_graph_to_device(g) for g in batch]
        predictions = self.model(seq_on_device)
        last_graph = seq_on_device[-1]
        return predictions, last_graph.y, last_graph.active_mask

    def _align_for_val(
        self, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _align_for_val_channel0(predictions, targets)

    # Wrappers para preservar la API pública (train_sequences/val_sequences).

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
