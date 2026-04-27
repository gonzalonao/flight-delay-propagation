"""Bucle de entrenamiento para modelos GNN basados en PyTorch Geometric.

Adapta el Trainer genérico para trabajar con grafos PyG (Data objects),
manejando edge_index, edge_weight y máscaras de nodos activos.

Incluye:
- ``GraphTrainer``: para modelos que procesan grafos individuales
  (BasicGCN, MultiHorizonGAT).
- ``SequenceGraphTrainer``: para modelos que procesan secuencias de
  grafos temporales (SpatioTemporalGNN).
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.data.graph_builder import TARGET_CHANNEL_ARR_DELAY
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _align_for_val(
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


class GraphTrainer:
    """Entrenador para modelos GNN sobre grafos temporales.

    A diferencia del Trainer tabular, itera sobre una lista de grafos
    (snapshots temporales) en vez de DataLoader de tensores.

    Args:
        model: Modelo GNN de PyTorch.
        optimizer: Optimizador.
        criterion: Función de pérdida para entrenamiento (puede ser ponderada).
        device: Dispositivo (cpu/cuda).
        scheduler: Learning rate scheduler (opcional).
        gradient_clip: Valor máximo de gradiente (0 = sin clip).
        val_criterion: Función de pérdida para validación. Si None, usa
            MSELoss estándar para que val_loss sea comparable entre runs.
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
        self.val_criterion = val_criterion or nn.MSELoss()
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

    def _process_graph(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mueve un grafo al dispositivo y extrae componentes.

        Maneja tanto targets single-horizon [num_nodes] como
        multi-horizonte [num_nodes, num_horizons].

        Returns:
            Tupla de (logits, targets, active_mask) para nodos activos.
        """
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_attr = (
            graph.edge_attr.to(self.device) if graph.edge_attr is not None else None
        )
        y = graph.y.to(self.device)
        active_mask = graph.active_mask.to(self.device)

        logits = self.model(x, edge_index, edge_attr=edge_attr)

        # Single-horizon: squeeze [N, 1] -> [N] para compatibilidad
        # Multi-horizon: mantener [N, H]
        if logits.dim() == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(-1)

        return logits, y, active_mask

    def train_epoch(self, graphs: list[Data]) -> float:
        """Ejecuta una época de entrenamiento sobre snapshots de grafos.

        Args:
            graphs: Lista de grafos temporales de entrenamiento.

        Returns:
            Pérdida promedio de la época.
        """
        self.model.train()
        total_loss = 0.0
        n_graphs = 0

        for graph in graphs:
            logits, targets, mask = self._process_graph(graph)

            # Solo calcular pérdida sobre nodos activos
            if mask.sum() == 0:
                continue

            loss = self.criterion(logits[mask], targets[mask])

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_graphs += 1

        return total_loss / max(n_graphs, 1)

    @torch.no_grad()
    def validate(self, graphs: list[Data]) -> float:
        """Ejecuta validación sin gradientes.

        Usa val_criterion (MSELoss estándar por defecto) para que las
        métricas de validación sean comparables independientemente de
        la función de pérdida usada en entrenamiento. Cuando el modelo
        o el target son multi-canal (W2), el val_loss se calcula sólo
        sobre el canal de ArrDelay para mantener la métrica comparable
        con runs single-task históricos.

        Args:
            graphs: Lista de grafos temporales de validación.

        Returns:
            Pérdida promedio de validación (MSE estándar sobre ArrDelay).
        """
        self.model.eval()
        total_loss = 0.0
        n_graphs = 0

        for graph in graphs:
            logits, targets, mask = self._process_graph(graph)

            if mask.sum() == 0:
                continue

            pred_v, target_v = _align_for_val(logits[mask], targets[mask])
            loss = self.val_criterion(pred_v, target_v)
            total_loss += loss.item()
            n_graphs += 1

        return total_loss / max(n_graphs, 1)

    def fit(
        self,
        train_graphs: list[Data],
        val_graphs: list[Data],
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Ejecuta el ciclo completo de entrenamiento con early stopping.

        Args:
            train_graphs: Grafos de entrenamiento.
            val_graphs: Grafos de validación.
            epochs: Número máximo de épocas.
            patience: Épocas sin mejora antes de parar.
            checkpoint_path: Ruta para guardar el mejor modelo.
            model_name: Nombre lógico del modelo, persistido en el checkpoint
                para validación de arquitectura al cargar.

        Returns:
            Diccionario con historial de pérdidas {'train_loss', 'val_loss'}.
        """
        early_stopping = EarlyStopping(patience=patience)
        checkpoint = (
            ModelCheckpoint(checkpoint_path, model_name=model_name)
            if checkpoint_path else None
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_graphs)
            val_loss = self.validate(val_graphs)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
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


class SequenceGraphTrainer:
    """Entrenador para modelos que procesan secuencias de grafos temporales.

    Diseñado para SpatioTemporalGNN y similares, donde cada muestra de
    entrenamiento es una secuencia de ``input_window`` grafos consecutivos.
    El target y la máscara de nodos activos se toman del último grafo
    de cada secuencia.

    Args:
        model: Modelo que acepta ``list[Data]`` en su ``forward()``.
        optimizer: Optimizador.
        criterion: Función de pérdida para entrenamiento.
        device: Dispositivo (cpu/cuda).
        scheduler: Learning rate scheduler (opcional).
        gradient_clip: Valor máximo de gradiente (0 = sin clip).
        val_criterion: Función de pérdida para validación. Si None, usa
            MSELoss estándar para métricas comparables.
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
        self.val_criterion = val_criterion or nn.MSELoss()
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip

    def _move_graph_to_device(self, graph: Data) -> Data:
        """Mueve los tensores de un grafo al dispositivo de entrenamiento.

        Returns:
            Grafo con tensores en ``self.device``.
        """
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

    def _process_sequence(
        self, sequence: list[Data],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mueve una secuencia al dispositivo y ejecuta el modelo.

        Args:
            sequence: Lista de grafos PyG (un input_window).

        Returns:
            Tupla de (predictions, targets, active_mask).
        """
        seq_on_device = [self._move_graph_to_device(g) for g in sequence]

        predictions = self.model(seq_on_device)

        # Targets y máscara del último grafo de la secuencia
        last_graph = seq_on_device[-1]
        targets = last_graph.y
        active_mask = last_graph.active_mask

        return predictions, targets, active_mask

    def train_epoch(self, sequences: list[list[Data]]) -> float:
        """Ejecuta una época de entrenamiento sobre secuencias de grafos.

        Args:
            sequences: Lista de secuencias temporales.

        Returns:
            Pérdida promedio de la época.
        """
        self.model.train()
        total_loss = 0.0
        n_seqs = 0

        for sequence in sequences:
            predictions, targets, mask = self._process_sequence(sequence)

            if mask.sum() == 0:
                continue

            loss = self.criterion(predictions[mask], targets[mask])

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_seqs += 1

        return total_loss / max(n_seqs, 1)

    @torch.no_grad()
    def validate(self, sequences: list[list[Data]]) -> float:
        """Ejecuta validación sin gradientes.

        Usa val_criterion (MSELoss estándar por defecto) para métricas
        comparables independientemente de la pérdida de entrenamiento.
        Cuando el modelo o el target son multi-canal (W2), el val_loss
        se calcula sólo sobre el canal de ArrDelay (channel 0) para
        comparabilidad histórica.

        Args:
            sequences: Lista de secuencias temporales de validación.

        Returns:
            Pérdida promedio de validación (MSE estándar sobre ArrDelay).
        """
        self.model.eval()
        total_loss = 0.0
        n_seqs = 0

        for sequence in sequences:
            predictions, targets, mask = self._process_sequence(sequence)

            if mask.sum() == 0:
                continue

            pred_v, target_v = _align_for_val(predictions[mask], targets[mask])
            loss = self.val_criterion(pred_v, target_v)
            total_loss += loss.item()
            n_seqs += 1

        return total_loss / max(n_seqs, 1)

    def fit(
        self,
        train_sequences: list[list[Data]],
        val_sequences: list[list[Data]],
        epochs: int = 100,
        patience: int = 15,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Ejecuta el ciclo completo de entrenamiento con early stopping.

        Args:
            train_sequences: Secuencias de entrenamiento.
            val_sequences: Secuencias de validación.
            epochs: Número máximo de épocas.
            patience: Épocas sin mejora antes de parar.
            checkpoint_path: Ruta para guardar el mejor modelo.
            model_name: Nombre lógico del modelo, persistido en el checkpoint
                para validación de arquitectura al cargar.

        Returns:
            Diccionario con historial {'train_loss', 'val_loss'}.
        """
        early_stopping = EarlyStopping(patience=patience)
        checkpoint = (
            ModelCheckpoint(checkpoint_path, model_name=model_name)
            if checkpoint_path else None
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_sequences)
            val_loss = self.validate(val_sequences)

            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
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
