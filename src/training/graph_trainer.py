"""Bucle de entrenamiento para modelos GNN basados en PyTorch Geometric.

Adapta el Trainer genérico para trabajar con grafos PyG (Data objects),
manejando edge_index, edge_weight y máscaras de nodos activos.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphTrainer:
    """Entrenador para modelos GNN sobre grafos temporales.

    A diferencia del Trainer tabular, itera sobre una lista de grafos
    (snapshots temporales) en vez de DataLoader de tensores.

    Args:
        model: Modelo GNN de PyTorch.
        optimizer: Optimizador.
        criterion: Función de pérdida (MSELoss para regresión).
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

    def _process_graph(self, graph: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mueve un grafo al dispositivo y extrae componentes.

        Maneja tanto targets single-horizon [num_nodes] como
        multi-horizonte [num_nodes, num_horizons].

        Returns:
            Tupla de (logits, targets, active_mask) para nodos activos.
        """
        x = graph.x.to(self.device)
        edge_index = graph.edge_index.to(self.device)
        edge_weight = None
        if graph.edge_attr is not None:
            edge_weight = graph.edge_attr.squeeze(-1).to(self.device)
        y = graph.y.to(self.device)
        active_mask = graph.active_mask.to(self.device)

        logits = self.model(x, edge_index, edge_weight=edge_weight)

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

        Args:
            graphs: Lista de grafos temporales de validación.

        Returns:
            Pérdida promedio de validación.
        """
        self.model.eval()
        total_loss = 0.0
        n_graphs = 0

        for graph in graphs:
            logits, targets, mask = self._process_graph(graph)

            if mask.sum() == 0:
                continue

            loss = self.criterion(logits[mask], targets[mask])
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
    ) -> dict[str, list[float]]:
        """Ejecuta el ciclo completo de entrenamiento con early stopping.

        Args:
            train_graphs: Grafos de entrenamiento.
            val_graphs: Grafos de validación.
            epochs: Número máximo de épocas.
            patience: Épocas sin mejora antes de parar.
            checkpoint_path: Ruta para guardar el mejor modelo.

        Returns:
            Diccionario con historial de pérdidas {'train_loss', 'val_loss'}.
        """
        early_stopping = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(checkpoint_path) if checkpoint_path else None

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
