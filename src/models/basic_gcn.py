"""Red de convolución sobre grafos (GCN) para clasificación binaria de retrasos.

Primer modelo GNN del proyecto. Opera sobre grafos de aeropuertos donde
los nodos representan aeropuertos y las aristas representan rutas aéreas.
Predice si cada aeropuerto tendrá un retraso significativo en la
siguiente ventana temporal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BasicGCN(nn.Module):
    """GCN para clasificación binaria de retrasos por aeropuerto.

    Arquitectura: N capas GCNConv con BatchNorm, ReLU y dropout,
    seguidas de una capa lineal de salida. Cada capa propaga
    información de los aeropuertos vecinos para capturar la
    propagación espacial de retrasos.

    Args:
        input_dim: Número de features por nodo.
        hidden_channels: Dimensión de las capas ocultas GCN.
        num_layers: Número de capas GCNConv.
        dropout: Probabilidad de dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        # Capas GCN
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Primera capa: input_dim -> hidden_channels
        self.convs.append(GCNConv(input_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Capas intermedias: hidden_channels -> hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Capa de salida: clasificación binaria por nodo
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass sobre el grafo.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Índices de aristas [2, num_edges].
            edge_weight: Pesos de aristas [num_edges] (opcional).

        Returns:
            Logits por nodo [num_nodes, 1].
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)
