"""Graph Attention Network multi-horizonte para prediccion de retrasos.

Segundo modelo GNN del proyecto. Usa capas GATConv con atención
multi-cabeza para ponderar la importancia de las conexiones entre
aeropuertos. Predice el retraso promedio (en minutos) de cada aeropuerto
a múltiples horizontes futuros simultáneamente (e.g., 1h, 2h, ..., 5h).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MultiHorizonGAT(nn.Module):
    """GAT para predicción de retrasos multi-horizonte por aeropuerto.

    Arquitectura: N capas GATConv con atención multi-cabeza, BatchNorm,
    ELU y dropout, seguidas de una capa de proyección por horizonte.
    Cada capa de atención aprende qué conexiones entre aeropuertos
    son más relevantes para propagar información de retrasos.

    Args:
        input_dim: Número de features por nodo.
        hidden_channels: Dimensión de las capas ocultas (por cabeza).
        num_heads: Número de cabezas de atención.
        num_layers: Número de capas GATConv.
        num_horizons: Número de horizontes a predecir simultáneamente.
        dropout: Probabilidad de dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        num_horizons: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.num_horizons = num_horizons

        # Capas GAT
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Primera capa: input_dim -> hidden_channels * num_heads
        self.convs.append(
            GATConv(
                input_dim,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Capas intermedias: hidden_channels * num_heads -> hidden_channels * num_heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Última capa GAT: promedia las cabezas (concat=False)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Dimensión de salida de la última capa GAT
        final_dim = hidden_channels

        # Cabeza de predicción multi-horizonte
        self.horizon_head = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, num_horizons),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass sobre el grafo con atención.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Índices de aristas [2, num_edges].
            edge_weight: Pesos de aristas [num_edges] (opcional).
                GATConv los usa como edge_attr para modular la atención.

        Returns:
            Predicciones por nodo [num_nodes, num_horizons].
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_weight)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.horizon_head(x)
