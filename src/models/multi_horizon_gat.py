"""Graph Attention Network multi-horizonte para prediccion de retrasos.

Segundo modelo GNN del proyecto. Usa capas GATConv con atención
multi-cabeza para ponderar la importancia de las conexiones entre
aeropuertos. Predice el retraso promedio (en minutos) de cada aeropuerto
a múltiples horizontes futuros simultáneamente (e.g., 1h, 2h, ..., 5h).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class MultiHorizonGAT(nn.Module):
    """GAT para predicción de retrasos multi-horizonte por aeropuerto.

    Arquitectura: N capas ``GATv2Conv`` (con ``edge_dim`` para integrar
    features de arista multi-canal), BatchNorm, ELU y dropout, seguidas
    de una cabeza de proyección por horizonte. La atención aprende qué
    conexiones entre aeropuertos son más relevantes para propagar
    información de retrasos, ahora condicionada por las 5 features de
    arista (Class A + B) que produce ``graph_builder``.

    Args:
        input_dim: Número de features por nodo.
        hidden_channels: Dimensión de las capas ocultas (por cabeza).
        num_heads: Número de cabezas de atención.
        num_layers: Número de capas GATv2Conv.
        num_horizons: Número de horizontes a predecir simultáneamente.
        dropout: Probabilidad de dropout.
        edge_dim: Dimensión del tensor edge_attr (5 con la pipeline
            actual). Si ``None``, GATv2Conv ignora el edge_attr.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        num_horizons: int = 5,
        dropout: float = 0.3,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.num_horizons = num_horizons
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Primera capa: input_dim -> hidden_channels * num_heads
        self.convs.append(
            GATv2Conv(
                input_dim,
                hidden_channels,
                heads=num_heads,
                dropout=dropout,
                concat=True,
                edge_dim=edge_dim,
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * num_heads))

        # Última capa GAT: promedia las cabezas (concat=False)
        if num_layers > 1:
            self.convs.append(
                GATv2Conv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        final_dim = hidden_channels

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
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass sobre el grafo con atención.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Índices de aristas [2, num_edges].
            edge_attr: Tensor de aristas. Si ``self.edge_dim`` es None
                o el tensor es 1D, se usa como peso escalar; en otro
                caso pasa íntegro a GATv2Conv como atributos de arista.

        Returns:
            Predicciones por nodo [num_nodes, num_horizons].
        """
        # GATv2Conv acepta edge_attr siempre que su última dim coincida
        # con `edge_dim`. Si recibimos un tensor 1D (legado), lo
        # expandimos a [E, 1].
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.horizon_head(x)
