"""Graph convolutional network (GCN) for delay prediction.

The project's first GNN model. Operates on airport graphs where nodes
represent airports and edges represent air routes. Predicts the average
delay (in minutes) of each airport over the next time window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class BasicGCN(nn.Module):
    """GCN for per-airport delay prediction (regression).

    Architecture: N GCNConv layers with BatchNorm, ReLU and dropout,
    followed by a linear output layer. Each layer propagates information
    from neighboring airports to capture the spatial propagation of
    delays.

    Args:
        input_dim: Number of features per node.
        hidden_channels: Dimension of the hidden GCN layers.
        num_layers: Number of GCNConv layers.
        dropout: Dropout probability.
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

        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: input_dim -> hidden_channels
        self.convs.append(GCNConv(input_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Intermediate layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer: per-node regression (delay in minutes)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the graph.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge tensor. May be ``[E]`` (legacy scalar weight)
                or ``[E, k]`` (multi-feature). GCNConv only accepts a
                scalar, so we extract the first column
                (``flight_count_norm``) when it is 2D.

        Returns:
            Per-node logits [num_nodes, 1].
        """
        edge_weight = None
        if edge_attr is not None:
            edge_weight = edge_attr[:, 0] if edge_attr.dim() == 2 else edge_attr

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)
