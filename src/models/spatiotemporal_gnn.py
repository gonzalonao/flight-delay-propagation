"""Spatio-Temporal GNN for multi-horizon delay propagation prediction.

Main thesis contribution. Combines a GAT encoder for spatial propagation
with an LSTM for temporal dynamics. Processes a sequence of consecutive
graph snapshots (e.g., 6 hours of history) and predicts delays at
multiple future horizons simultaneously (Strategy 2: direct multi-output).

Architecture:
    [G_t-5, ..., G_t] -> shared GATEncoder -> [emb_t-5, ..., emb_t]
    -> stack [N, T, hidden] -> LSTM -> h_n [N, lstm_hidden]
    -> MLP head -> [N, num_horizons]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """Spatial encoder using Graph Attention Network layers.

    Replicates the convolutional architecture from MultiHorizonGAT
    (GATConv + BatchNorm + ELU + dropout) without the prediction head.
    Produces per-node spatial embeddings of dimension ``hidden_channels``.

    Args:
        input_dim: Number of node features.
        hidden_channels: Hidden dimension per attention head.
        num_heads: Number of attention heads.
        num_layers: Number of GATConv layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: input_dim -> hidden_channels * num_heads
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

        # Intermediate layers
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

        # Last layer: average heads (concat=False)
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode node features via graph attention.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            edge_weight: Edge weights [num_edges] (optional).

        Returns:
            Node embeddings [num_nodes, hidden_channels].
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_weight)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class SpatioTemporalGNN(nn.Module):
    """GAT encoder + LSTM for spatio-temporal delay propagation.

    Processes a sequence of temporal graph snapshots:
    1. Shared GATEncoder extracts spatial embeddings per snapshot.
    2. LSTM reads the sequence of embeddings per node.
    3. MLP head maps the final LSTM hidden state to multi-horizon
       delay predictions.

    Args:
        input_dim: Number of node features per snapshot.
        gnn_hidden: Hidden dimension for GATEncoder (per head).
        lstm_hidden: Hidden dimension for the LSTM.
        num_heads: Number of GAT attention heads.
        num_gnn_layers: Number of GATConv layers in the encoder.
        num_horizons: Number of future horizons to predict.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        gnn_hidden: int = 64,
        lstm_hidden: int = 128,
        num_heads: int = 4,
        num_gnn_layers: int = 2,
        num_horizons: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.gnn_hidden = gnn_hidden
        self.lstm_hidden = lstm_hidden
        self.num_horizons = num_horizons

        self.encoder = GATEncoder(
            input_dim=input_dim,
            hidden_channels=gnn_hidden,
            num_heads=num_heads,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_horizons),
        )

    def forward(self, sequence: list[Data]) -> torch.Tensor:
        """Forward pass over a temporal sequence of graph snapshots.

        Args:
            sequence: List of PyG Data objects (length = input_window).
                Each Data must have x, edge_index, and optionally
                edge_attr already on the correct device.

        Returns:
            Predictions [num_nodes, num_horizons].
        """
        embeddings = []
        for graph in sequence:
            edge_weight = None
            if graph.edge_attr is not None:
                edge_weight = graph.edge_attr.squeeze(-1)

            emb = self.encoder(graph.x, graph.edge_index, edge_weight)
            embeddings.append(emb)

        # Stack: list of [N, gnn_hidden] -> [N, T, gnn_hidden]
        stacked = torch.stack(embeddings, dim=1)

        # LSTM: [N, T, gnn_hidden] -> h_n [1, N, lstm_hidden]
        _, (h_n, _) = self.lstm(stacked)

        # Final hidden state -> predictions
        return self.head(h_n.squeeze(0))
