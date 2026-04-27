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
from torch_geometric.nn import GATv2Conv


class GATEncoder(nn.Module):
    """Spatial encoder using Graph Attention v2 layers with edge attrs.

    Stacks ``GATv2Conv`` blocks with BatchNorm + ELU + dropout. The
    layers consume per-edge attribute vectors of dimension ``edge_dim``
    (the 5-channel edge tensor produced by ``graph_builder``).

    Args:
        input_dim: Number of node features.
        hidden_channels: Hidden dimension per attention head.
        num_heads: Number of attention heads.
        num_layers: Number of GATv2Conv layers.
        dropout: Dropout probability.
        edge_dim: Last-dimension size of ``edge_attr``. ``None`` (default)
            disables edge features and is the safe choice for unit tests
            with synthetic graphs that do not include edge attributes;
            the production factory passes ``edge_dim=5`` explicitly.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_layers == 1:
            # Single-layer encoder: average heads on the only layer so the
            # output dim is ``hidden_channels`` (consistent with the
            # multi-layer branch and what the downstream LSTM expects).
            self.convs.append(
                GATv2Conv(
                    input_dim,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        else:
            # First layer: input_dim -> hidden_channels * num_heads
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

            # Intermediate layers
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

            # Last layer: average heads (concat=False) so output dim
            # collapses back to ``hidden_channels``.
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode node features via graph attention.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge attributes [num_edges, edge_dim] (optional).
                A 1D tensor is automatically promoted to ``[E, 1]``.

        Returns:
            Node embeddings [num_nodes, hidden_channels].
        """
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
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
        edge_dim: Edge attribute dimension forwarded to ``GATEncoder``.
            ``None`` disables edge features (default for tests); the
            production factory wires ``edge_dim=5``.
        output_channels: Channels per horizon. ``1`` (default) keeps the
            historical ``[N, H]`` output (single-task ArrDelay regression).
            ``3`` activates the W2 multi-task head: output ``[N, H, 3]``
            with channels ``(arr_delay, dep_delay_aux, pct_15_logit)``,
            consumed by ``MultiTaskLoss`` and the BCE-derived metrics.
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
        edge_dim: int | None = None,
        output_channels: int = 1,
    ) -> None:
        super().__init__()

        if output_channels < 1:
            raise ValueError(
                f"output_channels must be >= 1, got {output_channels}"
            )

        self.gnn_hidden = gnn_hidden
        self.lstm_hidden = lstm_hidden
        self.num_horizons = num_horizons
        self.output_channels = output_channels

        self.encoder = GATEncoder(
            input_dim=input_dim,
            hidden_channels=gnn_hidden,
            num_heads=num_heads,
            num_layers=num_gnn_layers,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        self.lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Single linear projects to `num_horizons * output_channels` and
        # the forward reshapes to the multi-task tensor when needed.
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_horizons * output_channels),
        )

    def forward(self, sequence: list[Data]) -> torch.Tensor:
        """Forward pass over a temporal sequence of graph snapshots.

        Args:
            sequence: List of PyG Data objects (length = input_window).
                Each Data must have x, edge_index, and optionally
                edge_attr already on the correct device.

        Returns:
            ``[num_nodes, num_horizons]`` if ``output_channels == 1``
            (legacy single-task), else ``[num_nodes, num_horizons,
            output_channels]`` (W2 multi-task).
        """
        embeddings = []
        for graph in sequence:
            emb = self.encoder(graph.x, graph.edge_index, graph.edge_attr)
            embeddings.append(emb)

        # Stack: list of [N, gnn_hidden] -> [N, T, gnn_hidden]
        stacked = torch.stack(embeddings, dim=1)

        # LSTM: [N, T, gnn_hidden] -> h_n [1, N, lstm_hidden]
        _, (h_n, _) = self.lstm(stacked)

        out = self.head(h_n.squeeze(0))
        if self.output_channels == 1:
            return out  # [N, H] — legacy
        return out.view(out.size(0), self.num_horizons, self.output_channels)
