"""Multi-horizon Graph Attention Network for delay prediction.

The project's second GNN model. Uses GATConv layers with multi-head
attention to weight the importance of connections between airports.
Predicts the average delay (in minutes) of each airport at multiple future
horizons simultaneously (e.g., 1h, 2h, ..., 5h).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class MultiHorizonGAT(nn.Module):
    """GAT for multi-horizon per-airport delay prediction.

    Architecture: N ``GATv2Conv`` layers (with ``edge_dim`` to integrate
    multi-channel edge features), BatchNorm, ELU and dropout, followed by
    a per-horizon projection head. Attention learns which connections
    between airports are most relevant for propagating delay information,
    now conditioned on the 5 edge features (Class A + B) produced by
    ``graph_builder``.

    Args:
        input_dim: Number of features per node.
        hidden_channels: Dimension of the hidden layers (per head).
        num_heads: Number of attention heads.
        num_layers: Number of GATv2Conv layers.
        num_horizons: Number of horizons to predict simultaneously.
        dropout: Dropout probability.
        edge_dim: Dimension of the edge_attr tensor (5 with the current
            pipeline). If ``None``, GATv2Conv ignores edge_attr.
        output_channels: Number of channels per horizon. ``1`` (default)
            preserves the historical behavior — output ``[N, H]`` with
            only the ArrDelay regression. ``3`` enables the W2 multi-task
            head: output ``[N, H, 3]`` with channels
            ``(arr_delay, dep_delay_aux, pct_arr_delayed_15_logit)``,
            consumed by ``MultiTaskLoss`` and the BCE metrics.
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
        output_channels: int = 1,
    ) -> None:
        super().__init__()

        if output_channels < 1:
            raise ValueError(
                f"output_channels must be >= 1, received {output_channels}"
            )

        self.dropout = dropout
        self.num_horizons = num_horizons
        self.edge_dim = edge_dim
        self.output_channels = output_channels

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

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

        # Last GAT layer: averages the heads (concat=False)
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

        # The head projects to `num_horizons * output_channels` and then
        # reshapes to the final tensor. A single linear layer shared across
        # all channels and horizons is the simplest option and keeps the
        # parameter count almost identical to the previous design when
        # output_channels=1.
        self.horizon_head = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim, num_horizons * output_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass over the graph with attention.

        Args:
            x: Node features [num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge tensor. If ``self.edge_dim`` is None or the
                tensor is 1D, it is used as a scalar weight; otherwise it
                is passed through to GATv2Conv as edge attributes.

        Returns:
            If ``output_channels == 1``: tensor ``[num_nodes,
            num_horizons]`` (compatible with the previous design).
            If ``output_channels > 1``: tensor ``[num_nodes,
            num_horizons, output_channels]`` for multi-task (W2).
        """
        # GATv2Conv accepts edge_attr as long as its last dim matches
        # `edge_dim`. If we receive a 1D tensor (legacy), we expand it
        # to [E, 1].
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.horizon_head(x)
        if self.output_channels == 1:
            return out  # [N, H] — historical behavior
        return out.view(out.size(0), self.num_horizons, self.output_channels)
