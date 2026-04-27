"""Spatio-Temporal Transformer for multi-horizon delay propagation.

Redesign of the previous Seq2SeqGNN per workstream W3 of the project
reset plan. The class name and module path are preserved on purpose —
existing configs (``model.name: seq2seq_gnn``) and checkpoints with
``model_name="seq2seq_gnn"`` keep working — but the internals are
entirely different.

Architecture
------------

1. **Spatial encoder.** Three stacked ``GATv2Conv`` layers with
   pre-norm residual connections, ``LayerNorm`` and ``GELU``. The
   layers consume per-edge attribute vectors of dimension
   ``edge_dim`` (the 5-channel edge tensor produced by the graph
   builder). Pre-norm + residuals stabilise gradients and counteract
   the over-smoothing tendency of stacked GAT layers — both real
   problems with the previous deep-but-skip-less stack.

2. **Temporal encoder.** The per-snapshot embeddings are stacked into
   a ``[N, T, d]`` tensor, augmented with sinusoidal positional
   encoding over the ``T`` timesteps, and processed by a 2-layer
   pre-norm Transformer encoder. This replaces the previous single-
   layer LSTM. Self-attention captures longer-range temporal
   dependencies, parallelises over ``T`` and avoids the LSTM's serial
   gradient bottleneck.

3. **Horizon-query decoder.** ``num_horizons`` learnable query
   embeddings cross-attend to the per-node temporal sequence via a
   single ``MultiheadAttention`` block. A 2-layer MLP head projects
   each attended query to a scalar prediction. There is **no
   autoregression** and **no teacher forcing**, so the model behaves
   identically in train and eval modes (only dropout differs).

Why this should beat the old design on long horizons (4/6/8 h)
---------------------------------------------------------------

* **No autoregressive error propagation.** The 8 h prediction does
  not consume the 6 h prediction, so an early miss does not cascade.
* **Edge-aware spatial attention.** ``GATv2Conv`` consumes the new
  ``edge_dim`` features, letting the model weight neighbours by
  scheduled-flight count, recent route delay, etc.
* **Pre-norm residuals + LayerNorm.** Fix the gradient-flow / over-
  smoothing issues of the previous stack; let the encoder go deeper
  without collapsing node representations.
* **Horizon queries.** The decoder is explicitly aware of *which*
  horizon it is predicting; the previous decoder treated all five
  outputs as a single dense vector with no horizon-positional
  information.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Registered as a non-trainable, non-persistent buffer; sliced to the
    actual sequence length on the fly so a single instance supports any
    ``T <= max_len``.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # When d_model is odd the cos slice is one element shorter — clip
        # div_term to match. d_model is always even in production but the
        # smoke tests sometimes use small odd values for speed.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        return x + self.pe[:, : x.size(1)]


class SpatialGATEncoder(nn.Module):
    """Pre-norm residual stack of ``GATv2Conv`` layers.

    Layout (after the input projection)::

        h = LayerNorm(x)
        h = GATv2Conv(h, edge_index, edge_attr)
        h = GELU(h)
        h = Dropout(h)
        x = x + h

    Pre-norm + residuals stabilise gradients and counteract the over-
    smoothing tendency of stacked GAT layers. The first layer is a
    non-residual projection ``input_dim -> hidden_dim`` because the
    residual sum needs matching shapes.

    Args:
        input_dim: Node feature dimensionality.
        hidden_dim: Hidden / output embedding dimension. ``concat=False``
            on every layer means the head dimension is averaged so output
            stays at ``hidden_dim`` regardless of ``num_heads``.
        num_heads: Number of attention heads per layer.
        num_layers: Total number of GATv2 layers (>= 1). The first is
            the projection, the remaining ``num_layers - 1`` are residual.
        dropout: Dropout probability.
        edge_dim: Last-dimension size of ``edge_attr``. ``None`` disables
            edge features (safe default for synthetic unit tests); the
            production factory wires ``edge_dim=5``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(
                f"num_layers debe ser >= 1, recibido {num_layers}"
            )

        self.dropout = dropout
        self.edge_dim = edge_dim

        # Projection: input_dim -> hidden_dim (no residual: shapes mismatch).
        self.input_proj = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False,
            edge_dim=edge_dim,
        )
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Residual layers.
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Initial projection.
        x = self.input_proj(x, edge_index, edge_attr=edge_attr)
        x = self.input_norm(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pre-norm residual stack.
        for conv, norm in zip(self.layers, self.norms):
            h = norm(x)
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h

        return x


class Seq2SeqGNN(nn.Module):
    """Spatio-Temporal Transformer with a horizon-query decoder.

    The class name is preserved for config / checkpoint backward
    compatibility; the architecture is the redesign described in this
    module's docstring.

    Args:
        input_dim: Node feature dimensionality.
        hidden_dim: Unified embedding dimension across spatial encoder,
            temporal Transformer and decoder. Must be divisible by
            ``num_heads``.
        num_heads: Heads for ``GATv2Conv`` (spatial), ``TransformerEncoder``
            (temporal) and ``MultiheadAttention`` (decoder).
        num_spatial_layers: Number of GATv2 layers in the spatial encoder.
        num_temporal_layers: Number of Transformer encoder layers.
        num_horizons: Number of future horizons to predict.
        dropout: Dropout probability used by encoders, decoder and head.
        edge_dim: Last-dim size of ``edge_attr``. ``None`` disables edge
            features (default for unit tests); the factory wires the
            real per-edge dimension at construction time.
        output_channels: Channels per horizon. ``1`` (default) keeps the
            historical ``[N, H]`` output (single-task ArrDelay regression).
            ``3`` activates the W2 multi-task head: output ``[N, H, 3]``
            with channels ``(arr_delay, dep_delay_aux, pct_15_logit)``.
            The MLP head's final Linear widens from ``hidden_dim -> 1`` to
            ``hidden_dim -> output_channels`` — every channel shares the
            attended representation per (node, horizon) query.

    Raises:
        ValueError: If ``hidden_dim`` is not divisible by ``num_heads``
            (a hard requirement of ``MultiheadAttention`` /
            ``TransformerEncoderLayer``) or if ``output_channels < 1``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_spatial_layers: int = 3,
        num_temporal_layers: int = 2,
        num_horizons: int = 5,
        dropout: float = 0.3,
        edge_dim: int | None = None,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) debe ser divisible por num_heads "
                f"({num_heads}) — requerido por MultiheadAttention / "
                f"TransformerEncoderLayer."
            )
        if output_channels < 1:
            raise ValueError(
                f"output_channels debe ser >= 1, recibido {output_channels}"
            )

        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons
        self.output_channels = output_channels

        # 1) Spatial encoder (per-snapshot GAT).
        self.spatial_encoder = SpatialGATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_spatial_layers,
            dropout=dropout,
            edge_dim=edge_dim,
        )

        # 2) Temporal encoder (per-node Transformer over T timesteps).
        self.positional_encoding = _SinusoidalPositionalEncoding(
            d_model=hidden_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable on small batches
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_temporal_layers,
        )

        # 3) Horizon-query decoder.
        # Learnable [num_horizons, hidden_dim] queries, BERT-style small init.
        self.horizon_queries = nn.Parameter(
            torch.randn(num_horizons, hidden_dim) * 0.02
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # 2-layer MLP head applied to each attended horizon query. The
        # final Linear emits ``output_channels`` per query — 1 in the
        # historical single-task setup; 3 in the W2 multi-task setup
        # (arr_delay regression + dep_delay aux regression + pct_15
        # classification logit). The two regression channels are read in
        # minutes; the classification channel is a logit (sigmoid lives
        # in BCEWithLogitsLoss / metric extraction).
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_channels),
        )

    def forward(self, sequence: list[Data]) -> torch.Tensor:
        """Forward pass over a temporal sequence of graph snapshots.

        Args:
            sequence: List of PyG ``Data`` objects of length ``T``. Each
                snapshot must expose ``x``, ``edge_index`` and (optionally)
                ``edge_attr`` already on the target device. The model
                ignores ``y`` entirely — there is no teacher forcing, so
                the forward is identical in train and eval modes.

        Returns:
            ``[num_nodes, num_horizons]`` if ``output_channels == 1``
            (legacy single-task), else ``[num_nodes, num_horizons,
            output_channels]`` (W2 multi-task).
        """
        # 1) Per-snapshot spatial encoding.
        embeddings = [
            self.spatial_encoder(g.x, g.edge_index, g.edge_attr)
            for g in sequence
        ]
        # Stack list of [N, d] -> [N, T, d].
        spatial = torch.stack(embeddings, dim=1)

        # 2) Positional encoding + Transformer encoder over the T axis.
        spatial = self.positional_encoding(spatial)
        temporal = self.temporal_encoder(spatial)  # [N, T, d]

        # 3) Horizon-query cross-attention.
        num_nodes = temporal.shape[0]
        # Broadcast queries [H, d] -> [N, H, d].
        queries = self.horizon_queries.unsqueeze(0).expand(num_nodes, -1, -1)
        attended, _ = self.cross_attention(
            query=queries,
            key=temporal,
            value=temporal,
            need_weights=False,
        )
        # Residual on the queries + LayerNorm — lets the model fall back
        # to the bare query when attention is uninformative.
        attended = self.cross_attn_norm(attended + queries)

        # 4) Per-horizon MLP head: applied per (node, horizon) query.
        #    output_channels == 1 → [N, H, 1] squeezed to [N, H] (legacy).
        #    output_channels  > 1 → [N, H, output_channels] kept as-is
        #    (multi-task: arr_delay, dep_delay_aux, pct_15_logit).
        out = self.head(attended)  # [N, H, output_channels]
        if self.output_channels == 1:
            return out.squeeze(-1)
        return out
