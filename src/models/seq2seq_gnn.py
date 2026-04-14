"""Seq2Seq Spatio-Temporal GNN for multi-horizon delay propagation.

Alternative to SpatioTemporalGNN (Strategy 2). Implements Strategy 3
(encoder-decoder seq2seq) from the multi-step forecasting literature:

    1. Shared GAT encoder extracts spatial embeddings per snapshot.
    2. Encoder LSTM compresses the 6-step history into a context vector
       (h_n, c_n).
    3. Decoder LSTM unrolls N decoding steps, each producing one horizon's
       delay prediction. Each step is conditioned on the previous
       prediction (autoregressive) or the previous target (teacher forcing
       during training).
    4. Per-horizon Linear heads project the decoder hidden state to a
       scalar delay prediction.

The decoder generating predictions step-by-step explicitly models delay
cascading dynamics, which the direct multi-output head of
SpatioTemporalGNN cannot capture.

Training uses teacher forcing (ground-truth previous horizon as input).
Inference (eval mode) uses fully autoregressive decoding.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.spatiotemporal_gnn import GATEncoder


class Seq2SeqGNN(nn.Module):
    """Seq2Seq encoder-decoder over spatio-temporal graph sequences.

    Args:
        input_dim: Number of node features per snapshot.
        gnn_hidden: Hidden dimension for the shared GAT encoder.
        lstm_hidden: Hidden dimension for both encoder and decoder LSTM.
        num_heads: Number of GAT attention heads.
        num_gnn_layers: Number of GATConv layers in the encoder.
        num_horizons: Number of decoding steps (future horizons).
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

        # Shared spatial encoder (reused across all input timesteps)
        self.encoder = GATEncoder(
            input_dim=input_dim,
            hidden_channels=gnn_hidden,
            num_heads=num_heads,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # Temporal encoder: reads sequence of spatial embeddings
        self.encoder_lstm = nn.LSTM(
            input_size=gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Temporal decoder: unrolls step-by-step using previous prediction
        # Decoder input is a scalar delay value per node (shape [N, 1])
        self.decoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Learned start-of-sequence token for the first decoder step.
        # Shape [1, 1, 1] -> broadcastable to [N, 1, 1].
        self.start_token = nn.Parameter(torch.zeros(1, 1, 1))

        # Separate output head per horizon
        self.output_heads = nn.ModuleList(
            [nn.Linear(lstm_hidden, 1) for _ in range(num_horizons)]
        )

        self.dropout = nn.Dropout(dropout)

    def _encode(self, sequence: list[Data]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run shared GAT over each snapshot and compress via encoder LSTM.

        Args:
            sequence: List of PyG Data objects already on the target
                device.

        Returns:
            Tuple of (h_n, c_n), each of shape [1, num_nodes, lstm_hidden].
        """
        embeddings = []
        for graph in sequence:
            edge_weight = None
            if graph.edge_attr is not None:
                edge_weight = graph.edge_attr.squeeze(-1)
            emb = self.encoder(graph.x, graph.edge_index, edge_weight)
            embeddings.append(emb)

        # Stack spatial embeddings into a temporal sequence
        # List of [N, gnn_hidden] -> [N, T, gnn_hidden]
        stacked = torch.stack(embeddings, dim=1)

        _, (h_n, c_n) = self.encoder_lstm(stacked)
        return h_n, c_n

    def _decode(
        self,
        h_n: torch.Tensor,
        c_n: torch.Tensor,
        num_nodes: int,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unroll the decoder LSTM for ``num_horizons`` steps.

        Uses teacher forcing during training when targets are provided
        and the model is in training mode. Otherwise uses autoregressive
        decoding (previous prediction feeds next step).

        Args:
            h_n: Encoder final hidden state [1, num_nodes, lstm_hidden].
            c_n: Encoder final cell state [1, num_nodes, lstm_hidden].
            num_nodes: Number of graph nodes (batch size for the LSTM).
            targets: Ground-truth targets [num_nodes, num_horizons] for
                teacher forcing. If None or model in eval mode, decode
                autoregressively.

        Returns:
            Predictions [num_nodes, num_horizons].
        """
        use_teacher_forcing = self.training and targets is not None

        # Initial decoder input: learned start token broadcast across nodes
        # Shape [num_nodes, 1, 1]: batch=N, seq_len=1, feature=1
        decoder_input = self.start_token.expand(num_nodes, 1, 1)

        hidden = (h_n, c_n)
        predictions = []

        for step in range(self.num_horizons):
            # One-step decoder pass: out shape [N, 1, lstm_hidden]
            out, hidden = self.decoder_lstm(decoder_input, hidden)

            # Per-horizon output head
            step_pred = self.output_heads[step](self.dropout(out.squeeze(1)))
            # step_pred shape: [N, 1]
            predictions.append(step_pred)

            # Prepare input for next step
            if step + 1 < self.num_horizons:
                if use_teacher_forcing:
                    # Feed ground-truth previous horizon
                    next_input = targets[:, step].unsqueeze(-1).unsqueeze(-1)
                else:
                    # Feed own previous prediction
                    next_input = step_pred.unsqueeze(-1)
                decoder_input = next_input

        # Concatenate per-step predictions: list of [N, 1] -> [N, num_horizons]
        return torch.cat(predictions, dim=1)

    def forward(self, sequence: list[Data]) -> torch.Tensor:
        """Forward pass over a temporal sequence of graph snapshots.

        During training (``self.training == True``), uses teacher forcing
        with ground-truth targets from ``sequence[-1].y``. During eval,
        decodes autoregressively.

        Args:
            sequence: List of PyG Data objects on the target device.
                The last graph must contain ``y`` with shape
                [num_nodes, num_horizons] for teacher forcing.

        Returns:
            Predictions [num_nodes, num_horizons].
        """
        h_n, c_n = self._encode(sequence)
        num_nodes = sequence[0].x.shape[0]

        targets = sequence[-1].y if sequence[-1].y is not None else None
        return self._decode(h_n, c_n, num_nodes, targets=targets)
