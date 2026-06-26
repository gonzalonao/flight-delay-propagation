"""Tests for Seq2SeqGNN (Spatio-Temporal Transformer + horizon-query decoder).

Reflects the W3 workstream redesign:
    - Constructor: ``hidden_dim``, ``num_spatial_layers``, ``num_temporal_layers``
      (the old ``gnn_hidden`` / ``lstm_hidden`` / ``num_gnn_layers`` are kept
      only in the factory as a backward-compatibility fallback).
    - No teacher forcing: the output does not depend on ``y`` in train or
      eval; the model is identical except for dropout.
    - No ``start_token`` or ``output_heads`` ModuleList: the decoder uses
      per-horizon query embeddings and a single output MLP.
"""

import pytest
import torch
from torch_geometric.data import Data

from src.models.seq2seq_gnn import Seq2SeqGNN, SpatialGATEncoder
from src.training.graph_trainer import SequenceGraphTrainer


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def simple_graph():
    """PyG graph with 4 nodes, 6 features, 5 horizons."""
    num_nodes = 4
    input_dim = 6
    num_horizons = 5
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0, 3],
         [1, 0, 2, 1, 3, 2, 3, 0]], dtype=torch.long
    )
    # Non-negative ``edge_attr``: if a future test passes this to a layer
    # that normalizes weights (GCNConv) we avoid NaN. GATv2Conv treats it as
    # features and accepts arbitrary values anyway.
    edge_attr = torch.rand(edge_index.shape[1], 1)
    y = torch.randn(num_nodes, num_horizons)
    active_mask = torch.ones(num_nodes, dtype=torch.bool)
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, active_mask=active_mask,
    )


@pytest.fixture
def graph_sequence(simple_graph):
    """Sequence of 6 graphs; only the last exposes a real target."""
    sequence = []
    for i in range(6):
        g = simple_graph.clone()
        g.x = torch.randn_like(g.x)
        if i < 5:
            g.y = torch.randn_like(g.y)
        sequence.append(g)
    return sequence


def _make_model(
    input_dim: int = 6,
    hidden_dim: int = 16,
    num_horizons: int = 5,
    dropout: float = 0.3,
    num_spatial_layers: int = 2,
    num_temporal_layers: int = 1,
) -> Seq2SeqGNN:
    """Helper to reduce noise in the tests."""
    return Seq2SeqGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=2,
        num_spatial_layers=num_spatial_layers,
        num_temporal_layers=num_temporal_layers,
        num_horizons=num_horizons,
        dropout=dropout,
        edge_dim=1,  # the fixtures use 1-channel edge_attr
    )


# -- Seq2SeqGNN model tests --------------------------------------------------


class TestSeq2SeqGNN:
    """Tests for the new Seq2SeqGNN architecture."""

    def test_output_shape_eval(self, graph_sequence):
        """Output must be [num_nodes, num_horizons] in eval mode."""
        model = _make_model()
        model.eval()
        out = model(graph_sequence)
        assert out.shape == (4, 5)

    def test_output_shape_train(self, graph_sequence):
        """Output with the same shape in train mode (no distribution shift)."""
        model = _make_model()
        model.train()
        out = model(graph_sequence)
        assert out.shape == (4, 5)

    def test_different_num_horizons(self, graph_sequence):
        """Different num_horizons produce outputs of the correct size."""
        for h in [1, 3, 5, 7]:
            seq = [g.clone() for g in graph_sequence]
            seq[-1].y = torch.randn(4, h)
            model = _make_model(num_horizons=h, hidden_dim=8)
            model.eval()
            out = model(seq)
            assert out.shape == (4, h)

    def test_gradients_flow(self, graph_sequence):
        """Gradients flow through the spatial encoder, Transformer, horizon
        queries, cross-attention and head MLP."""
        model = _make_model()
        model.train()
        out = model(graph_sequence)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient in {name}"

    def test_eval_mode_deterministic(self, graph_sequence):
        """In eval mode dropout is off, so the output is deterministic."""
        model = _make_model()
        model.eval()
        out1 = model(graph_sequence)
        out2 = model(graph_sequence)
        assert torch.allclose(out1, out2)

    def test_targets_have_no_effect_in_eval(self, graph_sequence):
        """No teacher forcing: changing ``y`` does not change the eval prediction."""
        model = _make_model()
        model.eval()

        out1 = model(graph_sequence)

        seq2 = [g.clone() for g in graph_sequence]
        seq2[-1].y = graph_sequence[-1].y * 10.0
        out2 = model(seq2)

        assert torch.allclose(out1, out2), (
            "The new Seq2SeqGNN output must not depend on y "
            "(no teacher forcing)."
        )

    def test_targets_have_no_effect_in_train(self, graph_sequence):
        """No teacher forcing in train either: the model never reads ``y``."""
        torch.manual_seed(0)
        model = _make_model(dropout=0.0)
        model.train()

        out1 = model(graph_sequence)

        seq2 = [g.clone() for g in graph_sequence]
        seq2[-1].y = graph_sequence[-1].y * 10.0
        out2 = model(seq2)

        assert torch.allclose(out1, out2), (
            "With dropout=0, train and eval must match and no horizon can "
            "depend on y (the redesign removes teacher forcing)."
        )

    def test_horizon_queries_are_learnable(self):
        """The per-horizon query embeddings are trainable parameters."""
        model = _make_model(num_horizons=5)
        assert model.horizon_queries.requires_grad
        assert model.horizon_queries.shape == (5, model.hidden_dim)

    def test_single_mlp_head_not_per_horizon_linears(self):
        """The head is a single MLP — no longer one Linear per horizon."""
        model = _make_model()
        # The old design's ``output_heads`` attribute must not exist.
        assert not hasattr(model, "output_heads")
        # And there must be a ``head`` that is a Sequential MLP.
        assert isinstance(model.head, torch.nn.Sequential)

    def test_no_lstm_modules(self):
        """The redesign uses no LSTMs (neither encoder nor decoder)."""
        model = _make_model()
        for module in model.modules():
            assert not isinstance(module, torch.nn.LSTM), (
                "The new Seq2SeqGNN must not contain LSTMs"
            )

    def test_output_channels_widens_to_multitask(self, graph_sequence):
        """With output_channels=3 (W2), output is [N, H, 3] in eval and train."""
        model = _make_model()
        # Build with the same signature but output_channels=3.
        model = Seq2SeqGNN(
            input_dim=6, hidden_dim=16, num_heads=2,
            num_spatial_layers=2, num_temporal_layers=1,
            num_horizons=5, dropout=0.3, edge_dim=1,
            output_channels=3,
        )
        model.eval()
        out_eval = model(graph_sequence)
        assert out_eval.shape == (4, 5, 3)
        model.train()
        out_train = model(graph_sequence)
        assert out_train.shape == (4, 5, 3)

    def test_invalid_output_channels_raises(self):
        """output_channels < 1 must raise ValueError at construction."""
        with pytest.raises(ValueError, match=">= 1"):
            Seq2SeqGNN(
                input_dim=6, hidden_dim=16, num_heads=2,
                num_spatial_layers=1, num_temporal_layers=1,
                num_horizons=3, output_channels=0,
            )

    def test_hidden_dim_must_be_divisible_by_num_heads(self):
        """The constructor fails fast if hidden_dim % num_heads != 0."""
        with pytest.raises(ValueError, match="divisible by num_heads"):
            Seq2SeqGNN(
                input_dim=6,
                hidden_dim=15,    # 15 is not divisible by 2
                num_heads=2,
                num_spatial_layers=1,
                num_temporal_layers=1,
                num_horizons=3,
            )

    def test_spatial_encoder_supports_single_layer(self, simple_graph):
        """SpatialGATEncoder with num_layers=1 still produces hidden_dim."""
        enc = SpatialGATEncoder(
            input_dim=6, hidden_dim=16, num_heads=2,
            num_layers=1, dropout=0.0, edge_dim=1,
        )
        out = enc(simple_graph.x, simple_graph.edge_index, simple_graph.edge_attr)
        assert out.shape == (4, 16)


# -- Integration tests with SequenceGraphTrainer ----------------------------


class TestSeq2SeqWithTrainer:
    """Tests verifying that Seq2SeqGNN trains with the existing trainer."""

    def _make_sequences(self, n_seqs: int = 3) -> list[list[Data]]:
        """Create synthetic sequences with 1-channel edge_attr."""
        sequences = []
        for _ in range(n_seqs):
            seq = []
            for _ in range(6):
                g = Data(
                    x=torch.randn(4, 6),
                    edge_index=torch.tensor(
                        [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
                    ),
                    edge_attr=torch.rand(4, 1),
                    y=torch.randn(4, 5),
                    active_mask=torch.ones(4, dtype=torch.bool),
                )
                seq.append(g)
            sequences.append(seq)
        return sequences

    def test_trains_with_sequence_trainer(self):
        """Seq2SeqGNN integrates with SequenceGraphTrainer unchanged."""
        model = _make_model(hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            device=torch.device("cpu"),
        )

        sequences = self._make_sequences(n_seqs=3)
        loss = trainer.train_epoch(sequences)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self):
        """The loss decreases after several epochs over fixed data."""
        torch.manual_seed(42)
        model = _make_model(hidden_dim=8, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            device=torch.device("cpu"),
        )

        sequences = self._make_sequences(n_seqs=5)
        loss_first = trainer.train_epoch(sequences)
        for _ in range(14):
            loss_last = trainer.train_epoch(sequences)

        assert loss_last < loss_first

    def test_validate_uses_eval_mode(self):
        """validate() puts the model in eval."""
        model = _make_model(hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            device=torch.device("cpu"),
        )

        sequences = self._make_sequences(n_seqs=2)
        _ = trainer.validate(sequences)
        assert not model.training
