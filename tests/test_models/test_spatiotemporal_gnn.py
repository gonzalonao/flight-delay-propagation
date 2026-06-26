"""Tests for SpatioTemporalGNN, GATEncoder, temporal sequences and trainer."""

import pytest
import torch
from torch_geometric.data import Data

from src.data.graph_builder import create_temporal_sequences
from src.models.spatiotemporal_gnn import GATEncoder, SpatioTemporalGNN
from src.training.graph_trainer import SequenceGraphTrainer


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def simple_graph():
    """Simple PyG graph with 4 nodes and 6 features."""
    num_nodes = 4
    input_dim = 6
    num_horizons = 5
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0, 3],
         [1, 0, 2, 1, 3, 2, 3, 0]], dtype=torch.long
    )
    edge_attr = torch.ones(edge_index.shape[1], 5)
    y = torch.randn(num_nodes, num_horizons)
    active_mask = torch.tensor([True, True, True, False])
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, active_mask=active_mask,
    )


@pytest.fixture
def graph_sequence(simple_graph):
    """Sequence of 6 graphs with slightly different data."""
    sequence = []
    for _ in range(6):
        g = simple_graph.clone()
        g.x = torch.randn_like(g.x)
        g.y = torch.randn_like(g.y)
        sequence.append(g)
    return sequence


@pytest.fixture
def graph_list():
    """List of 15 PyG graphs for sequence tests."""
    graphs = []
    for i in range(15):
        num_nodes = 4
        g = Data(
            x=torch.randn(num_nodes, 6),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            edge_attr=torch.ones(3, 5),
            y=torch.randn(num_nodes, 5),
            active_mask=torch.ones(num_nodes, dtype=torch.bool),
        )
        graphs.append(g)
    return graphs


# -- GATEncoder tests --------------------------------------------------------


class TestGATEncoder:
    """Tests for the spatial GAT encoder."""

    def test_output_shape(self, simple_graph):
        """Output must be [num_nodes, hidden_channels]."""
        encoder = GATEncoder(
            input_dim=6, hidden_channels=16, num_heads=2, num_layers=2,
        )
        out = encoder(simple_graph.x, simple_graph.edge_index)
        assert out.shape == (4, 16)

    def test_with_edge_attr(self, simple_graph):
        """Works with multi-channel edge_attr."""
        encoder = GATEncoder(
            input_dim=6, hidden_channels=16, num_heads=2, num_layers=2,
            edge_dim=5,
        )
        out = encoder(
            simple_graph.x, simple_graph.edge_index, simple_graph.edge_attr,
        )
        assert out.shape == (4, 16)

    def test_single_layer(self):
        """Encoder with a single GAT layer."""
        encoder = GATEncoder(
            input_dim=6, hidden_channels=8, num_heads=2, num_layers=1,
        )
        x = torch.randn(4, 6)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = encoder(x, edge_index)
        # With num_layers=1, output is hidden_channels * num_heads (concat=True)
        assert out.shape[0] == 4


# -- SpatioTemporalGNN tests ------------------------------------------------


class TestSpatioTemporalGNN:
    """Tests for the full SpatioTemporalGNN model."""

    def test_output_shape(self, graph_sequence):
        """Output must be [num_nodes, num_horizons]."""
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.eval()
        out = model(graph_sequence)
        assert out.shape == (4, 5)

    def test_different_configs(self, graph_sequence):
        """Different configurations produce correct shapes."""
        configs = [
            {"gnn_hidden": 8, "lstm_hidden": 16, "num_heads": 1,
             "num_gnn_layers": 2, "num_horizons": 3},
            {"gnn_hidden": 32, "lstm_hidden": 64, "num_heads": 4,
             "num_gnn_layers": 2, "num_horizons": 5},
        ]
        for cfg in configs:
            model = SpatioTemporalGNN(input_dim=6, **cfg)
            model.eval()
            out = model(graph_sequence)
            assert out.shape == (4, cfg["num_horizons"])

    def test_gradients_flow(self, graph_sequence):
        """Gradients flow through GAT, LSTM and head."""
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        out = model(graph_sequence)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient in {name}"

    def test_eval_mode_deterministic(self, graph_sequence):
        """In eval mode, the output must be deterministic."""
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.eval()
        out1 = model(graph_sequence)
        out2 = model(graph_sequence)
        assert torch.allclose(out1, out2)

    def test_variable_sequence_lengths(self):
        """Sequences of different lengths work correctly."""
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.eval()

        for seq_len in [1, 3, 6, 10]:
            seq = []
            for _ in range(seq_len):
                g = Data(
                    x=torch.randn(4, 6),
                    edge_index=torch.tensor(
                        [[0, 1], [1, 0]], dtype=torch.long
                    ),
                )
                seq.append(g)
            out = model(seq)
            assert out.shape == (4, 5)

    def test_output_channels_widens_to_multitask(self, graph_sequence):
        """With output_channels=3 (W2), output is [N, H, 3]."""
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
            output_channels=3,
        )
        model.eval()
        out = model(graph_sequence)
        assert out.shape == (4, 5, 3)

    def test_invalid_output_channels_raises(self):
        """output_channels < 1 must raise ValueError at construction."""
        with pytest.raises(ValueError, match=">= 1"):
            SpatioTemporalGNN(
                input_dim=6, gnn_hidden=16, lstm_hidden=32,
                num_heads=2, num_gnn_layers=2, num_horizons=5,
                output_channels=0,
            )

    def test_more_params_than_gat(self, graph_sequence):
        """SpatioTemporalGNN must have more parameters than MultiHorizonGAT."""
        from src.models.multi_horizon_gat import MultiHorizonGAT

        gat = MultiHorizonGAT(
            input_dim=6, hidden_channels=64, num_heads=4,
            num_layers=2, num_horizons=5,
        )
        stgnn = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=64, lstm_hidden=128,
            num_heads=4, num_gnn_layers=2, num_horizons=5,
        )

        gat_params = sum(p.numel() for p in gat.parameters())
        stgnn_params = sum(p.numel() for p in stgnn.parameters())
        assert stgnn_params > gat_params


# -- create_temporal_sequences tests ----------------------------------------


class TestCreateTemporalSequences:
    """Tests for temporal sequence creation."""

    def test_basic_sliding_window(self, graph_list):
        """10 graphs with window=6 produce 5 sequences."""
        graphs = graph_list[:10]
        sequences = create_temporal_sequences(graphs, input_window=6)
        assert len(sequences) == 5  # 10 - 6 + 1

    def test_sequence_length(self, graph_list):
        """Each sequence has exactly input_window graphs."""
        sequences = create_temporal_sequences(graph_list, input_window=6)
        for seq in sequences:
            assert len(seq) == 6

    def test_empty_when_too_few_graphs(self):
        """Fewer graphs than the window returns an empty list."""
        graphs = [
            Data(x=torch.randn(4, 6), edge_index=torch.zeros(2, 0, dtype=torch.long))
            for _ in range(3)
        ]
        sequences = create_temporal_sequences(graphs, input_window=6)
        assert len(sequences) == 0

    def test_targets_from_last_graph(self, graph_list):
        """Each sequence's target is the .y of the last graph."""
        sequences = create_temporal_sequences(graph_list, input_window=6)
        for seq in sequences:
            # Last graph's y is the sequence target
            assert seq[-1].y is not None
            assert seq[-1].y.shape == (4, 5)

    def test_sliding_step_one(self, graph_list):
        """Consecutive sequences overlap in input_window-1 graphs."""
        sequences = create_temporal_sequences(graph_list, input_window=6)
        if len(sequences) >= 2:
            # First sequence [0:6], second [1:7] -> overlap is [1:6]
            for i in range(1, 6):
                assert sequences[0][i] is sequences[1][i - 1]

    def test_exact_window_size(self):
        """Exactly input_window graphs produces 1 sequence."""
        graphs = [
            Data(x=torch.randn(4, 6), edge_index=torch.zeros(2, 0, dtype=torch.long))
            for _ in range(6)
        ]
        sequences = create_temporal_sequences(graphs, input_window=6)
        assert len(sequences) == 1


# -- SequenceGraphTrainer tests ---------------------------------------------


class TestSequenceGraphTrainer:
    """Tests for the graph-sequence trainer."""

    def _make_sequences(self, n_seqs=3, seq_len=6):
        """Create synthetic sequences for testing."""
        sequences = []
        for _ in range(n_seqs):
            seq = []
            for _ in range(seq_len):
                g = Data(
                    x=torch.randn(4, 6),
                    edge_index=torch.tensor(
                        [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
                    ),
                    edge_attr=torch.ones(4, 1),
                    y=torch.randn(4, 5),
                    active_mask=torch.ones(4, dtype=torch.bool),
                )
                seq.append(g)
            sequences.append(seq)
        return sequences

    def test_train_epoch_runs(self):
        """train_epoch runs without errors and returns a float."""
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=8, lstm_hidden=16,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
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

    def test_validate_runs(self):
        """validate runs without errors and returns a float."""
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=8, lstm_hidden=16,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            device=torch.device("cpu"),
        )

        sequences = self._make_sequences(n_seqs=3)
        loss = trainer.validate(sequences)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases(self):
        """The loss decreases after a few training epochs."""
        torch.manual_seed(42)
        model = SpatioTemporalGNN(
            input_dim=6, gnn_hidden=8, lstm_hidden=16,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            device=torch.device("cpu"),
        )

        sequences = self._make_sequences(n_seqs=5)
        loss_first = trainer.train_epoch(sequences)
        for _ in range(9):
            loss_last = trainer.train_epoch(sequences)

        # After 10 epochs on fixed data, loss should decrease
        assert loss_last < loss_first
