"""Tests para Seq2SeqGNN (encoder-decoder con teacher forcing)."""

import pytest
import torch
from torch_geometric.data import Data

from src.models.seq2seq_gnn import Seq2SeqGNN
from src.training.graph_trainer import SequenceGraphTrainer


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def simple_graph():
    """Grafo PyG con 4 nodos, 6 features, 5 horizontes."""
    num_nodes = 4
    input_dim = 6
    num_horizons = 5
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0, 3],
         [1, 0, 2, 1, 3, 2, 3, 0]], dtype=torch.long
    )
    edge_attr = torch.ones(edge_index.shape[1], 1)
    y = torch.randn(num_nodes, num_horizons)
    active_mask = torch.ones(num_nodes, dtype=torch.bool)
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, active_mask=active_mask,
    )


@pytest.fixture
def graph_sequence(simple_graph):
    """Secuencia de 6 grafos, solo el último tiene target (y)."""
    sequence = []
    for i in range(6):
        g = simple_graph.clone()
        g.x = torch.randn_like(g.x)
        if i < 5:
            # Intermediate graphs still have y (they came from the dataset)
            g.y = torch.randn_like(g.y)
        sequence.append(g)
    return sequence


# -- Tests del modelo Seq2SeqGNN ---------------------------------------------


class TestSeq2SeqGNN:
    """Tests para la arquitectura Seq2SeqGNN."""

    def test_output_shape_eval(self, graph_sequence):
        """Output debe ser [num_nodes, num_horizons] en modo eval."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.eval()
        out = model(graph_sequence)
        assert out.shape == (4, 5)

    def test_output_shape_train(self, graph_sequence):
        """Output correcto en modo train (con teacher forcing)."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.train()
        out = model(graph_sequence)
        assert out.shape == (4, 5)

    def test_different_num_horizons(self, graph_sequence):
        """Distintos num_horizons producen outputs del tamaño correcto."""
        for h in [1, 3, 5, 7]:
            # Adjust target shape to match
            seq = [g.clone() for g in graph_sequence]
            seq[-1].y = torch.randn(4, h)

            model = Seq2SeqGNN(
                input_dim=6, gnn_hidden=8, lstm_hidden=16,
                num_heads=2, num_gnn_layers=2, num_horizons=h,
            )
            model.eval()
            out = model(seq)
            assert out.shape == (4, h)

    def test_gradients_flow(self, graph_sequence):
        """Los gradientes fluyen a través de encoder, decoder y heads."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.train()
        out = model(graph_sequence)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente en {name}"

    def test_eval_mode_deterministic(self, graph_sequence):
        """En modo eval, el output es determinista."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.eval()
        out1 = model(graph_sequence)
        out2 = model(graph_sequence)
        assert torch.allclose(out1, out2)

    def test_teacher_forcing_uses_targets(self, graph_sequence):
        """En modo train, cambiar targets afecta el output (teacher forcing)."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
            dropout=0.0,
        )
        model.train()

        # First forward pass with original targets
        out1 = model(graph_sequence)

        # Second forward pass with different targets
        seq2 = [g.clone() for g in graph_sequence]
        seq2[-1].y = graph_sequence[-1].y * 10.0
        out2 = model(seq2)

        # With teacher forcing, predictions should differ meaningfully.
        # The first horizon prediction is NOT affected (uses start token),
        # but horizons 2..5 use previous target as input.
        assert not torch.allclose(out1[:, 1:], out2[:, 1:])

    def test_autoregressive_inference_no_target_dependency(
        self, graph_sequence
    ):
        """En modo eval, los targets no deben afectar el output."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=16, lstm_hidden=32,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        model.eval()

        out1 = model(graph_sequence)

        seq2 = [g.clone() for g in graph_sequence]
        seq2[-1].y = graph_sequence[-1].y * 10.0
        out2 = model(seq2)

        # In eval, autoregressive decoding ignores targets
        assert torch.allclose(out1, out2)

    def test_separate_output_heads(self):
        """Cada horizonte tiene su propia Linear head."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=8, lstm_hidden=16,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        assert len(model.output_heads) == 5
        # Todas las heads son Linear(lstm_hidden, 1)
        for head in model.output_heads:
            assert isinstance(head, torch.nn.Linear)
            assert head.in_features == 16
            assert head.out_features == 1

    def test_start_token_parameter(self):
        """El start token es un parámetro aprendible."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=8, lstm_hidden=16,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        assert model.start_token.requires_grad
        assert model.start_token.shape == (1, 1, 1)


# -- Tests de integración con SequenceGraphTrainer --------------------------


class TestSeq2SeqWithTrainer:
    """Tests que verifican que Seq2SeqGNN entrena con el trainer existente."""

    def _make_sequences(self, n_seqs=3):
        """Crea secuencias sintéticas."""
        sequences = []
        for _ in range(n_seqs):
            seq = []
            for _ in range(6):
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

    def test_trains_with_sequence_trainer(self):
        """Seq2SeqGNN se integra con SequenceGraphTrainer sin cambios."""
        model = Seq2SeqGNN(
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

    def test_loss_decreases(self):
        """La pérdida disminuye tras varias épocas sobre datos fijos."""
        torch.manual_seed(42)
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=8, lstm_hidden=16,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
            dropout=0.0,
        )
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
        """validate() pone el modelo en eval (autoregressive)."""
        model = Seq2SeqGNN(
            input_dim=6, gnn_hidden=8, lstm_hidden=16,
            num_heads=2, num_gnn_layers=2, num_horizons=5,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SequenceGraphTrainer(
            model=model, optimizer=optimizer,
            criterion=torch.nn.MSELoss(),
            device=torch.device("cpu"),
        )

        sequences = self._make_sequences(n_seqs=2)
        _ = trainer.validate(sequences)
        # After validate, model should be in eval mode
        assert not model.training
