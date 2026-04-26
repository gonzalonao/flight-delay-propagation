"""Tests para Seq2SeqGNN (Spatio-Temporal Transformer + horizon-query decoder).

Refleja el rediseño del workstream W3:
    - Constructor: ``hidden_dim``, ``num_spatial_layers``, ``num_temporal_layers``
      (las antiguas ``gnn_hidden`` / ``lstm_hidden`` / ``num_gnn_layers`` se
      mantienen sólo en la fábrica como fallback de retro-compatibilidad).
    - Sin teacher forcing: el output no depende de ``y`` ni en train ni en
      eval; el modelo es idéntico salvo dropout.
    - Sin ``start_token`` ni ``output_heads`` ModuleList: el decoder usa
      embeddings de query por horizonte y un único MLP de salida.
"""

import pytest
import torch
from torch_geometric.data import Data

from src.models.seq2seq_gnn import Seq2SeqGNN, SpatialGATEncoder
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
    # ``edge_attr`` no negativo: si en el futuro un test pasa esto a una
    # capa que normaliza pesos (GCNConv) evitamos NaN. GATv2Conv lo trata
    # como features y acepta valores arbitrarios igualmente.
    edge_attr = torch.rand(edge_index.shape[1], 1)
    y = torch.randn(num_nodes, num_horizons)
    active_mask = torch.ones(num_nodes, dtype=torch.bool)
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=y, active_mask=active_mask,
    )


@pytest.fixture
def graph_sequence(simple_graph):
    """Secuencia de 6 grafos; sólo el último expone target real."""
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
    """Helper para reducir ruido en los tests."""
    return Seq2SeqGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=2,
        num_spatial_layers=num_spatial_layers,
        num_temporal_layers=num_temporal_layers,
        num_horizons=num_horizons,
        dropout=dropout,
        edge_dim=1,  # los fixtures usan edge_attr de 1 canal
    )


# -- Tests del modelo Seq2SeqGNN ---------------------------------------------


class TestSeq2SeqGNN:
    """Tests para la nueva arquitectura Seq2SeqGNN."""

    def test_output_shape_eval(self, graph_sequence):
        """Output debe ser [num_nodes, num_horizons] en modo eval."""
        model = _make_model()
        model.eval()
        out = model(graph_sequence)
        assert out.shape == (4, 5)

    def test_output_shape_train(self, graph_sequence):
        """Output con la misma forma en modo train (no hay distribution shift)."""
        model = _make_model()
        model.train()
        out = model(graph_sequence)
        assert out.shape == (4, 5)

    def test_different_num_horizons(self, graph_sequence):
        """Distintos num_horizons producen outputs del tamaño correcto."""
        for h in [1, 3, 5, 7]:
            seq = [g.clone() for g in graph_sequence]
            seq[-1].y = torch.randn(4, h)
            model = _make_model(num_horizons=h, hidden_dim=8)
            model.eval()
            out = model(seq)
            assert out.shape == (4, h)

    def test_gradients_flow(self, graph_sequence):
        """Los gradientes fluyen a través de encoder espacial, Transformer,
        queries de horizonte, cross-attention y head MLP."""
        model = _make_model()
        model.train()
        out = model(graph_sequence)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente en {name}"

    def test_eval_mode_deterministic(self, graph_sequence):
        """En modo eval el dropout está apagado, así que el output es determinista."""
        model = _make_model()
        model.eval()
        out1 = model(graph_sequence)
        out2 = model(graph_sequence)
        assert torch.allclose(out1, out2)

    def test_targets_have_no_effect_in_eval(self, graph_sequence):
        """Sin teacher forcing: cambiar ``y`` no cambia la predicción en eval."""
        model = _make_model()
        model.eval()

        out1 = model(graph_sequence)

        seq2 = [g.clone() for g in graph_sequence]
        seq2[-1].y = graph_sequence[-1].y * 10.0
        out2 = model(seq2)

        assert torch.allclose(out1, out2), (
            "El output del nuevo Seq2SeqGNN no debe depender de y "
            "(no hay teacher forcing)."
        )

    def test_targets_have_no_effect_in_train(self, graph_sequence):
        """Sin teacher forcing tampoco en train: el modelo nunca lee ``y``."""
        torch.manual_seed(0)
        model = _make_model(dropout=0.0)
        model.train()

        out1 = model(graph_sequence)

        seq2 = [g.clone() for g in graph_sequence]
        seq2[-1].y = graph_sequence[-1].y * 10.0
        out2 = model(seq2)

        assert torch.allclose(out1, out2), (
            "Con dropout=0, train y eval deben coincidir y ningún horizonte "
            "puede depender de y (el rediseño elimina teacher forcing)."
        )

    def test_horizon_queries_are_learnable(self):
        """Los embeddings de query por horizonte son parámetros entrenables."""
        model = _make_model(num_horizons=5)
        assert model.horizon_queries.requires_grad
        assert model.horizon_queries.shape == (5, model.hidden_dim)

    def test_single_mlp_head_not_per_horizon_linears(self):
        """El head es un único MLP — ya no hay una Linear por horizonte."""
        model = _make_model()
        # No debe existir el atributo ``output_heads`` del diseño antiguo.
        assert not hasattr(model, "output_heads")
        # Y sí debe existir un ``head`` que sea Sequential MLP.
        assert isinstance(model.head, torch.nn.Sequential)

    def test_no_lstm_modules(self):
        """El rediseño no usa LSTMs (ni encoder ni decoder)."""
        model = _make_model()
        for module in model.modules():
            assert not isinstance(module, torch.nn.LSTM), (
                "El nuevo Seq2SeqGNN no debe contener LSTMs"
            )

    def test_hidden_dim_must_be_divisible_by_num_heads(self):
        """Constructor falla rápido si hidden_dim % num_heads != 0."""
        with pytest.raises(ValueError, match="divisible por num_heads"):
            Seq2SeqGNN(
                input_dim=6,
                hidden_dim=15,    # 15 no es divisible por 2
                num_heads=2,
                num_spatial_layers=1,
                num_temporal_layers=1,
                num_horizons=3,
            )

    def test_spatial_encoder_supports_single_layer(self, simple_graph):
        """SpatialGATEncoder con num_layers=1 sigue produciendo hidden_dim."""
        enc = SpatialGATEncoder(
            input_dim=6, hidden_dim=16, num_heads=2,
            num_layers=1, dropout=0.0, edge_dim=1,
        )
        out = enc(simple_graph.x, simple_graph.edge_index, simple_graph.edge_attr)
        assert out.shape == (4, 16)


# -- Tests de integración con SequenceGraphTrainer --------------------------


class TestSeq2SeqWithTrainer:
    """Tests que verifican que Seq2SeqGNN entrena con el trainer existente."""

    def _make_sequences(self, n_seqs: int = 3) -> list[list[Data]]:
        """Crea secuencias sintéticas con edge_attr de 1 canal."""
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
        """Seq2SeqGNN se integra con SequenceGraphTrainer sin cambios."""
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
        """La pérdida disminuye tras varias épocas sobre datos fijos."""
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
        """validate() pone el modelo en eval."""
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
