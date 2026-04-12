"""Tests para el modelo BasicGCN y el graph_builder."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.graph_builder import (
    build_airport_mapping,
    build_edge_index,
    compute_node_features,
    compute_node_targets,
    create_temporal_graphs,
    split_graphs_temporal,
)
from src.models.basic_gcn import BasicGCN


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_airports():
    """Lista de aeropuertos de ejemplo."""
    return ["ATL", "DFW", "LAX", "ORD"]


@pytest.fixture
def airport_map(sample_airports):
    """Mapeo de aeropuertos a índices."""
    return build_airport_mapping(sample_airports)


@pytest.fixture
def sample_flight_df():
    """DataFrame de vuelos sintético para testing."""
    np.random.seed(42)
    n = 500
    airports = ["ATL", "DFW", "LAX", "ORD"]

    dates = pd.date_range("2018-01-01", periods=5, freq="D")

    df = pd.DataFrame({
        "FlightDate": np.random.choice(dates, n),
        "Origin": np.random.choice(airports, n),
        "Dest": np.random.choice(airports, n),
        "DepDelay": np.random.normal(5, 20, n),
        "ArrDelay": np.random.normal(5, 20, n),
        "CRSDepTime": np.random.choice(range(600, 2200, 100), n),
        "Hour": np.random.randint(6, 22, n),
    })
    # Evitar rutas de un aeropuerto a sí mismo
    mask = df["Origin"] != df["Dest"]
    return df[mask].reset_index(drop=True)


@pytest.fixture
def simple_graph():
    """Grafo PyG simple para tests del modelo."""
    num_nodes = 4
    input_dim = 6
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 3],
                                [1, 0, 2, 1, 3, 2, 3, 0]], dtype=torch.long)
    return x, edge_index


# ── Tests del modelo BasicGCN ────────────────────────────────────────


class TestBasicGCN:
    """Tests para la arquitectura BasicGCN."""

    def test_output_shape(self, simple_graph):
        """Verifica que la salida tiene shape correcto."""
        x, edge_index = simple_graph
        model = BasicGCN(input_dim=6, hidden_channels=16, num_layers=2)
        out = model(x, edge_index)
        assert out.shape == (4, 1)

    def test_output_shape_with_edge_weight(self, simple_graph):
        """Verifica output con pesos de aristas."""
        x, edge_index = simple_graph
        edge_weight = torch.ones(edge_index.shape[1])
        model = BasicGCN(input_dim=6, hidden_channels=16, num_layers=2)
        out = model(x, edge_index, edge_weight=edge_weight)
        assert out.shape == (4, 1)

    def test_gradients_flow(self, simple_graph):
        """Verifica que los gradientes fluyen correctamente."""
        x, edge_index = simple_graph
        model = BasicGCN(input_dim=6, hidden_channels=16, num_layers=2)
        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente en {name}"

    def test_different_configs(self):
        """Verifica que distintas configuraciones funcionan."""
        x = torch.randn(4, 6)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        for layers in [1, 2, 4]:
            for channels in [8, 32, 64]:
                model = BasicGCN(
                    input_dim=6,
                    hidden_channels=channels,
                    num_layers=layers,
                    dropout=0.5,
                )
                out = model(x, edge_index)
                assert out.shape == (4, 1)

    def test_eval_mode_deterministic(self, simple_graph):
        """En modo eval, el output debe ser determinista."""
        x, edge_index = simple_graph
        model = BasicGCN(input_dim=6, hidden_channels=16, num_layers=2)
        model.eval()

        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
        assert torch.allclose(out1, out2)


# ── Tests del graph_builder ──────────────────────────────────────────


class TestGraphBuilder:
    """Tests para la construcción de grafos."""

    def test_airport_mapping(self, sample_airports):
        """Verifica que el mapeo es correcto y ordenado."""
        mapping = build_airport_mapping(sample_airports)
        assert len(mapping) == 4
        # Debe estar ordenado alfabéticamente
        assert mapping["ATL"] < mapping["DFW"]
        assert mapping["DFW"] < mapping["LAX"]
        assert mapping["LAX"] < mapping["ORD"]

    def test_edge_index_shape(self, sample_flight_df, airport_map):
        """Verifica que edge_index tiene el shape correcto."""
        edge_index, edge_weight = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == edge_weight.shape[0]
        # Bidireccional: num_edges debe ser par
        assert edge_index.shape[1] % 2 == 0

    def test_edge_weight_normalized(self, sample_flight_df, airport_map):
        """Verifica que los pesos están normalizados al rango [0, 1]."""
        _, edge_weight = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        assert edge_weight.max() <= 1.0
        assert edge_weight.min() >= 0.0

    def test_min_flights_filter(self, sample_flight_df, airport_map):
        """Con min_flights alto, deben crearse menos aristas."""
        _, weights_low = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        _, weights_high = build_edge_index(
            sample_flight_df, airport_map, min_flights=100
        )
        assert len(weights_high) <= len(weights_low)

    def test_node_features_shape(self, sample_flight_df, airport_map):
        """Verifica shape de features por nodo."""
        features = compute_node_features(
            sample_flight_df, airport_map, delay_threshold=15.0
        )
        assert features.shape == (len(airport_map), 6)
        # No debe haber NaN
        assert not torch.isnan(features).any()

    def test_node_targets_continuous(self, sample_flight_df, airport_map):
        """Verifica que los targets son continuos (minutos de retraso)."""
        targets = compute_node_targets(sample_flight_df, airport_map)
        assert targets.shape == (len(airport_map),)
        # Targets deben ser continuos (no solo 0/1)
        assert not torch.isnan(targets).any()
        # Deben ser valores reales en minutos (no binarios)
        assert targets.dtype == torch.float32

    def test_temporal_graphs_created(self, sample_flight_df, airport_map):
        """Verifica que se crean snapshots temporales."""
        edge_index, edge_weight = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        graphs = create_temporal_graphs(
            sample_flight_df, airport_map, edge_index, edge_weight,
            window_hours=6, delay_threshold=15.0,
        )
        # Con 5 días de datos y ventanas de 6h, debería haber múltiples grafos
        assert len(graphs) > 0
        # Cada grafo debe tener los atributos esperados
        g = graphs[0]
        assert hasattr(g, "x")
        assert hasattr(g, "edge_index")
        assert hasattr(g, "y")
        assert hasattr(g, "active_mask")

    def test_split_graphs_proportions(self):
        """Verifica las proporciones del split temporal."""
        # Crear lista mock de grafos
        from torch_geometric.data import Data
        graphs = [Data(x=torch.randn(4, 6)) for _ in range(100)]

        splits = split_graphs_temporal(graphs, train_ratio=0.7, val_ratio=0.15)
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15
