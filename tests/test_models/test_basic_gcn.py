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

    def test_output_shape_with_edge_attr(self, simple_graph):
        """Verifica output con edge_attr (1D legado y 2D multi-feature)."""
        x, edge_index = simple_graph
        model = BasicGCN(input_dim=6, hidden_channels=16, num_layers=2)

        # 1D (legado)
        edge_attr_1d = torch.ones(edge_index.shape[1])
        out = model(x, edge_index, edge_attr=edge_attr_1d)
        assert out.shape == (4, 1)

        # 2D multi-feature: GCN extrae la primera columna como peso
        edge_attr_2d = torch.ones(edge_index.shape[1], 5)
        out = model(x, edge_index, edge_attr=edge_attr_2d)
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
        """Verifica que edge_index y edge_attr_static tienen shape coherente."""
        edge_index, edge_attr_static, edge_pairs = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == edge_attr_static.shape[0]
        assert edge_attr_static.shape[1] == 3
        assert len(edge_pairs) == edge_index.shape[1]
        # Bidireccional: num_edges debe ser par
        assert edge_index.shape[1] % 2 == 0

    def test_edge_attr_static_normalized(self, sample_flight_df, airport_map):
        """Las features estáticas (count, air_time, distance) están en [0, 1]."""
        _, edge_attr_static, _ = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        assert edge_attr_static.max() <= 1.0 + 1e-6
        assert edge_attr_static.min() >= 0.0

    def test_min_flights_filter(self, sample_flight_df, airport_map):
        """Con min_flights alto, deben crearse menos aristas."""
        _, attrs_low, _ = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        _, attrs_high, _ = build_edge_index(
            sample_flight_df, airport_map, min_flights=100
        )
        assert attrs_high.shape[0] <= attrs_low.shape[0]

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
        edge_index, edge_attr_static, edge_pairs = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        graphs = create_temporal_graphs(
            sample_flight_df, airport_map, edge_index,
            edge_attr_static, edge_pairs,
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
