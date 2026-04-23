"""Tests para el modelo MultiHorizonGAT y el graph_builder multi-horizonte."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.graph_builder import (
    build_airport_mapping,
    build_edge_index,
    compute_multi_horizon_targets,
    compute_node_targets,
    create_temporal_graphs,
)
from src.models.multi_horizon_gat import MultiHorizonGAT


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def sample_airports():
    """Lista de aeropuertos de ejemplo."""
    return ["ATL", "DFW", "LAX", "ORD"]


@pytest.fixture
def airport_map(sample_airports):
    """Mapeo de aeropuertos a indices."""
    return build_airport_mapping(sample_airports)


@pytest.fixture
def sample_flight_df():
    """DataFrame de vuelos sintetico para testing (5 dias, 4 aeropuertos)."""
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
    mask = df["Origin"] != df["Dest"]
    return df[mask].reset_index(drop=True)


@pytest.fixture
def simple_graph():
    """Grafo PyG simple para tests del modelo."""
    num_nodes = 4
    input_dim = 6
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0, 3],
         [1, 0, 2, 1, 3, 2, 3, 0]], dtype=torch.long
    )
    return x, edge_index


# -- Tests del modelo MultiHorizonGAT ----------------------------------------


class TestMultiHorizonGAT:
    """Tests para la arquitectura MultiHorizonGAT."""

    def test_output_shape(self, simple_graph):
        """Output debe ser [num_nodes, num_horizons]."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5,
        )
        out = model(x, edge_index)
        assert out.shape == (4, 5)

    def test_output_shape_with_edge_attr(self, simple_graph):
        """Output correcto con edge_attr multi-canal."""
        x, edge_index = simple_graph
        edge_attr = torch.ones(edge_index.shape[1], 5)
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5, edge_dim=5,
        )
        out = model(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (4, 5)

    def test_single_horizon(self, simple_graph):
        """Con num_horizons=1, output es [num_nodes, 1]."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=1,
        )
        out = model(x, edge_index)
        assert out.shape == (4, 1)

    def test_gradients_flow(self, simple_graph):
        """Los gradientes fluyen a traves de todas las capas."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5,
        )
        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Sin gradiente en {name}"

    def test_different_configs(self):
        """Distintas configuraciones de capas, cabezas y horizontes."""
        x = torch.randn(4, 6)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        for layers in [2, 3]:
            for heads in [1, 2, 4]:
                for horizons in [1, 3, 5]:
                    model = MultiHorizonGAT(
                        input_dim=6, hidden_channels=8,
                        num_heads=heads, num_layers=layers,
                        num_horizons=horizons, dropout=0.1,
                    )
                    out = model(x, edge_index)
                    assert out.shape == (4, horizons)

    def test_eval_mode_deterministic(self, simple_graph):
        """En modo eval, el output debe ser determinista."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5,
        )
        model.eval()

        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
        assert torch.allclose(out1, out2)

    def test_more_params_than_gcn(self, simple_graph):
        """GAT con atencion debe tener mas parametros que GCN equivalente."""
        from src.models.basic_gcn import BasicGCN

        gcn = BasicGCN(input_dim=6, hidden_channels=64, num_layers=3)
        gat = MultiHorizonGAT(
            input_dim=6, hidden_channels=64, num_heads=4,
            num_layers=3, num_horizons=5,
        )

        gcn_params = sum(p.numel() for p in gcn.parameters())
        gat_params = sum(p.numel() for p in gat.parameters())
        assert gat_params > gcn_params


# -- Tests del graph_builder multi-horizonte ----------------------------------


class TestMultiHorizonGraphBuilder:
    """Tests para la construccion de grafos multi-horizonte."""

    def test_multi_horizon_targets_shape(self, sample_flight_df, airport_map):
        """Targets multi-horizonte deben ser [num_nodes, num_horizons]."""
        df = sample_flight_df.copy()
        df["timestamp"] = pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(
            df["Hour"], unit="h"
        )
        df = df.sort_values("timestamp").reset_index(drop=True)

        current_end = df["timestamp"].min() + pd.Timedelta(hours=6)
        window_delta = pd.Timedelta(hours=6)
        horizons = [1, 2, 3]

        targets = compute_multi_horizon_targets(
            df, airport_map, current_end, window_delta, horizons
        )
        # Puede ser None si no hay datos, pero si hay debe tener la forma
        if targets is not None:
            assert targets.shape == (len(airport_map), len(horizons))
            assert not torch.isnan(targets).any()

    def test_multi_horizon_targets_match_single(
        self, sample_flight_df, airport_map
    ):
        """Horizonte 1 de multi debe coincidir con compute_node_targets."""
        df = sample_flight_df.copy()
        df["timestamp"] = pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(
            df["Hour"], unit="h"
        )
        df = df.sort_values("timestamp").reset_index(drop=True)

        current_end = df["timestamp"].min() + pd.Timedelta(hours=6)
        window_delta = pd.Timedelta(hours=6)

        # Single horizon
        next_mask = (
            (df["timestamp"] >= current_end)
            & (df["timestamp"] < current_end + window_delta)
        )
        next_df = df[next_mask]
        if len(next_df) >= 10:
            single = compute_node_targets(next_df, airport_map)
            multi = compute_multi_horizon_targets(
                df, airport_map, current_end, window_delta, [1]
            )
            if multi is not None:
                assert torch.allclose(single, multi.squeeze(-1), atol=1e-5)

    def test_create_temporal_graphs_single_horizon(
        self, sample_flight_df, airport_map
    ):
        """Sin prediction_horizons, targets son [num_nodes] (retrocompatible)."""
        edge_index, edge_attr_static, edge_pairs = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        graphs = create_temporal_graphs(
            sample_flight_df, airport_map, edge_index,
            edge_attr_static, edge_pairs,
            window_hours=6, delay_threshold=15.0,
            prediction_horizons=None,
        )
        assert len(graphs) > 0
        # Single-horizon: targets son 1D
        assert graphs[0].y.dim() == 1
        assert graphs[0].y.shape == (len(airport_map),)

    def test_create_temporal_graphs_multi_horizon(
        self, sample_flight_df, airport_map
    ):
        """Con prediction_horizons, targets son [num_nodes, num_horizons]."""
        edge_index, edge_attr_static, edge_pairs = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        horizons = [1, 2]
        graphs = create_temporal_graphs(
            sample_flight_df, airport_map, edge_index,
            edge_attr_static, edge_pairs,
            window_hours=6, delay_threshold=15.0,
            prediction_horizons=horizons,
        )
        # Puede haber menos grafos por la mayor ventana de lookahead
        if len(graphs) > 0:
            assert graphs[0].y.dim() == 2
            assert graphs[0].y.shape == (len(airport_map), len(horizons))

    def test_single_horizon_list_backward_compat(
        self, sample_flight_df, airport_map
    ):
        """Con prediction_horizons=[1], se mantiene single-horizon [num_nodes]."""
        edge_index, edge_attr_static, edge_pairs = build_edge_index(
            sample_flight_df, airport_map, min_flights=1
        )
        graphs = create_temporal_graphs(
            sample_flight_df, airport_map, edge_index,
            edge_attr_static, edge_pairs,
            window_hours=6, delay_threshold=15.0,
            prediction_horizons=[1],
        )
        assert len(graphs) > 0
        # [1] es tratado como single-horizon
        assert graphs[0].y.dim() == 1
