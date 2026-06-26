"""Tests for the MultiHorizonGAT model and the multi-horizon graph_builder."""

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
    """Example airport list."""
    return ["ATL", "DFW", "LAX", "ORD"]


@pytest.fixture
def airport_map(sample_airports):
    """Airport-to-index mapping."""
    return build_airport_mapping(sample_airports)


@pytest.fixture
def sample_flight_df():
    """Synthetic flight DataFrame for testing (5 days, 4 airports)."""
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
    """Simple PyG graph for the model tests."""
    num_nodes = 4
    input_dim = 6
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0, 3],
         [1, 0, 2, 1, 3, 2, 3, 0]], dtype=torch.long
    )
    return x, edge_index


# -- MultiHorizonGAT model tests ---------------------------------------------


class TestMultiHorizonGAT:
    """Tests for the MultiHorizonGAT architecture."""

    def test_output_shape(self, simple_graph):
        """Output must be [num_nodes, num_horizons]."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5,
        )
        out = model(x, edge_index)
        assert out.shape == (4, 5)

    def test_output_shape_with_edge_attr(self, simple_graph):
        """Correct output with multi-channel edge_attr."""
        x, edge_index = simple_graph
        edge_attr = torch.ones(edge_index.shape[1], 5)
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5, edge_dim=5,
        )
        out = model(x, edge_index, edge_attr=edge_attr)
        assert out.shape == (4, 5)

    def test_single_horizon(self, simple_graph):
        """With num_horizons=1, output is [num_nodes, 1]."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=1,
        )
        out = model(x, edge_index)
        assert out.shape == (4, 1)

    def test_gradients_flow(self, simple_graph):
        """Gradients flow through all layers."""
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
                assert param.grad is not None, f"No gradient in {name}"

    def test_different_configs(self):
        """Different configurations of layers, heads and horizons."""
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
        """In eval mode, the output must be deterministic."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5,
        )
        model.eval()

        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
        assert torch.allclose(out1, out2)

    def test_output_channels_widens_to_multitask(self, simple_graph):
        """With output_channels=3 (W2), output is [N, H, 3]."""
        x, edge_index = simple_graph
        model = MultiHorizonGAT(
            input_dim=6, hidden_channels=16, num_heads=2,
            num_layers=3, num_horizons=5, output_channels=3,
        )
        out = model(x, edge_index)
        assert out.shape == (4, 5, 3)

    def test_invalid_output_channels_raises(self):
        """output_channels < 1 must raise ValueError at construction."""
        with pytest.raises(ValueError, match=">= 1"):
            MultiHorizonGAT(
                input_dim=6, hidden_channels=8, num_heads=2,
                num_layers=2, num_horizons=3, output_channels=0,
            )

    def test_more_params_than_gcn(self, simple_graph):
        """GAT with attention must have more parameters than an equivalent GCN."""
        from src.models.basic_gcn import BasicGCN

        gcn = BasicGCN(input_dim=6, hidden_channels=64, num_layers=3)
        gat = MultiHorizonGAT(
            input_dim=6, hidden_channels=64, num_heads=4,
            num_layers=3, num_horizons=5,
        )

        gcn_params = sum(p.numel() for p in gcn.parameters())
        gat_params = sum(p.numel() for p in gat.parameters())
        assert gat_params > gcn_params


# -- Multi-horizon graph_builder tests ----------------------------------------


class TestMultiHorizonGraphBuilder:
    """Tests for multi-horizon graph construction."""

    def test_multi_horizon_targets_shape(self, sample_flight_df, airport_map):
        """Multi-horizon targets are now [num_nodes, num_horizons, 3] (W2)."""
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
        # May be None if there is no data, but if there is it must have the shape
        if targets is not None:
            # W2 multi-task: [N, H, 3] with channels (arr, dep, pct_15).
            assert targets.shape == (len(airport_map), len(horizons), 3)
            assert not torch.isnan(targets).any()
            # pct channel ∈ [0, 1] by construction.
            pct = targets[..., 2]
            assert (pct >= 0.0).all() and (pct <= 1.0).all()

    def test_multi_horizon_targets_match_single(
        self, sample_flight_df, airport_map
    ):
        """Horizon 1 channel 0 (arr) must match compute_node_targets."""
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
                # multi is [N, 1, 3]; channel 0 (arr_delay) replicates single.
                assert torch.allclose(single, multi[:, 0, 0], atol=1e-5)

    def test_create_temporal_graphs_single_horizon(
        self, sample_flight_df, airport_map
    ):
        """Without prediction_horizons, targets are [num_nodes] (backward-compatible)."""
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
        # Single-horizon: targets are 1D
        assert graphs[0].y.dim() == 1
        assert graphs[0].y.shape == (len(airport_map),)

    def test_create_temporal_graphs_multi_horizon(
        self, sample_flight_df, airport_map
    ):
        """With prediction_horizons, targets are [N, num_horizons, 3] (W2)."""
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
        # There may be fewer graphs due to the larger lookahead window
        if len(graphs) > 0:
            assert graphs[0].y.dim() == 3
            assert graphs[0].y.shape == (len(airport_map), len(horizons), 3)

    def test_single_horizon_list_backward_compat(
        self, sample_flight_df, airport_map
    ):
        """With prediction_horizons=[1], single-horizon [num_nodes] is kept."""
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
        # [1] is treated as single-horizon
        assert graphs[0].y.dim() == 1
