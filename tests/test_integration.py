"""Integration smoke tests for the 5 models in the registry.

Runs a forward + backward on each model with minimal synthetic data (5
nodes, 20 features, 5 horizons, 6 sequence snapshots). Loads no real data —
it only validates that:

1. ``build_model`` instantiates the model from a valid config.
2. The forward accepts the tensors with the shape documented in the README.
3. The loss returns a finite scalar.
4. The backward does not produce NaN or break any layer.
5. ``save_checkpoint`` + ``load_checkpoint`` with ``model_name`` validate
   the architecture.

Designed to run on CPU in <5 s, it serves as a safety net for refactors of
models, factory or trainer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.models.factory import (
    GRAPH_MODELS,
    MULTI_HORIZON_MODELS,
    SEQUENCE_MODELS,
    build_model,
)
from src.training.losses import MultiTaskLoss
from src.utils.io import load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Canonical config for the smoke tests. Small, but with all the keys any
# model in the registry might need.
# ---------------------------------------------------------------------------

NUM_NODES = 5
INPUT_DIM = 20
EDGE_DIM = 5
NUM_EDGES = 8
NUM_HORIZONS = 3
SEQUENCE_LEN = 6


def _make_config(model_name: str) -> dict:
    """Return a minimal valid config for ``build_model``."""
    horizons = list(range(1, NUM_HORIZONS + 1))
    return {
        "model": {
            "name": model_name,
            "dense_nn": {
                "hidden_dims": [16, 8],
                "dropout": 0.1,
                "activation": "relu",
            },
            "basic_gcn": {
                "hidden_channels": 16,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "multi_horizon_gat": {
                "hidden_channels": 16,
                "num_heads": 2,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "spatiotemporal_gnn": {
                "gnn_hidden": 16,
                "lstm_hidden": 16,
                "num_heads": 2,
                "num_gnn_layers": 1,
                "dropout": 0.1,
            },
            "seq2seq_gnn": {
                # Redesigned in W3: new keys (hidden_dim,
                # num_spatial_layers, num_temporal_layers).
                "hidden_dim": 16,
                "num_heads": 2,
                "num_spatial_layers": 1,
                "num_temporal_layers": 1,
                "dropout": 0.1,
            },
        },
        "graph": {
            "prediction_horizons": horizons,
        },
    }


def _make_snapshot() -> Data:
    """Create a single PyG snapshot with node features and edge_attr.

    ``edge_attr`` is generated with ``torch.rand`` (uniform in [0, 1)) so
    that column 0 — used by ``BasicGCN`` as ``edge_weight`` — is
    non-negative. ``GCNConv`` normalizes with ``D^(-1/2) A D^(-1/2)``:
    negative weights produce negative degrees, ``sqrt`` returns NaN and the
    loss breaks. In production all column-0 edge features
    (``flight_count_norm``) are non-negative by construction, so the test
    reflects that contract.
    """
    x = torch.randn(NUM_NODES, INPUT_DIM)
    # Directed chain 0→1→2→3→4 + 3 extra edges for minimal density.
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2, 4, 1], [1, 2, 3, 4, 2, 4, 0, 3]],
        dtype=torch.long,
    )
    edge_attr = torch.rand(NUM_EDGES, EDGE_DIM)
    y = torch.randn(NUM_NODES, NUM_HORIZONS)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_rejects_unknown_model() -> None:
    """``build_model`` must fail with a clear message on an invalid name."""
    with pytest.raises(ValueError, match="Unrecognized model"):
        build_model({"model": {"name": "does_not_exist"}}, input_dim=10)


@pytest.mark.parametrize("model_name", sorted({"dense_nn"}))
def test_tabular_model_smoke(model_name: str) -> None:
    """DenseNN: 1 forward + 1 backward over a tabular batch."""
    config = _make_config(model_name)
    model = build_model(config, input_dim=INPUT_DIM)
    model.train()

    batch_size = 8
    x = torch.randn(batch_size, INPUT_DIM)
    y = torch.randn(batch_size)

    pred = model(x).squeeze(-1)
    assert pred.shape == (batch_size,)

    loss = torch.nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"

    loss.backward()
    # Some gradient must have propagated.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "No parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads), "Gradients with NaN"


@pytest.mark.parametrize(
    "model_name",
    sorted(GRAPH_MODELS - SEQUENCE_MODELS),
)
def test_single_snapshot_graph_model_smoke(model_name: str) -> None:
    """BasicGCN, MultiHorizonGAT: forward+backward over 1 snapshot."""
    config = _make_config(model_name)
    snapshot = _make_snapshot()
    edge_dim = (
        snapshot.edge_attr.shape[1] if model_name in MULTI_HORIZON_MODELS else None
    )
    model = build_model(config, input_dim=INPUT_DIM, edge_dim=edge_dim)
    model.train()

    if model_name == "basic_gcn":
        # BasicGCN does not support multi-channel edge_attr; use the scalar weight.
        edge_weight = snapshot.edge_attr[:, 0]
        pred = model(snapshot.x, snapshot.edge_index, edge_weight)
    else:
        pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

    if model_name in MULTI_HORIZON_MODELS:
        assert pred.shape == (NUM_NODES, NUM_HORIZONS)
        target = snapshot.y
    else:
        # BasicGCN is single-horizon: produces [N, 1] or [N].
        pred = pred.squeeze(-1)
        assert pred.shape == (NUM_NODES,)
        target = snapshot.y[:, 0]

    loss = torch.nn.functional.mse_loss(pred, target)
    assert torch.isfinite(loss), f"{model_name}: non-finite loss"
    loss.backward()


@pytest.mark.parametrize("model_name", sorted(SEQUENCE_MODELS))
def test_sequence_graph_model_smoke(model_name: str) -> None:
    """SpatioTemporalGNN, Seq2SeqGNN: forward+backward over a sequence."""
    config = _make_config(model_name)
    sequence = [_make_snapshot() for _ in range(SEQUENCE_LEN)]
    edge_dim = sequence[0].edge_attr.shape[1]
    model = build_model(config, input_dim=INPUT_DIM, edge_dim=edge_dim)
    model.train()

    pred = model(sequence)
    assert pred.shape == (NUM_NODES, NUM_HORIZONS), (
        f"{model_name}: pred shape {pred.shape} != ({NUM_NODES}, {NUM_HORIZONS})"
    )

    # The last snapshot's target is the real target in the pipeline.
    target = sequence[-1].y
    loss = torch.nn.functional.mse_loss(pred, target)
    assert torch.isfinite(loss), f"{model_name}: non-finite loss"
    loss.backward()


@pytest.mark.parametrize("model_name", sorted(MULTI_HORIZON_MODELS))
def test_multi_task_smoke(model_name: str) -> None:
    """W2: when ``training.loss == "multi_task"`` the factory creates the
    model with ``output_channels=3`` and the forward + backward over
    ``MultiTaskLoss`` with a ``[N, H, 3]`` target must give a finite loss.
    """
    config = _make_config(model_name)
    config["training"] = {"loss": "multi_task"}
    snapshot = _make_snapshot()
    edge_dim = snapshot.edge_attr.shape[1]
    model = build_model(config, input_dim=INPUT_DIM, edge_dim=edge_dim)
    model.train()

    # Multi-channel target [N, H, 3] — pct channel ∈ [0, 1].
    y_arr = torch.randn(NUM_NODES, NUM_HORIZONS)
    y_dep = torch.randn(NUM_NODES, NUM_HORIZONS)
    y_pct = torch.rand(NUM_NODES, NUM_HORIZONS)
    y_multi = torch.stack([y_arr, y_dep, y_pct], dim=-1)
    assert y_multi.shape == (NUM_NODES, NUM_HORIZONS, 3)

    if model_name in SEQUENCE_MODELS:
        sequence = [_make_snapshot() for _ in range(SEQUENCE_LEN)]
        pred = model(sequence)
    else:
        pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

    assert pred.shape == (NUM_NODES, NUM_HORIZONS, 3), (
        f"{model_name}: pred shape {pred.shape} != ({NUM_NODES}, {NUM_HORIZONS}, 3)"
    )

    criterion = MultiTaskLoss(
        main_weight=1.0,
        aux_weight=0.3,
        bce_weight=0.5,
        horizon_weights=[1.0] * NUM_HORIZONS,
    )
    loss = criterion(pred, y_multi)
    assert torch.isfinite(loss), f"{model_name}: non-finite multi_task loss"
    loss.backward()

    # Individual components recorded in the last forward.
    components = criterion.last_components
    for key in ("arr_huber", "dep_huber", "pct_bce", "total"):
        assert key in components, f"missing component {key} in MultiTaskLoss"
        assert np.isfinite(components[key]), f"component {key} non-finite"


def test_checkpoint_roundtrip_with_model_name() -> None:
    """save_checkpoint + load_checkpoint must validate model_name on load."""
    config = _make_config("multi_horizon_gat")
    model = build_model(config, input_dim=INPUT_DIM, edge_dim=EDGE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / "test.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={"val_loss": 0.5},
            path=ckpt_path,
            model_name="multi_horizon_gat",
        )

        # Load with the same name → OK.
        new_model = build_model(config, input_dim=INPUT_DIM, edge_dim=EDGE_DIM)
        load_checkpoint(
            ckpt_path,
            new_model,
            expected_model_name="multi_horizon_gat",
        )

        # Load with a different name → clear error.
        wrong_model = build_model(
            _make_config("dense_nn"),
            input_dim=INPUT_DIM,
        )
        with pytest.raises(ValueError, match="Architecture mismatch"):
            load_checkpoint(
                ckpt_path,
                wrong_model,
                expected_model_name="dense_nn",
            )
