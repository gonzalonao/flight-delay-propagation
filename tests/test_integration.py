"""Smoke tests de integración para los 5 modelos del registro.

Ejecuta un forward + backward sobre cada modelo con datos sintéticos
mínimos (5 nodos, 20 features, 5 horizontes, 6 snapshots de secuencia).
No carga datos reales — sólo valida que:

1. ``build_model`` instancia el modelo desde un config válido.
2. El forward acepta los tensores con la forma documentada en el README.
3. La loss devuelve un escalar finito.
4. El backward no lanza NaN ni rompe ninguna capa.
5. ``save_checkpoint`` + ``load_checkpoint`` con ``model_name`` validan
   la arquitectura.

Diseñado para correr en CPU en <5 s, sirve como red de seguridad para
refactors de modelos, fábrica o trainer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.models.factory import (
    GRAPH_MODELS,
    MULTI_HORIZON_MODELS,
    SEQUENCE_MODELS,
    build_model,
)
from src.utils.io import load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Config canónica para los smoke tests. Pequeña, pero con todas las claves
# que cualquier modelo del registro pueda necesitar.
# ---------------------------------------------------------------------------

NUM_NODES = 5
INPUT_DIM = 20
EDGE_DIM = 5
NUM_EDGES = 8
NUM_HORIZONS = 3
SEQUENCE_LEN = 6


def _make_config(model_name: str) -> dict:
    """Devuelve un config mínimo válido para ``build_model``."""
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
                "gnn_hidden": 16,
                "lstm_hidden": 16,
                "num_heads": 2,
                "num_gnn_layers": 1,
                "dropout": 0.1,
            },
        },
        "graph": {
            "prediction_horizons": horizons,
        },
    }


def _make_snapshot() -> Data:
    """Crea un único snapshot PyG con node features y edge_attr.

    ``edge_attr`` se genera con ``torch.rand`` (uniforme en [0, 1)) para
    que la columna 0 — usada por ``BasicGCN`` como ``edge_weight`` — sea
    no-negativa. ``GCNConv`` normaliza con ``D^(-1/2) A D^(-1/2)``: pesos
    negativos producen grados negativos, ``sqrt`` devuelve NaN y la loss
    se rompe. En producción todas las features de arista de la columna 0
    (``flight_count_norm``) son no-negativas por construcción, así que
    el test refleja ese contrato.
    """
    x = torch.randn(NUM_NODES, INPUT_DIM)
    # Cadena dirigida 0→1→2→3→4 + 3 aristas extra para densidad mínima.
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 2, 4, 1],
         [1, 2, 3, 4, 2, 4, 0, 3]],
        dtype=torch.long,
    )
    edge_attr = torch.rand(NUM_EDGES, EDGE_DIM)
    y = torch.randn(NUM_NODES, NUM_HORIZONS)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_factory_rejects_unknown_model() -> None:
    """``build_model`` debe fallar con un mensaje claro ante un nombre inválido."""
    with pytest.raises(ValueError, match="Modelo no reconocido"):
        build_model({"model": {"name": "no_existe"}}, input_dim=10)


@pytest.mark.parametrize("model_name", sorted({"dense_nn"}))
def test_tabular_model_smoke(model_name: str) -> None:
    """DenseNN: 1 forward + 1 backward sobre batch tabular."""
    config = _make_config(model_name)
    model = build_model(config, input_dim=INPUT_DIM)
    model.train()

    batch_size = 8
    x = torch.randn(batch_size, INPUT_DIM)
    y = torch.randn(batch_size)

    pred = model(x).squeeze(-1)
    assert pred.shape == (batch_size,)

    loss = torch.nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss), f"Loss no finita: {loss.item()}"

    loss.backward()
    # Algún gradiente debe haberse propagado.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "Ningún parámetro recibió gradiente"
    assert all(torch.isfinite(g).all() for g in grads), "Gradientes con NaN"


@pytest.mark.parametrize(
    "model_name", sorted(GRAPH_MODELS - SEQUENCE_MODELS),
)
def test_single_snapshot_graph_model_smoke(model_name: str) -> None:
    """BasicGCN, MultiHorizonGAT: forward+backward sobre 1 snapshot."""
    config = _make_config(model_name)
    snapshot = _make_snapshot()
    edge_dim = (
        snapshot.edge_attr.shape[1]
        if model_name in MULTI_HORIZON_MODELS else None
    )
    model = build_model(config, input_dim=INPUT_DIM, edge_dim=edge_dim)
    model.train()

    if model_name == "basic_gcn":
        # BasicGCN no soporta edge_attr multi-canal; usa el peso escalar.
        edge_weight = snapshot.edge_attr[:, 0]
        pred = model(snapshot.x, snapshot.edge_index, edge_weight)
    else:
        pred = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

    if model_name in MULTI_HORIZON_MODELS:
        assert pred.shape == (NUM_NODES, NUM_HORIZONS)
        target = snapshot.y
    else:
        # BasicGCN es single-horizon: produce [N, 1] o [N].
        pred = pred.squeeze(-1)
        assert pred.shape == (NUM_NODES,)
        target = snapshot.y[:, 0]

    loss = torch.nn.functional.mse_loss(pred, target)
    assert torch.isfinite(loss), f"{model_name}: loss no finita"
    loss.backward()


@pytest.mark.parametrize("model_name", sorted(SEQUENCE_MODELS))
def test_sequence_graph_model_smoke(model_name: str) -> None:
    """SpatioTemporalGNN, Seq2SeqGNN: forward+backward sobre secuencia."""
    config = _make_config(model_name)
    sequence = [_make_snapshot() for _ in range(SEQUENCE_LEN)]
    edge_dim = sequence[0].edge_attr.shape[1]
    model = build_model(config, input_dim=INPUT_DIM, edge_dim=edge_dim)
    model.train()

    pred = model(sequence)
    assert pred.shape == (NUM_NODES, NUM_HORIZONS), (
        f"{model_name}: pred shape {pred.shape} != ({NUM_NODES}, {NUM_HORIZONS})"
    )

    # El target del último snapshot es el target real en la pipeline.
    target = sequence[-1].y
    loss = torch.nn.functional.mse_loss(pred, target)
    assert torch.isfinite(loss), f"{model_name}: loss no finita"
    loss.backward()


def test_checkpoint_roundtrip_with_model_name() -> None:
    """save_checkpoint + load_checkpoint deben validar model_name al cargar."""
    config = _make_config("multi_horizon_gat")
    snapshot = _make_snapshot()
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

        # Carga con el mismo nombre → OK.
        new_model = build_model(config, input_dim=INPUT_DIM, edge_dim=EDGE_DIM)
        load_checkpoint(
            ckpt_path, new_model, expected_model_name="multi_horizon_gat",
        )

        # Carga con nombre distinto → error claro.
        wrong_model = build_model(
            _make_config("dense_nn"), input_dim=INPUT_DIM,
        )
        with pytest.raises(ValueError, match="Mismatch de arquitectura"):
            load_checkpoint(
                ckpt_path, wrong_model, expected_model_name="dense_nn",
            )
