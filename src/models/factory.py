"""Fábrica única de modelos + registries compartidos.

Antes de este módulo, ``scripts/train.py`` y ``scripts/evaluate.py``
duplicaban la función ``build_model`` y los conjuntos de constantes
(``MODEL_REGISTRY``, ``GRAPH_MODELS``, ``MULTI_HORIZON_MODELS``,
``SEQUENCE_MODELS``). La duplicación contribuyó a errores reales —
por ejemplo, la versión de ``evaluate.py`` no propagaba ``activation``
a ``DenseNN``. Centralizando aquí evitamos la divergencia.

Convenciones:
    - Las claves de modelo son strings minúsculas (``"dense_nn"``,
      ``"basic_gcn"``, ``"multi_horizon_gat"``, ``"spatiotemporal_gnn"``,
      ``"seq2seq_gnn"``).
    - ``build_model(config, input_dim, edge_dim=None)`` es el único
      camino soportado para instanciar modelos desde config.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.basic_gcn import BasicGCN
from src.models.dense_nn import DenseNN
from src.models.multi_horizon_gat import MultiHorizonGAT
from src.models.seq2seq_gnn import Seq2SeqGNN
from src.models.spatiotemporal_gnn import SpatioTemporalGNN

# Registro completo de modelos por nombre lógico → clase.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "dense_nn": DenseNN,
    "basic_gcn": BasicGCN,
    "multi_horizon_gat": MultiHorizonGAT,
    "spatiotemporal_gnn": SpatioTemporalGNN,
    "seq2seq_gnn": Seq2SeqGNN,
}

# Modelos que reciben grafos PyG (en vez de tensores tabulares).
GRAPH_MODELS: set[str] = {
    "basic_gcn", "multi_horizon_gat", "spatiotemporal_gnn", "seq2seq_gnn",
}

# Modelos que producen un tensor de targets por horizonte (multi-horizon).
MULTI_HORIZON_MODELS: set[str] = {
    "multi_horizon_gat", "spatiotemporal_gnn", "seq2seq_gnn",
}

# Modelos que procesan secuencias de snapshots (no un único snapshot).
SEQUENCE_MODELS: set[str] = {"spatiotemporal_gnn", "seq2seq_gnn"}


def build_model(
    config: dict,
    input_dim: int,
    edge_dim: int | None = None,
) -> nn.Module:
    """Instancia un modelo a partir de la configuración completa.

    Args:
        config: Diccionario completo de config (con secciones ``model``,
            ``graph``, etc.).
        input_dim: Número de features por nodo (o por muestra tabular).
        edge_dim: Dimensión de ``edge_attr`` para capas que la soporten
            (``GATv2Conv``). Pasa ``None`` para tabulares o ``BasicGCN``,
            donde se ignora.

    Returns:
        Instancia ``nn.Module`` lista para mover a dispositivo.

    Raises:
        ValueError: Si ``config['model']['name']`` no está en
            ``MODEL_REGISTRY``.
    """
    model_name = config["model"]["name"]
    model_config = config["model"].get(model_name, {}) or {}

    if model_name == "dense_nn":
        return DenseNN(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [256, 128, 64]),
            dropout=model_config.get("dropout", 0.3),
            activation=model_config.get("activation", "relu"),
        )

    if model_name == "basic_gcn":
        return BasicGCN(
            input_dim=input_dim,
            hidden_channels=model_config.get("hidden_channels", 64),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.3),
        )

    horizons = (config.get("graph", {}) or {}).get(
        "prediction_horizons", [1, 2, 4, 6, 8],
    )

    if model_name == "multi_horizon_gat":
        return MultiHorizonGAT(
            input_dim=input_dim,
            hidden_channels=model_config.get("hidden_channels", 64),
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 3),
            num_horizons=len(horizons),
            dropout=model_config.get("dropout", 0.3),
            edge_dim=edge_dim,
        )

    if model_name == "spatiotemporal_gnn":
        return SpatioTemporalGNN(
            input_dim=input_dim,
            gnn_hidden=model_config.get("gnn_hidden", 64),
            lstm_hidden=model_config.get("lstm_hidden", 128),
            num_heads=model_config.get("num_heads", 4),
            num_gnn_layers=model_config.get("num_gnn_layers", 2),
            num_horizons=len(horizons),
            dropout=model_config.get("dropout", 0.3),
            edge_dim=edge_dim,
        )

    if model_name == "seq2seq_gnn":
        # Backward compat: el modelo se redefinió en W3 (Spatio-Temporal
        # Transformer + horizon-query decoder). Las claves nuevas son
        # ``hidden_dim`` / ``num_spatial_layers`` / ``num_temporal_layers``.
        # Si el config aún trae las antiguas (``gnn_hidden`` /
        # ``num_gnn_layers``) las usamos como fallback. ``lstm_hidden`` no
        # tiene equivalente — el nuevo modelo no usa LSTM — y se ignora
        # silenciosamente para no romper configs heredados.
        hidden_dim = model_config.get(
            "hidden_dim", model_config.get("gnn_hidden", 128),
        )
        num_spatial_layers = model_config.get(
            "num_spatial_layers", model_config.get("num_gnn_layers", 3),
        )
        num_temporal_layers = model_config.get("num_temporal_layers", 2)
        return Seq2SeqGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=model_config.get("num_heads", 4),
            num_spatial_layers=num_spatial_layers,
            num_temporal_layers=num_temporal_layers,
            num_horizons=len(horizons),
            dropout=model_config.get("dropout", 0.3),
            edge_dim=edge_dim,
        )

    raise ValueError(
        f"Modelo no reconocido: '{model_name}'. "
        f"Disponibles: {sorted(MODEL_REGISTRY.keys())}"
    )
