"""Single model factory + shared registries.

Before this module, ``scripts/train.py`` and ``scripts/evaluate.py``
duplicated the ``build_model`` function and the constant sets
(``MODEL_REGISTRY``, ``GRAPH_MODELS``, ``MULTI_HORIZON_MODELS``,
``SEQUENCE_MODELS``). The duplication contributed to real bugs — for
example, the ``evaluate.py`` version did not propagate ``activation`` to
``DenseNN``. Centralizing here avoids the divergence.

Conventions:
    - Model keys are lowercase strings (``"dense_nn"``, ``"basic_gcn"``,
      ``"multi_horizon_gat"``, ``"spatiotemporal_gnn"``,
      ``"seq2seq_gnn"``).
    - ``build_model(config, input_dim, edge_dim=None)`` is the only
      supported path for instantiating models from config.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.basic_gcn import BasicGCN
from src.models.dense_nn import DenseNN
from src.models.multi_horizon_gat import MultiHorizonGAT
from src.models.seq2seq_gnn import Seq2SeqGNN
from src.models.spatiotemporal_gnn import SpatioTemporalGNN

# Full registry of models by logical name → class.
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "dense_nn": DenseNN,
    "basic_gcn": BasicGCN,
    "multi_horizon_gat": MultiHorizonGAT,
    "spatiotemporal_gnn": SpatioTemporalGNN,
    "seq2seq_gnn": Seq2SeqGNN,
}

# Models that receive PyG graphs (instead of tabular tensors).
GRAPH_MODELS: set[str] = {
    "basic_gcn",
    "multi_horizon_gat",
    "spatiotemporal_gnn",
    "seq2seq_gnn",
}

# Models that produce a per-horizon target tensor (multi-horizon).
MULTI_HORIZON_MODELS: set[str] = {
    "multi_horizon_gat",
    "spatiotemporal_gnn",
    "seq2seq_gnn",
}

# Models that process sequences of snapshots (not a single snapshot).
SEQUENCE_MODELS: set[str] = {"spatiotemporal_gnn", "seq2seq_gnn"}


def build_model(
    config: dict,
    input_dim: int,
    edge_dim: int | None = None,
) -> nn.Module:
    """Instantiate a model from the full configuration.

    Args:
        config: Full config dictionary (with ``model``, ``graph``, etc.
            sections).
        input_dim: Number of features per node (or per tabular sample).
        edge_dim: Dimension of ``edge_attr`` for layers that support it
            (``GATv2Conv``). Pass ``None`` for tabular models or
            ``BasicGCN``, where it is ignored.

    Returns:
        An ``nn.Module`` instance ready to be moved to a device.

    Raises:
        ValueError: If ``config['model']['name']`` is not in
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
        "prediction_horizons",
        [1, 2, 4, 6, 8],
    )

    # Multi-task (W2): we enable the model's 3 output channels when
    # ``training.loss == "multi_task"``. Living in the factory keeps the
    # decision in a single place — the config says what to train and the
    # factory configures the model accordingly.
    output_channels = _resolve_output_channels(config, model_name)

    if model_name == "multi_horizon_gat":
        return MultiHorizonGAT(
            input_dim=input_dim,
            hidden_channels=model_config.get("hidden_channels", 64),
            num_heads=model_config.get("num_heads", 4),
            num_layers=model_config.get("num_layers", 3),
            num_horizons=len(horizons),
            dropout=model_config.get("dropout", 0.3),
            edge_dim=edge_dim,
            output_channels=output_channels,
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
            output_channels=output_channels,
        )

    if model_name == "seq2seq_gnn":
        # Backward compat: the model was redefined in W3 (Spatio-Temporal
        # Transformer + horizon-query decoder). The new keys are
        # ``hidden_dim`` / ``num_spatial_layers`` / ``num_temporal_layers``.
        # If the config still carries the old ones (``gnn_hidden`` /
        # ``num_gnn_layers``) we use them as a fallback. ``lstm_hidden`` has
        # no equivalent — the new model uses no LSTM — and is silently
        # ignored so legacy configs do not break.
        hidden_dim = model_config.get(
            "hidden_dim",
            model_config.get("gnn_hidden", 128),
        )
        num_spatial_layers = model_config.get(
            "num_spatial_layers",
            model_config.get("num_gnn_layers", 3),
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
            output_channels=output_channels,
        )

    raise ValueError(
        f"Unrecognized model: '{model_name}'. "
        f"Available: {sorted(MODEL_REGISTRY.keys())}"
    )


# Number of channels in the W2 multi-task head — three tasks: arr_delay
# (primary regression), dep_delay (auxiliary regression) and
# pct_arr_delayed_15 (classification). Matches ``NUM_TARGET_CHANNELS`` in
# graph_builder but is re-declared here to avoid introducing a circular
# import of the data layer into the factory.
_MULTI_TASK_OUTPUT_CHANNELS = 3


def _resolve_output_channels(config: dict, model_name: str) -> int:
    """Determine ``output_channels`` for the multi-horizon models.

    Returns 3 if and only if the model is multi-horizon and the configured
    loss is ``"multi_task"`` (the only supported way to consume the third
    channel). In any other case it returns 1 (historical ``[N, H]`` output).
    """
    if model_name not in MULTI_HORIZON_MODELS:
        return 1
    loss_name = (config.get("training", {}) or {}).get("loss")
    return _MULTI_TASK_OUTPUT_CHANNELS if loss_name == "multi_task" else 1
