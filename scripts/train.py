"""Main training script.

Loads the configuration, prepares the data, instantiates the model and
runs the training loop.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FlightDelayDataset, create_splits
from src.data.features import build_feature_matrix
from src.data.graph_builder import (
    build_graph_dataset,
    create_temporal_sequences,
    split_graphs_temporal,
)
from src.data.loader import load_multiple_years
from src.data.preprocessing import preprocess_pipeline
from src.evaluation.metrics import (
    evaluate_graph_model,
    evaluate_model,
    evaluate_multi_horizon_graph_model,
    evaluate_multi_horizon_sequence_model,
)
from src.evaluation.reporting import log_multi_horizon_results, log_test_results
from src.models.factory import (
    GRAPH_MODELS,
    MULTI_HORIZON_MODELS,
    SEQUENCE_MODELS,
    build_model,
)
from src.training.graph_trainer import GraphTrainer, SequenceGraphTrainer
from src.training.losses import MultiTaskLoss, WeightedHuberLoss, WeightedMSELoss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.io import get_data_dir, get_output_dir
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a flight-delay prediction model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def _build_optimizer_and_scheduler(
    model: torch.nn.Module, config: dict
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Create the optimizer and scheduler according to the configuration.

    Args:
        model: PyTorch model.
        config: Full configuration.

    Returns:
        Tuple of (optimizer, scheduler or None).
    """
    training_config = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0001),
    )

    scheduler_name = training_config.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.get("epochs", 100)
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
    else:
        scheduler = None

    return optimizer, scheduler


def _build_criterion(config: dict) -> torch.nn.Module:
    """Build the loss function according to ``training.loss``.

    Supports:

    - ``"mse"`` (default if unspecified): ``nn.MSELoss()``.
    - ``"weighted_mse"``: MSE weighted by delay + horizon.
    - ``"weighted_huber"``: Huber weighted by delay + horizon, with a delta
      configurable via ``training.huber_delta`` (default 10 min).
    - ``"multi_task"`` (W2): combines ``Huber(arr) + aux·Huber(dep) +
      bce·BCE(pct_15)`` with weights configurable via the
      ``training.multi_task`` block. It only takes effect on the
      multi-horizon models when ``factory.build_model`` also creates them
      with ``output_channels=3`` (which it does automatically when it sees
      this ``loss``).

    Args:
        config: Full configuration.

    Returns:
        A loss module ready to use in the trainer.
    """
    training_config = config["training"]
    loss_name = training_config.get("loss", "mse")

    if loss_name in ("weighted_mse", "weighted_huber"):
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        delay_weight = training_config.get("delay_weight", 2.0)
        horizon_weights = training_config.get("horizon_weights")
        hw_str = f", horizon_weights={horizon_weights}" if horizon_weights else ""

        if loss_name == "weighted_mse":
            criterion = WeightedMSELoss(
                high_delay_threshold=delay_threshold,
                high_delay_weight=delay_weight,
                horizon_weights=horizon_weights,
            )
            logger.info(
                "Loss: WeightedMSE (threshold=%.0f min, weight=%.1f%s)",
                delay_threshold,
                delay_weight,
                hw_str,
            )
        else:  # weighted_huber
            delta = training_config.get("huber_delta", 10.0)
            criterion = WeightedHuberLoss(
                high_delay_threshold=delay_threshold,
                high_delay_weight=delay_weight,
                horizon_weights=horizon_weights,
                delta=delta,
            )
            logger.info(
                "Loss: WeightedHuber (delta=%.1f min, threshold=%.0f min, "
                "weight=%.1f%s)",
                delta,
                delay_threshold,
                delay_weight,
                hw_str,
            )
        return criterion

    if loss_name == "multi_task":
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        delay_weight = training_config.get("delay_weight", 2.0)
        horizon_weights = training_config.get("horizon_weights")
        delta = training_config.get("huber_delta", 10.0)
        mt_cfg = training_config.get("multi_task", {}) or {}
        criterion = MultiTaskLoss(
            main_weight=mt_cfg.get("main_weight", 1.0),
            aux_weight=mt_cfg.get("aux_weight", 0.3),
            bce_weight=mt_cfg.get("bce_weight", 0.5),
            high_delay_threshold=delay_threshold,
            high_delay_weight=delay_weight,
            horizon_weights=horizon_weights,
            huber_delta=delta,
            bce_pos_weight=mt_cfg.get("bce_pos_weight"),
        )
        logger.info(
            "Loss: MultiTask (main=%.2f, aux=%.2f, bce=%.2f, "
            "huber_delta=%.1f, threshold=%.0f min, delay_weight=%.1f, "
            "horizon_weights=%s, bce_pos_weight=%s)",
            criterion.main_weight,
            criterion.aux_weight,
            criterion.bce_weight,
            delta,
            delay_threshold,
            delay_weight,
            horizon_weights,
            mt_cfg.get("bce_pos_weight"),
        )
        return criterion

    return torch.nn.MSELoss()


def _save_deployment_artifacts(
    output_dir: Path,
    airport_map: dict[str, int],
    norm_stats: dict | None,
    config: dict,
) -> None:
    """Save the artifacts needed for deployment alongside the checkpoint.

    Produces three files in output_dir:
      - airport_map.json    IATA → node-index map (frozen at deploy time)
      - feature_stats.pt    mean and std of the train node features
      - metadata.json       config metrics for champion/challenger

    These files must be uploaded to OneLake together with
    best_{model}_inference.pt (generated by
    deploy/prep_inference_checkpoint.py).
    """
    airport_map_path = output_dir / "airport_map.json"
    with open(airport_map_path, "w") as f:
        json.dump(airport_map, f, indent=2)
    logger.info(
        "airport_map saved: %s (%d airports)", airport_map_path, len(airport_map)
    )

    if norm_stats is not None:
        stats_path = output_dir / "feature_stats.pt"
        torch.save(norm_stats, stats_path)
        logger.info("feature_stats saved: %s", stats_path)
    else:
        logger.warning(
            "normalize_features=False in config → feature_stats.pt not generated. "
            "The inference notebook must apply the same scaling as during training."
        )

    meta = {
        "model_name": config["model"]["name"],
        "input_dim": None,
        "edge_dim": 5,
        "num_airports": len(airport_map),
        "prediction_horizons": config.get("graph", {}).get(
            "prediction_horizons", [1, 2, 4, 6, 8]
        ),
        "normalize_features": config.get("graph", {}).get("normalize_features", False),
        "loss": config.get("training", {}).get("loss", "mse"),
        "checkpoint_file": f"best_{config['model']['name']}_inference.pt",
        "airport_map_file": "airport_map.json",
        "feature_stats_file": "feature_stats.pt" if norm_stats is not None else None,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("metadata.json saved: %s", meta_path)


def _train_tabular(config: dict, df, airports: list[str]) -> None:
    """Training pipeline for tabular models (DenseNN, LSTM)."""
    target_col = config.get("features", {}).get("target", "ArrDelay")
    df, encoders, scaler = build_feature_matrix(df, config)

    splits = create_splits(df, config, target_col=target_col)
    train_features, train_targets = splits["train"]
    val_features, val_targets = splits["val"]

    if len(val_features) == 0:
        raise RuntimeError(
            f"Empty tabular validation split (train={len(train_features)}, "
            f"val=0, test={len(splits['test'][0])}). Check that the years "
            f"{config['data'].get('years')} cover the chronological cutoffs "
            f"{config.get('split', {})}."
        )

    train_dataset = FlightDelayDataset(train_features, train_targets)
    val_dataset = FlightDelayDataset(val_features, val_targets)

    batch_size = config["training"].get("batch_size", 512)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]
    model = build_model(config, input_dim)
    logger.info(
        "Model: %s | Parameters: %d",
        config["model"]["name"],
        sum(p.numel() for p in model.parameters()),
    )

    device = select_device(config)

    training_config = config["training"]
    optimizer, scheduler = _build_optimizer_and_scheduler(model, config)
    criterion = torch.nn.MSELoss()

    output_dir = get_output_dir()
    checkpoint_path = str(output_dir / f"best_{config['model']['name']}.pt")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=training_config.get("gradient_clip", 1.0),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 15),
        checkpoint_path=checkpoint_path,
        model_name=config["model"]["name"],
    )

    # Final evaluation
    if "test" in splits:
        test_features, test_targets = splits["test"]
        test_dataset = FlightDelayDataset(test_features, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        metrics = evaluate_model(model, test_loader, device, delay_threshold)
        log_test_results(metrics, config["model"]["name"], logger)

    logger.info("Training complete. Checkpoint: %s", checkpoint_path)


def _train_graph(config: dict, df, airports: list[str]) -> None:
    """Training pipeline for graph-based models (GCN, GAT)."""
    model_name = config["model"]["name"]
    is_multi_horizon = model_name in MULTI_HORIZON_MODELS

    # For single-horizon models, do not pass prediction_horizons to the builder
    graph_config = config.copy()
    if not is_multi_horizon:
        graph_config = {**config, "graph": {**config.get("graph", {})}}
        graph_config["graph"].pop("prediction_horizons", None)

    # Build temporal graphs
    graphs, airport_map, norm_stats = build_graph_dataset(df, airports, graph_config)
    split_cfg = config.get("split", {})
    graph_splits = split_graphs_temporal(
        graphs,
        train_end=split_cfg.get("train_end"),
        val_end=split_cfg.get("val_end"),
    )

    train_graphs = graph_splits["train"]
    val_graphs = graph_splits["val"]

    if not train_graphs:
        logger.error("No training graphs. Check the data and the configuration.")
        return

    if not val_graphs:
        # Without val_graphs the trainer silently returns val_loss=0.0 each
        # epoch (division by max(0,1)=1) and early stopping never fires. Fail
        # with a clear message indicating the typical cause: the split's
        # chronological cutoffs find no data on disk.
        split_cfg_str = (
            f"train_end={split_cfg.get('train_end')}, "
            f"val_end={split_cfg.get('val_end')}"
        )
        raise RuntimeError(
            f"Empty validation split (total_graphs={len(graphs)}, "
            f"train={len(train_graphs)}, val=0, test={len(graph_splits['test'])}, "
            f"{split_cfg_str}). Typical cause: the config declares "
            f"years={config['data'].get('years')} but only a subset is "
            f"downloaded. Check the files in {get_data_dir('raw', config)}."
        )

    # input_dim comes from the first graph's node features;
    # edge_dim from edge_attr (None if absent).
    input_dim = train_graphs[0].x.shape[1]
    sample_ea = train_graphs[0].edge_attr
    edge_dim = (
        sample_ea.shape[1] if sample_ea is not None and sample_ea.dim() == 2 else None
    )
    model = build_model(config, input_dim, edge_dim=edge_dim)
    logger.info(
        "Model: %s | Parameters: %d | Nodes: %d | edge_dim: %s",
        model_name,
        sum(p.numel() for p in model.parameters()),
        len(airport_map),
        edge_dim,
    )

    device = select_device(config)

    training_config = config["training"]
    optimizer, scheduler = _build_optimizer_and_scheduler(model, config)
    criterion = _build_criterion(config)

    output_dir = get_output_dir()
    checkpoint_path = str(output_dir / f"best_{model_name}.pt")

    trainer = GraphTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=training_config.get("gradient_clip", 1.0),
    )

    trainer.fit(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 15),
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )

    # Final evaluation on the test graphs
    test_graphs = graph_splits["test"]
    if test_graphs:
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )

        if is_multi_horizon:
            horizons = config.get("graph", {}).get(
                "prediction_horizons", [1, 2, 3, 4, 5]
            )
            metrics = evaluate_multi_horizon_graph_model(
                model, test_graphs, device, horizons, delay_threshold
            )
            log_multi_horizon_results(metrics, model_name, horizons, logger)
        else:
            metrics = evaluate_graph_model(model, test_graphs, device, delay_threshold)
            log_test_results(metrics, model_name, logger)

    _save_deployment_artifacts(output_dir, airport_map, norm_stats, config)
    logger.info("Training complete. Checkpoint: %s", checkpoint_path)


def _train_sequence_graph(config: dict, df, airports: list[str]) -> None:
    """Training pipeline for graph-sequence models.

    For models like SpatioTemporalGNN that process sequences of consecutive
    temporal snapshots (e.g., 6 hours of history).
    """
    model_name = config["model"]["name"]

    # Build temporal graphs (multi-horizon)
    graphs, airport_map, norm_stats = build_graph_dataset(df, airports, config)
    split_cfg = config.get("split", {})
    graph_splits = split_graphs_temporal(
        graphs,
        train_end=split_cfg.get("train_end"),
        val_end=split_cfg.get("val_end"),
    )

    # Create sequences per split (avoids leakage between sets)
    input_window = config.get("graph", {}).get("input_window", 6)
    train_sequences = create_temporal_sequences(graph_splits["train"], input_window)
    val_sequences = create_temporal_sequences(graph_splits["val"], input_window)

    if not val_sequences:
        # Same logic as in _train_graph: empty val -> silent val_loss=0.
        raise RuntimeError(
            f"Empty validation split for sequences "
            f"(val_graphs={len(graph_splits['val'])}, "
            f"input_window={input_window}). Check that the date range "
            f"{config.get('split', {}).get('train_end')} → "
            f"{config.get('split', {}).get('val_end')} is covered by the "
            f"downloaded years ({config['data'].get('years')})."
        )

    if not train_sequences:
        logger.error(
            "No training sequences. Check the data and the configuration "
            "(input_window=%d, train_graphs=%d).",
            input_window,
            len(graph_splits["train"]),
        )
        return

    logger.info(
        "Sequences: train=%d, val=%d (input_window=%d)",
        len(train_sequences),
        len(val_sequences),
        input_window,
    )

    # input_dim comes from the first graph's node features;
    # edge_dim from edge_attr (None if absent).
    input_dim = train_sequences[0][0].x.shape[1]
    sample_ea = train_sequences[0][0].edge_attr
    edge_dim = (
        sample_ea.shape[1] if sample_ea is not None and sample_ea.dim() == 2 else None
    )
    model = build_model(config, input_dim, edge_dim=edge_dim)
    logger.info(
        "Model: %s | Parameters: %d | Nodes: %d | Sequence: %d graphs | edge_dim: %s",
        model_name,
        sum(p.numel() for p in model.parameters()),
        len(airport_map),
        input_window,
        edge_dim,
    )

    device = select_device(config)

    training_config = config["training"]
    optimizer, scheduler = _build_optimizer_and_scheduler(model, config)
    criterion = _build_criterion(config)

    output_dir = get_output_dir()
    checkpoint_path = str(output_dir / f"best_{model_name}.pt")

    trainer = SequenceGraphTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        gradient_clip=training_config.get("gradient_clip", 1.0),
    )

    trainer.fit(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        epochs=training_config.get("epochs", 100),
        patience=training_config.get("patience", 15),
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )

    # Final evaluation on the test sequences
    test_sequences = create_temporal_sequences(graph_splits["test"], input_window)
    if test_sequences:
        delay_threshold = config.get("evaluation", {}).get(
            "delay_threshold_minutes", 15
        )
        horizons = config.get("graph", {}).get("prediction_horizons", [1, 2, 3, 4, 5])
        metrics = evaluate_multi_horizon_sequence_model(
            model, test_sequences, device, horizons, delay_threshold
        )
        log_multi_horizon_results(metrics, model_name, horizons, logger)

    _save_deployment_artifacts(output_dir, airport_map, norm_stats, config)
    logger.info("Training complete. Checkpoint: %s", checkpoint_path)


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    config = load_config(args.config)

    # Set the seed for reproducibility
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    model_name = config["model"]["name"]
    logger.info("=" * 60)
    logger.info("TRAINING - %s", model_name)
    logger.info("=" * 60)

    # --- Data loading ---
    data_dir = get_data_dir("raw", config)
    years = config["data"].get("years", [2018])
    columns = config["data"].get("columns")
    sample_frac = config["data"].get("sample_frac")

    logger.info(
        "Loading data: years=%s, sampling=%.1f%%", years, (sample_frac or 1.0) * 100
    )

    # Load ALL years from the config — the splits' chronological cutoff
    # (train < train_end < val < val_end ≤ test) requires the dates falling
    # in each bucket to be present. If the config says [2018, 2019] but only
    # 2018 is loaded, val/test end up empty and val_loss=0 silently.
    df = load_multiple_years(
        data_dir,
        years,
        columns=columns,
        sample_frac=sample_frac,
        random_seed=seed,
        skip_missing=True,
    )

    # --- Preprocessing ---
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)

    # --- Train according to model type ---
    if model_name in SEQUENCE_MODELS:
        _train_sequence_graph(config, df, airports)
    elif model_name in GRAPH_MODELS:
        _train_graph(config, df, airports)
    else:
        _train_tabular(config, df, airports)


if __name__ == "__main__":
    main()
