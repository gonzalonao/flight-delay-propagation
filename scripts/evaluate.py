"""Evaluation script for trained models.

Loads a checkpoint and runs metrics over the test set.

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/best_dense_nn.pt \
        --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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
from src.evaluation.visualization import (
    plot_error_distribution,
    plot_predictions_vs_actual,
)
from src.models.factory import (
    GRAPH_MODELS,
    MULTI_HORIZON_MODELS,
    SEQUENCE_MODELS,
    build_model,
)
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.io import get_data_dir, load_checkpoint
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate evaluation plots",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    config = load_config(args.config)
    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    model_name = config["model"]["name"]
    logger.info("=" * 60)
    logger.info("EVALUATION - %s", model_name)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("=" * 60)

    # --- Data loading ---
    data_dir = get_data_dir("raw", config)
    years = config["data"].get("years", [2018])
    columns = config["data"].get("columns")

    # For evaluation, NO sampling is used (evaluate over the full data).
    # Load ALL years; if some are missing on disk, proceed with what is
    # available — but warn. Loading only years[0] makes val/test (with
    # chronological cutoffs in 2019) empty when years=[2018,2019].
    df = load_multiple_years(
        data_dir,
        years,
        columns=columns,
        sample_frac=None,
        skip_missing=True,
    )

    # --- Preprocessing ---
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)

    device = select_device(config)
    is_multi_horizon = model_name in MULTI_HORIZON_MODELS

    delay_threshold = config.get("evaluation", {}).get("delay_threshold_minutes", 15)

    if model_name in GRAPH_MODELS:
        # For single-horizon models, do not pass prediction_horizons
        graph_config = config.copy()
        if not is_multi_horizon:
            graph_config = {**config, "graph": {**config.get("graph", {})}}
            graph_config["graph"].pop("prediction_horizons", None)

        # --- Graph pipeline ---
        graphs, airport_map, _norm_stats = build_graph_dataset(
            df, airports, graph_config
        )
        split_cfg = config.get("split", {})
        graph_splits = split_graphs_temporal(
            graphs,
            train_end=split_cfg.get("train_end"),
            val_end=split_cfg.get("val_end"),
        )
        test_graphs = graph_splits["test"]

        if not test_graphs:
            logger.error("No test graphs to evaluate.")
            return

        input_dim = test_graphs[0].x.shape[1]
        sample_ea = test_graphs[0].edge_attr
        edge_dim = (
            sample_ea.shape[1]
            if sample_ea is not None and sample_ea.dim() == 2
            else None
        )
        model = build_model(config, input_dim, edge_dim=edge_dim)

        checkpoint_info = load_checkpoint(
            args.checkpoint,
            model,
            expected_model_name=model_name,
        )
        logger.info(
            "Checkpoint loaded: epoch %d, metrics=%s",
            checkpoint_info["epoch"],
            checkpoint_info["metrics"],
        )

        model = model.to(device)

        if model_name in SEQUENCE_MODELS:
            input_window = config.get("graph", {}).get("input_window", 6)
            test_sequences = create_temporal_sequences(test_graphs, input_window)
            if not test_sequences:
                logger.error(
                    "No test sequences (graphs=%d, window=%d).",
                    len(test_graphs),
                    input_window,
                )
                return
            horizons = config.get("graph", {}).get(
                "prediction_horizons", [1, 2, 3, 4, 5]
            )
            metrics = evaluate_multi_horizon_sequence_model(
                model, test_sequences, device, horizons, delay_threshold
            )
            log_multi_horizon_results(metrics, model_name, horizons, logger)
        elif is_multi_horizon:
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

    else:
        # --- Tabular pipeline ---
        target_col = config.get("features", {}).get("target", "ArrDelay")
        df, encoders, scaler = build_feature_matrix(df, config)

        splits = create_splits(df, config, target_col=target_col)
        test_features, test_targets = splits["test"]

        input_dim = test_features.shape[1]
        model = build_model(config, input_dim)

        checkpoint_info = load_checkpoint(
            args.checkpoint,
            model,
            expected_model_name=model_name,
        )
        logger.info(
            "Checkpoint loaded: epoch %d, metrics=%s",
            checkpoint_info["epoch"],
            checkpoint_info["metrics"],
        )

        model = model.to(device)

        test_dataset = FlightDelayDataset(test_features, test_targets)
        batch_size = config["training"].get("batch_size", 512)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        metrics = evaluate_model(model, test_loader, device, delay_threshold)
        log_test_results(metrics, model_name, logger)

    # --- Optional plots (tabular models only for now) ---
    if args.plots and model_name not in GRAPH_MODELS:
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for features, targets in test_loader:
                preds = model(features.to(device)).squeeze(-1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.numpy())

        predictions = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        plot_predictions_vs_actual(predictions, targets)
        plot_error_distribution(predictions, targets)


if __name__ == "__main__":
    main()
