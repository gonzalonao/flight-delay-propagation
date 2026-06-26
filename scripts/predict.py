"""Local batch inference for sequence models (seq2seq_gnn).

Generates offline predictions over a temporal split (``test`` by default)
and dumps them to Parquet with the **same long schema** as the
``predictions_latest`` table in OneLake, plus ground-truth columns so
accuracy can be measured. Reuses the same graph pipeline as ``evaluate.py``
and the assembler in :mod:`src.inference.predictor`.

Usage:
    python scripts/predict.py \
        --checkpoint outputs/runs/20260514-213242/seq2seq_gnn_large.pt \
        --config configs/weekend/seq2seq_gnn_large.yaml \
        --split test \
        --out outputs/powerbi/fact_predictions.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.graph_builder import (
    build_graph_dataset,
    create_temporal_sequences,
    split_graphs_temporal,
)
from src.data.loader import load_multiple_years
from src.data.preprocessing import preprocess_pipeline
from src.inference.predictor import (
    PREDICTION_COLUMNS,
    invert_airport_map,
    predict_to_frame,
)
from src.models.factory import SEQUENCE_MODELS, build_model
from src.utils.config import load_config
from src.utils.device import select_device
from src.utils.io import get_data_dir, load_checkpoint
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)

DEFAULT_CONFIG = "configs/weekend/seq2seq_gnn_large.yaml"
SPLIT_CHOICES = ("train", "val", "test", "all")


def load_preprocessed(config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load and preprocess the configured years (top-N airport filter).

    Returns the already-cleaned DataFrame and the list of top-N airports,
    ready both for aggregations (Power BI export) and for building graphs.
    """
    data_dir = get_data_dir("raw", config)
    years = config["data"].get("years", [2018])
    columns = config["data"].get("columns")
    df = load_multiple_years(
        data_dir, years, columns=columns, sample_frac=None, skip_missing=True,
    )
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)
    return df, airports


def _sequences_for_split(graph_splits, split, input_window) -> list:
    """Build the sliding sequences for the requested split(s)."""
    targets = ["train", "val", "test"] if split == "all" else [split]
    sequences: list = []
    for name in targets:
        graphs = graph_splits.get(name, [])
        if not graphs:
            logger.warning("Split '%s' has no graphs; skipping.", name)
            continue
        seqs = create_temporal_sequences(graphs, input_window)
        logger.info("Split '%s': %d graphs -> %d sequences", name, len(graphs), len(seqs))
        sequences.extend(seqs)
    return sequences


def generate_predictions_from_df(
    df: pd.DataFrame,
    airports: list[str],
    config: dict,
    checkpoint_path: str,
    split: str = "test",
    include_actuals: bool = True,
    device=None,
) -> pd.DataFrame:
    """Build graphs from ``df`` and return the long predictions frame.

    Designed to be reused by ``scripts/export_powerbi.py`` without
    re-loading the dataset. The graph pipeline uses the snapshot cache if
    configured (``graph.cache_dir``), so the second call is fast.

    Args:
        df: Preprocessed DataFrame (output of :func:`load_preprocessed`).
        airports: List of top-N airports.
        config: Configuration (must match the checkpoint's architecture).
        checkpoint_path: Path to the champion's ``.pt`` checkpoint.
        split: ``train`` | ``val`` | ``test`` | ``all``.
        include_actuals: Add ground-truth columns (accuracy).
        device: torch device; if None it is resolved with ``select_device``.

    Returns:
        ``pd.DataFrame`` with the canonical predictions schema.
    """
    model_name = config["model"]["name"]
    if model_name not in SEQUENCE_MODELS:
        raise ValueError(
            f"predict.py only supports sequence models {sorted(SEQUENCE_MODELS)}; "
            f"the config declares model.name='{model_name}'."
        )

    if device is None:
        device = select_device(config)

    graphs, airport_map, _norm_stats = build_graph_dataset(df, airports, config)
    split_cfg = config.get("split", {})
    graph_splits = split_graphs_temporal(
        graphs,
        train_end=split_cfg.get("train_end"),
        val_end=split_cfg.get("val_end"),
    )

    input_window = config.get("graph", {}).get("input_window", 6)
    sequences = _sequences_for_split(graph_splits, split, input_window)
    if not sequences:
        logger.error("No sequences for split '%s'.", split)
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    # Input dimensions from any available graph.
    sample_graph = graphs[0]
    input_dim = sample_graph.x.shape[1]
    sample_ea = sample_graph.edge_attr
    edge_dim = (
        sample_ea.shape[1]
        if sample_ea is not None and sample_ea.dim() == 2 else None
    )
    model = build_model(config, input_dim, edge_dim=edge_dim)
    info = load_checkpoint(checkpoint_path, model, expected_model_name=model_name)
    logger.info("Checkpoint loaded: epoch %s", info.get("epoch"))
    model = model.to(device)

    horizons = config.get("graph", {}).get("prediction_horizons", [1, 2, 4, 6, 8])
    idx_to_iata = invert_airport_map(airport_map)

    preds_df = predict_to_frame(
        model, sequences, horizons, idx_to_iata, device,
        include_actuals=include_actuals,
    )
    logger.info(
        "Predictions generated: %d rows (%d airports x %d horizons x %d instants)",
        len(preds_df),
        preds_df["airport_code"].nunique() if not preds_df.empty else 0,
        len(horizons),
        preds_df["prediction_ts"].nunique() if not preds_df.empty else 0,
    )
    return preds_df


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Local batch inference (seq2seq_gnn)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="YAML config (must match the checkpoint)")
    parser.add_argument("--split", type=str, default="test", choices=SPLIT_CHOICES, help="Temporal split to predict")
    parser.add_argument("--out", type=str, default="outputs/predictions_local.parquet", help="Parquet output path")
    parser.add_argument("--no-actuals", action="store_true", help="Do not include ground-truth columns")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("reproducibility", {}).get("seed", 42))

    logger.info("=" * 60)
    logger.info("LOCAL INFERENCE - %s", config["model"]["name"])
    logger.info("Checkpoint: %s | split=%s", args.checkpoint, args.split)
    logger.info("=" * 60)

    df, airports = load_preprocessed(config)
    preds_df = generate_predictions_from_df(
        df, airports, config, args.checkpoint,
        split=args.split, include_actuals=not args.no_actuals,
    )

    if preds_df.empty:
        logger.error("No predictions; output not written.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_parquet(out_path, index=False)
    logger.info("Predictions written to %s (%d rows)", out_path, len(preds_df))


if __name__ == "__main__":
    main()
