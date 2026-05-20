"""Generate the 3 deployment artifacts for an existing checkpoint.

Re-runs the same data pipeline that ``scripts/train.py`` would (load → preprocess →
``build_graph_dataset``), but stops before training and dumps:

  - ``airport_map.json``   the IATA → node-index map (frozen for inference)
  - ``feature_stats.pt``   per-feature mean/std from the train split
  - ``metadata.json``      model name, horizons, input/edge dims, plus the
                            metrics extracted from the existing checkpoint

This is the fast path when you have a trained ``.pt`` from before train.py
started persisting these alongside the checkpoint.

Usage:
    python scripts/extract_artifacts.py \
        --config     configs/seq2seq_gnn.yaml \
        --checkpoint outputs/best_seq2seq_gnn.pt \
        --output-dir outputs/

The checkpoint must come from the SAME config (same years, same top_n_airports,
same weather settings) — otherwise airport_map / feature_stats will not match
the model weights and inference will produce garbage.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.graph_builder import build_graph_dataset
from src.data.loader import load_multiple_years
from src.data.preprocessing import preprocess_pipeline
from src.utils.config import load_config
from src.utils.io import get_data_dir
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="Same YAML used to train the checkpoint")
    p.add_argument("--checkpoint", required=True, help="Existing trained .pt file")
    p.add_argument("--output-dir", default="outputs", help="Where to write the 3 artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = config.get("reproducibility", {}).get("seed", 42)
    set_seed(seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data load (mirrors scripts/train.py:main) ---
    data_dir = get_data_dir("raw", config)
    years = config["data"].get("years", [2018])
    columns = config["data"].get("columns")
    sample_frac = config["data"].get("sample_frac")

    logger.info("Cargando datos: años=%s", years)
    df = load_multiple_years(
        data_dir, years, columns=columns,
        sample_frac=sample_frac, random_seed=seed,
        skip_missing=True,
    )

    # --- Preprocess + graph build (mirrors _train_sequence_graph) ---
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    df, airports = preprocess_pipeline(df, top_n_airports=top_n)

    logger.info("Construyendo grafos (no se entrena, solo se extraen artefactos)...")
    graphs, airport_map, norm_stats = build_graph_dataset(df, airports, config)

    # --- Read the existing checkpoint to recover metrics + dims ---
    # weights_only=False because legacy checkpoints may contain optimizer state.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    ckpt_model_name = ckpt.get("model_name", config["model"]["name"])

    # --- Persist the 3 artifacts ---
    am_path = output_dir / "airport_map.json"
    with open(am_path, "w") as f:
        json.dump(airport_map, f, indent=2)
    logger.info("airport_map: %s (%d aeropuertos)", am_path, len(airport_map))

    if norm_stats is not None:
        fs_path = output_dir / "feature_stats.pt"
        torch.save(norm_stats, fs_path)
        logger.info("feature_stats: %s", fs_path)
    else:
        logger.warning(
            "normalize_features=False — feature_stats.pt no se genera. "
            "El notebook de inferencia debe replicar la normalización manualmente."
        )

    # Recover input_dim / edge_dim from the first graph in the dataset
    # so metadata.json carries the same numbers train.py would have logged.
    sample = graphs[0]
    input_dim = int(sample.x.shape[1])
    edge_dim = (
        int(sample.edge_attr.shape[1])
        if sample.edge_attr is not None and sample.edge_attr.dim() == 2 else None
    )

    meta = {
        "model_name": ckpt_model_name,
        "input_dim": input_dim,
        "edge_dim": edge_dim if edge_dim is not None else 5,
        "num_airports": len(airport_map),
        "prediction_horizons": config.get("graph", {}).get("prediction_horizons", [1, 2, 4, 6, 8]),
        "normalize_features": config.get("graph", {}).get("normalize_features", False),
        "loss": config.get("training", {}).get("loss", "mse"),
        "checkpoint_file": f"best_{ckpt_model_name}_inference.pt",
        "airport_map_file": "airport_map.json",
        "feature_stats_file": "feature_stats.pt" if norm_stats is not None else None,
        "checkpoint_metrics": metrics,
        "epoch": ckpt.get("epoch"),
        "extracted_from": str(ckpt_path),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("metadata: %s", meta_path)

    logger.info("OK — 3 artefactos listos en %s", output_dir)


if __name__ == "__main__":
    main()
