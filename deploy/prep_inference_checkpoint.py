"""Pre-deployment script: strip optimizer state from a training checkpoint.

Run once after you have a trained checkpoint, before uploading to OneLake:

    python deploy/prep_inference_checkpoint.py \
        --input  outputs/best_seq2seq_gnn.pt \
        --output outputs/best_seq2seq_gnn_inference.pt

The inference-only checkpoint:
  - Contains only model_state_dict, metrics, model_name.
  - Drops optimizer_state_dict (cuts file size by ~3x).
  - Can be loaded with weights_only=True (safe, no arbitrary pickle).
"""

import argparse
import json
from pathlib import Path

import torch


def strip_optimizer(input_path: Path, output_path: Path) -> None:
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    inference_ckpt = {
        "model_state_dict": ckpt["model_state_dict"],
        "metrics": ckpt.get("metrics", {}),
    }
    if "model_name" in ckpt:
        inference_ckpt["model_name"] = ckpt["model_name"]
    if "epoch" in ckpt:
        inference_ckpt["epoch"] = ckpt["epoch"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_ckpt, output_path)

    original_mb = input_path.stat().st_size / 1e6
    stripped_mb = output_path.stat().st_size / 1e6
    print(f"Saved inference checkpoint: {output_path}")
    print(f"  {original_mb:.1f} MB  →  {stripped_mb:.1f} MB  ({100*(1-stripped_mb/original_mb):.0f}% smaller)")
    print(f"  model_name : {inference_ckpt.get('model_name', 'n/a')}")
    print(f"  epoch      : {inference_ckpt.get('epoch', 'n/a')}")
    print(f"  metrics    : {inference_ckpt.get('metrics', {})}")


def export_metadata(ckpt_path: Path, meta_path: Path, airport_map_path: Path) -> None:
    """Write metadata.json alongside the inference checkpoint.

    metadata.json is read by nb_champion_challenger.ipynb to compare MAE
    between champion and challenger without loading the full model.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})

    # Try to load airport_map if it exists alongside the checkpoint.
    airport_map = None
    if airport_map_path.exists():
        with open(airport_map_path) as f:
            airport_map = json.load(f)

    meta = {
        "model_name": ckpt.get("model_name", "seq2seq_gnn"),
        "epoch": ckpt.get("epoch"),
        "metrics": metrics,
        "composite_mae": metrics.get("val_mae"),
        "num_airports": len(airport_map) if airport_map else None,
        "input_dim": 55,
        "edge_dim": 5,
        "checkpoint_file": "best_seq2seq_gnn_inference.pt",
        "airport_map_file": "airport_map.json" if airport_map_path.exists() else None,
        "feature_stats_file": "feature_stats.pt",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip optimizer state from checkpoint")
    parser.add_argument("--input", required=True, help="Path to full training checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path for inference-only checkpoint (.pt)")
    parser.add_argument("--airport-map", default=None, help="Path to airport_map.json (optional)")
    parser.add_argument("--metadata", default=None, help="Path to write metadata.json (optional)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    strip_optimizer(input_path, output_path)

    if args.metadata:
        airport_map_path = Path(args.airport_map) if args.airport_map else Path(args.output).parent / "airport_map.json"
        export_metadata(input_path, Path(args.metadata), airport_map_path)
