"""Input/output utilities for paths and checkpoints.

Provides functions to resolve project paths and manage saving/loading of
model checkpoints.
"""

from pathlib import Path
from typing import Any

import torch


def get_project_root() -> Path:
    """Return the project root path.

    Searches upward from this file until it finds pyproject.toml.

    Returns:
        Absolute path to the project root directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("pyproject.toml not found in any parent directory")


def get_data_dir(
    subdir: str = "raw",
    config: dict[str, Any] | None = None,
) -> Path:
    """Return the path to the data directory.

    If config is provided, uses the paths defined in data.raw_dir or
    data.processed_dir. Otherwise, uses the default paths inside the
    project. Supports absolute paths (e.g., "E:/TFM-Data/raw") and relative
    ones.

    Args:
        subdir: Data type ("raw" or "processed").
        config: Configuration dictionary (optional).

    Returns:
        Path to the requested data directory.
    """
    if config is not None:
        key = "raw_dir" if subdir == "raw" else "processed_dir"
        path_str = config.get("data", {}).get(key)
        if path_str:
            data_dir = Path(path_str)
            if not data_dir.is_absolute():
                data_dir = get_project_root() / data_dir
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir

    data_dir = get_project_root() / "data" / subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_output_dir() -> Path:
    """Return the path to the output directory for checkpoints and results.

    Returns:
        Path to the outputs/ directory.
    """
    output_dir = get_project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: str | Path,
    model_name: str | None = None,
) -> None:
    """Save a model checkpoint with optimizer state and metrics.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer with its current state.
        epoch: Current training epoch.
        metrics: Dictionary of evaluation metrics.
        path: Path where the checkpoint is saved.
        model_name: Logical model identifier (e.g., "multi_horizon_gat").
            If provided, ``load_checkpoint`` can validate that the target
            model has the same architecture, avoiding cryptic
            ``state_dict`` errors when the user forgets to pass the correct
            ``--config`` to ``evaluate.py``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if model_name is not None:
        payload["model_name"] = model_name

    torch.save(payload, path)


def load_checkpoint_inference_only(
    path: str | Path,
    model: torch.nn.Module,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load an inference-only checkpoint (without optimizer_state_dict).

    Uses ``weights_only=True`` to avoid executing arbitrary pickle code.
    Requires that the checkpoint was prepared with
    ``deploy/prep_inference_checkpoint.py`` (which removes the
    optimizer_state_dict, which is not deserializable with
    weights_only=True in PyTorch ≥2.1).

    Args:
        path: Path to the inference-only checkpoint file (.pt).
        model: Model into which to load the weights.
        device: Target device ("cpu" or "cuda").

    Returns:
        Dictionary with ``epoch``, ``metrics`` and optionally ``model_name``.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    result: dict[str, Any] = {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }
    if "model_name" in checkpoint:
        result["model_name"] = checkpoint["model_name"]
    return result


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_model_name: str | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore the model and optimizer state.

    If the checkpoint carries ``model_name`` and ``expected_model_name`` is
    passed, the match is validated BEFORE attempting ``load_state_dict`` to
    give a clear error instead of PyTorch's cryptic ``Missing/Unexpected
    keys`` (which typically appears when evaluating a checkpoint without
    passing the correct ``--config``).

    Args:
        path: Path to the checkpoint file.
        model: Model into which to load the weights.
        optimizer: Optimizer into which to restore the state (optional).
        expected_model_name: Expected model name (optional). If the
            checkpoint records a different name, raises ``ValueError``.

    Returns:
        Dictionary with ``epoch``, ``metrics`` and, if available,
        ``model_name``.
    """
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    saved_model_name = checkpoint.get("model_name")
    if (
        expected_model_name is not None
        and saved_model_name is not None
        and saved_model_name != expected_model_name
    ):
        raise ValueError(
            f"Architecture mismatch loading checkpoint '{path}': "
            f"the checkpoint corresponds to model_name='{saved_model_name}' "
            f"but the passed configuration builds '{expected_model_name}'. "
            f"Pass --config configs/{saved_model_name}.yaml or the original "
            f"config it was trained with."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    result: dict[str, Any] = {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
    }
    if saved_model_name is not None:
        result["model_name"] = saved_model_name
    return result
