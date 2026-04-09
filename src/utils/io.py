"""Utilidades de entrada/salida para rutas y checkpoints.

Provee funciones para resolver rutas del proyecto y gestionar
el guardado/carga de checkpoints de modelos.
"""

from pathlib import Path
from typing import Any

import torch


def get_project_root() -> Path:
    """Devuelve la ruta raíz del proyecto.

    Busca hacia arriba desde este archivo hasta encontrar pyproject.toml.

    Returns:
        Ruta absoluta al directorio raíz del proyecto.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("No se encontró pyproject.toml en los directorios padres")


def get_data_dir(
    subdir: str = "raw",
    config: dict[str, Any] | None = None,
) -> Path:
    """Devuelve la ruta al directorio de datos.

    Si se proporciona config, usa las rutas definidas en data.raw_dir o
    data.processed_dir. Si no, usa las rutas por defecto dentro del proyecto.
    Soporta rutas absolutas (e.g., "E:/TFM-Data/raw") y relativas.

    Args:
        subdir: Tipo de datos ("raw" o "processed").
        config: Diccionario de configuración (opcional).

    Returns:
        Ruta al directorio de datos solicitado.
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
    """Devuelve la ruta al directorio de salida para checkpoints y resultados.

    Returns:
        Ruta al directorio outputs/.
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
) -> None:
    """Guarda un checkpoint del modelo con estado del optimizador y métricas.

    Args:
        model: Modelo PyTorch a guardar.
        optimizer: Optimizador con su estado actual.
        epoch: Época actual del entrenamiento.
        metrics: Diccionario con métricas de evaluación.
        path: Ruta donde guardar el checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Carga un checkpoint y restaura el estado del modelo y optimizador.

    Args:
        path: Ruta al archivo de checkpoint.
        model: Modelo donde cargar los pesos.
        optimizer: Optimizador donde restaurar el estado (opcional).

    Returns:
        Diccionario con epoch y métricas del checkpoint.
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {"epoch": checkpoint["epoch"], "metrics": checkpoint["metrics"]}
