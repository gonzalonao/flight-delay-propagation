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
    model_name: str | None = None,
) -> None:
    """Guarda un checkpoint del modelo con estado del optimizador y métricas.

    Args:
        model: Modelo PyTorch a guardar.
        optimizer: Optimizador con su estado actual.
        epoch: Época actual del entrenamiento.
        metrics: Diccionario con métricas de evaluación.
        path: Ruta donde guardar el checkpoint.
        model_name: Identificador lógico del modelo (e.g., "multi_horizon_gat").
            Si se proporciona, ``load_checkpoint`` puede validar que el
            modelo destino tenga la misma arquitectura, evitando errores
            crípticos de ``state_dict`` cuando el usuario olvida pasar el
            ``--config`` correcto a ``evaluate.py``.
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
    """Carga un checkpoint inference-only (sin optimizer_state_dict).

    Usa ``weights_only=True`` para evitar la ejecución de código pickle
    arbitrario. Requiere que el checkpoint haya sido preparado con
    ``deploy/prep_inference_checkpoint.py`` (que elimina el optimizer_state_dict,
    el cual no es deserializable con weights_only=True en PyTorch ≥2.1).

    Args:
        path: Ruta al archivo de checkpoint inference-only (.pt).
        model: Modelo donde cargar los pesos.
        device: Dispositivo destino ("cpu" o "cuda").

    Returns:
        Diccionario con ``epoch``, ``metrics`` y opcionalmente ``model_name``.
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
    """Carga un checkpoint y restaura el estado del modelo y optimizador.

    Si el checkpoint trae ``model_name`` y se pasa ``expected_model_name``,
    se valida la coincidencia ANTES de intentar ``load_state_dict`` para dar
    un error claro en vez del críptico ``Missing/Unexpected keys`` de
    PyTorch (que aparece típicamente cuando se evalúa un checkpoint sin
    pasar el ``--config`` correcto).

    Args:
        path: Ruta al archivo de checkpoint.
        model: Modelo donde cargar los pesos.
        optimizer: Optimizador donde restaurar el estado (opcional).
        expected_model_name: Nombre esperado del modelo (opcional). Si el
            checkpoint registra un nombre distinto, lanza ``ValueError``.

    Returns:
        Diccionario con ``epoch``, ``metrics`` y, si está disponible,
        ``model_name``.
    """
    checkpoint = torch.load(path, weights_only=False)

    saved_model_name = checkpoint.get("model_name")
    if (
        expected_model_name is not None
        and saved_model_name is not None
        and saved_model_name != expected_model_name
    ):
        raise ValueError(
            f"Mismatch de arquitectura al cargar checkpoint '{path}': "
            f"el checkpoint corresponde a model_name='{saved_model_name}' "
            f"pero la configuración pasada construye '{expected_model_name}'. "
            f"Pasa --config configs/{saved_model_name}.yaml o el config "
            f"original con el que se entrenó."
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
