"""Carga y gestión de configuración YAML para el proyecto.

Provee funciones para cargar archivos de configuración y acceder a los
parámetros de forma estructurada mediante un diccionario anidado.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Carga un archivo YAML de configuración y devuelve un diccionario.

    Args:
        config_path: Ruta al archivo YAML de configuración.

    Returns:
        Diccionario con los parámetros de configuración.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el archivo tiene formato YAML inválido.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Accede a un valor anidado usando notación con puntos.

    Ejemplo:
        get_nested(config, "model.dense_nn.hidden_dims") -> [256, 128, 64]

    Args:
        config: Diccionario de configuración.
        key: Clave con notación de puntos (e.g., "model.dense_nn.dropout").
        default: Valor por defecto si la clave no existe.

    Returns:
        El valor correspondiente a la clave, o el valor por defecto.
    """
    keys = key.split(".")
    value = config
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            return default
        value = value[k]
    return value
