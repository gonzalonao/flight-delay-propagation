"""Carga y gestión de configuración YAML para el proyecto.

Soporta un archivo de configuración base (default.yaml) con sobreescritura
opcional desde un archivo local (local.yaml) para rutas específicas de cada
máquina. El archivo local.yaml está en .gitignore.
"""

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusiona recursivamente dos diccionarios, priorizando override.

    Args:
        base: Diccionario base.
        override: Diccionario cuyos valores sobreescriben los de base.

    Returns:
        Diccionario fusionado.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Carga configuración YAML, fusionando con local.yaml si existe.

    Busca un archivo local.yaml en el mismo directorio que config_path.
    Si existe, sus valores sobreescriben los del archivo base. Esto permite
    que cada miembro del equipo configure rutas de datos locales sin afectar
    al repositorio.

    Args:
        config_path: Ruta al archivo YAML de configuración base.

    Returns:
        Diccionario con los parámetros de configuración fusionados.

    Raises:
        FileNotFoundError: Si el archivo base no existe.
        yaml.YAMLError: Si el archivo tiene formato YAML inválido.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Buscar y fusionar local.yaml si existe
    local_path = config_path.parent / "local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f)
        if local_config:
            config = _deep_merge(config, local_config)

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
