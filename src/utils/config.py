"""Carga y gestión de configuración YAML para el proyecto.

Soporta un archivo de configuración base (default.yaml) con sobreescritura
opcional desde un archivo local (local.yaml) para rutas específicas de cada
máquina. El archivo local.yaml está en .gitignore.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    """Sustituye ``${VAR}`` por ``os.environ[VAR]`` en strings (recursivo).

    Si la variable no está definida, deja el placeholder intacto (permite
    detectar errores en runs locales sin romper los configs).
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


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
    """Carga configuración YAML, fusionando default.yaml + per-model + local.yaml.

    Orden de precedencia (de menor a mayor):
        1. ``configs/default.yaml`` — defaults compartidos por todos los modelos
           (e.g., ``graph.cache_dir``, ``data.raw_dir``, etc.). Se carga
           automáticamente si existe en el mismo directorio que ``config_path``
           y NO es el propio archivo solicitado.
        2. ``config_path`` — el config específico pasado por --config.
        3. ``local.yaml`` — overrides por máquina (rutas locales). Está en
           ``.gitignore``.

    Esto evita el bug histórico en el que claves declaradas sólo en
    ``default.yaml`` (por ejemplo ``graph.cache_dir``) se perdían cuando se
    cargaba un config por modelo, deshabilitando silenciosamente el caché
    de snapshots y obligando a regenerarlos en cada ejecución.

    Args:
        config_path: Ruta al archivo YAML de configuración específico.

    Returns:
        Diccionario con los parámetros de configuración fusionados.

    Raises:
        FileNotFoundError: Si el archivo solicitado no existe.
        yaml.YAMLError: Si el archivo tiene formato YAML inválido.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    # Capa 1: defaults globales del proyecto (si existen y no son el propio archivo).
    config: dict[str, Any] = {}
    default_path = config_path.parent / "default.yaml"
    if default_path.exists() and default_path.resolve() != config_path.resolve():
        with open(default_path, "r", encoding="utf-8") as f:
            default_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, default_config)

    # Capa 2: config específico solicitado.
    with open(config_path, "r", encoding="utf-8") as f:
        specific_config = yaml.safe_load(f) or {}
    config = _deep_merge(config, specific_config)

    # Capa 3: overrides locales por máquina.
    local_path = config_path.parent / "local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f)
        if local_config:
            config = _deep_merge(config, local_config)

    # Capa 4: expansión de ${VAR} en strings (paths de AzureML, etc.).
    config = _expand_env_vars(config)

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
