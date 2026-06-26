"""YAML configuration loading and management for the project.

Supports a base configuration file (default.yaml) with optional overrides
from a local file (local.yaml) for machine-specific paths. The local.yaml
file is gitignored.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    """Replace ``${VAR}`` with ``os.environ[VAR]`` in strings (recursive).

    If the variable is not defined, leaves the placeholder intact (lets you
    catch errors in local runs without breaking the configs).
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries, prioritizing override.

    Args:
        base: Base dictionary.
        override: Dictionary whose values override those of base.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config, merging default.yaml + per-model + local.yaml.

    Precedence order (lowest to highest):
        1. ``configs/default.yaml`` — defaults shared by all models
           (e.g., ``graph.cache_dir``, ``data.raw_dir``, etc.). Loaded
           automatically if it exists in the same directory as
           ``config_path`` and is NOT the requested file itself.
        2. ``config_path`` — the specific config passed via --config.
        3. ``local.yaml`` — per-machine overrides (local paths). Gitignored.

    This avoids the historical bug where keys declared only in
    ``default.yaml`` (for example ``graph.cache_dir``) were lost when a
    per-model config was loaded, silently disabling the snapshot cache and
    forcing it to be regenerated on every run.

    Args:
        config_path: Path to the specific YAML config file.

    Returns:
        Dictionary with the merged configuration parameters.

    Raises:
        FileNotFoundError: If the requested file does not exist.
        yaml.YAMLError: If the file has invalid YAML format.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Layer 1: project-wide defaults (if they exist and are not the file itself).
    config: dict[str, Any] = {}
    default_path = config_path.parent / "default.yaml"
    if default_path.exists() and default_path.resolve() != config_path.resolve():
        with open(default_path, "r", encoding="utf-8") as f:
            default_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, default_config)

    # Layer 2: the requested specific config.
    with open(config_path, "r", encoding="utf-8") as f:
        specific_config = yaml.safe_load(f) or {}
    config = _deep_merge(config, specific_config)

    # Layer 3: per-machine local overrides.
    local_path = config_path.parent / "local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f)
        if local_config:
            config = _deep_merge(config, local_config)

    # Layer 4: expansion of ${VAR} in strings (AzureML paths, etc.).
    config = _expand_env_vars(config)

    return config


def get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Access a nested value using dotted notation.

    Example:
        get_nested(config, "model.dense_nn.hidden_dims") -> [256, 128, 64]

    Args:
        config: Configuration dictionary.
        key: Dotted-notation key (e.g., "model.dense_nn.dropout").
        default: Default value if the key does not exist.

    Returns:
        The value corresponding to the key, or the default value.
    """
    keys = key.split(".")
    value = config
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            return default
        value = value[k]
    return value
