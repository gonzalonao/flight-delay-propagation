"""Tests para el módulo de configuración."""

import pytest

from src.utils.config import get_nested, load_config


class TestLoadConfig:
    """Tests para la función load_config."""

    def test_load_default_config(self, project_root):
        """Verifica que el archivo default.yaml se carga correctamente."""
        config = load_config(project_root / "configs" / "default.yaml")
        assert isinstance(config, dict)
        assert "data" in config
        assert "model" in config
        assert "training" in config

    def test_load_nonexistent_file(self):
        """Verifica que lanza error si el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            load_config("no_existe.yaml")


class TestGetNested:
    """Tests para la función get_nested."""

    def test_simple_key(self, sample_config):
        """Accede a una clave de primer nivel."""
        result = get_nested(sample_config, "data")
        assert isinstance(result, dict)

    def test_nested_key(self, sample_config):
        """Accede a una clave anidada con notación de puntos."""
        result = get_nested(sample_config, "training.learning_rate")
        assert result == 0.001

    def test_missing_key_returns_default(self, sample_config):
        """Devuelve el valor por defecto si la clave no existe."""
        result = get_nested(sample_config, "nonexistent.key", default="fallback")
        assert result == "fallback"

    def test_missing_key_returns_none(self, sample_config):
        """Devuelve None si la clave no existe y no hay valor por defecto."""
        result = get_nested(sample_config, "nonexistent.key")
        assert result is None
