"""Tests for the configuration module."""

import pytest

from src.utils.config import get_nested, load_config


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_default_config(self, project_root):
        """Check that the default.yaml file loads correctly."""
        config = load_config(project_root / "configs" / "default.yaml")
        assert isinstance(config, dict)
        assert "data" in config
        assert "model" in config
        assert "training" in config

    def test_load_nonexistent_file(self):
        """Check that it raises an error if the file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_config("does_not_exist.yaml")


class TestGetNested:
    """Tests for the get_nested function."""

    def test_simple_key(self, sample_config):
        """Access a top-level key."""
        result = get_nested(sample_config, "data")
        assert isinstance(result, dict)

    def test_nested_key(self, sample_config):
        """Access a nested key with dotted notation."""
        result = get_nested(sample_config, "training.learning_rate")
        assert result == 0.001

    def test_missing_key_returns_default(self, sample_config):
        """Return the default value if the key does not exist."""
        result = get_nested(sample_config, "nonexistent.key", default="fallback")
        assert result == "fallback"

    def test_missing_key_returns_none(self, sample_config):
        """Return None if the key does not exist and there is no default."""
        result = get_nested(sample_config, "nonexistent.key")
        assert result is None
