"""Tests para el módulo de carga de datos."""

import pytest

from src.data.loader import load_parquet, load_flight_data


class TestLoadParquet:
    """Tests para la función load_parquet."""

    def test_file_not_found(self):
        """Verifica que lanza error si el archivo no existe."""
        with pytest.raises(FileNotFoundError):
            load_parquet("no_existe.parquet")


class TestLoadFlightData:
    """Tests para la función load_flight_data."""

    def test_file_not_found(self, tmp_path):
        """Verifica que lanza error si no hay datos para el año."""
        with pytest.raises(FileNotFoundError, match="No se encontró"):
            load_flight_data(tmp_path, 2018)
