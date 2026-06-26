"""Tests for the data-loading module."""

import pytest

from src.data.loader import load_parquet, load_flight_data


class TestLoadParquet:
    """Tests for the load_parquet function."""

    def test_file_not_found(self):
        """Check that it raises an error if the file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_parquet("does_not_exist.parquet")


class TestLoadFlightData:
    """Tests for the load_flight_data function."""

    def test_file_not_found(self, tmp_path):
        """Check that it raises an error if there is no data for the year."""
        with pytest.raises(FileNotFoundError, match="No data file found"):
            load_flight_data(tmp_path, 2018)
