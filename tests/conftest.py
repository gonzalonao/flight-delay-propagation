"""Shared fixtures for all project tests.

Provides reusable example data and test configurations.
"""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root path."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal configuration for tests."""
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "years": [2018],
            "sample_frac": 0.01,
            "random_seed": 42,
            "columns": [
                "FlightDate", "Airline", "Origin", "Dest",
                "CRSDepTime", "DepDelay", "ArrDelay", "Distance",
            ],
        },
        "features": {
            "target": "ArrDelay",
        },
        "training": {
            "epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "reproducibility": {
            "seed": 42,
        },
    }


@pytest.fixture
def sample_flights_df() -> pd.DataFrame:
    """Return a small DataFrame simulating flight data."""
    return pd.DataFrame({
        "FlightDate": pd.to_datetime(["2018-01-01"] * 10),
        "Airline": ["AA", "UA", "DL", "AA", "UA", "DL", "AA", "UA", "DL", "AA"],
        "Origin": ["ATL", "ORD", "LAX", "ATL", "ORD", "DFW", "JFK", "ATL", "ORD", "LAX"],
        "Dest": ["ORD", "LAX", "ATL", "DFW", "ATL", "ORD", "ATL", "LAX", "DFW", "ORD"],
        "CRSDepTime": [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        "DepDelay": [5.0, -3.0, 45.0, 0.0, 12.0, -5.0, 30.0, 8.0, 0.0, 15.0],
        "ArrDelay": [10.0, -1.0, 50.0, 2.0, 15.0, -3.0, 35.0, 12.0, 5.0, 20.0],
        "Distance": [600, 1750, 1950, 730, 600, 800, 760, 1950, 800, 1750],
        "Cancelled": [False] * 10,
        "Diverted": [False] * 10,
        "AirTime": [90, 210, 250, 100, 90, 110, 105, 250, 110, 210],
    })
