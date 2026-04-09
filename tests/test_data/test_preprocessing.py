"""Tests para el módulo de preprocesamiento."""

import pandas as pd
import pytest

from src.data.preprocessing import (
    clean_flights,
    encode_time,
    fill_delay_nulls,
    filter_top_airports,
)


class TestCleanFlights:
    """Tests para la función clean_flights."""

    def test_removes_cancelled(self, sample_flights_df):
        """Verifica que elimina vuelos cancelados."""
        df = sample_flights_df.copy()
        df.loc[0, "Cancelled"] = True
        result = clean_flights(df)
        assert len(result) == len(df) - 1

    def test_removes_null_arr_delay(self, sample_flights_df):
        """Verifica que elimina filas sin ArrDelay."""
        df = sample_flights_df.copy()
        df.loc[0, "ArrDelay"] = None
        result = clean_flights(df)
        assert len(result) == len(df) - 1

    def test_converts_date(self, sample_flights_df):
        """Verifica que convierte FlightDate a datetime."""
        result = clean_flights(sample_flights_df)
        assert pd.api.types.is_datetime64_any_dtype(result["FlightDate"])


class TestEncodeTime:
    """Tests para la función encode_time."""

    def test_extracts_hour(self, sample_flights_df):
        """Verifica que extrae la hora de CRSDepTime correctamente."""
        result = encode_time(sample_flights_df)
        assert "Hour" in result.columns
        assert result.loc[0, "Hour"] == 8   # 800 → 8
        assert result.loc[2, "Hour"] == 10  # 1000 → 10


class TestFillDelayNulls:
    """Tests para la función fill_delay_nulls."""

    def test_fills_with_zero(self):
        """Verifica que rellena nulos en columnas de retraso con 0."""
        df = pd.DataFrame({"DepDelay": [5.0, None, 3.0], "Other": [1, 2, 3]})
        result = fill_delay_nulls(df)
        assert result["DepDelay"].isna().sum() == 0
        assert result.loc[1, "DepDelay"] == 0.0


class TestFilterTopAirports:
    """Tests para la función filter_top_airports."""

    def test_filters_correctly(self, sample_flights_df):
        """Verifica que filtra a los aeropuertos con más tráfico."""
        df_filtered, airports = filter_top_airports(sample_flights_df, top_n=3)
        assert len(airports) == 3
        # Todos los vuelos deben tener origen y destino en top airports
        assert df_filtered["Origin"].isin(airports).all()
        assert df_filtered["Dest"].isin(airports).all()
