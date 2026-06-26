"""Tests for the preprocessing module."""

import pandas as pd

from src.data.preprocessing import (
    clean_flights,
    encode_time,
    fill_delay_nulls,
    filter_top_airports,
)


class TestCleanFlights:
    """Tests for the clean_flights function."""

    def test_removes_cancelled(self, sample_flights_df):
        """Check that it removes cancelled flights."""
        df = sample_flights_df.copy()
        df.loc[0, "Cancelled"] = True
        result = clean_flights(df)
        assert len(result) == len(df) - 1

    def test_removes_null_arr_delay(self, sample_flights_df):
        """Check that it removes rows without ArrDelay."""
        df = sample_flights_df.copy()
        df.loc[0, "ArrDelay"] = None
        result = clean_flights(df)
        assert len(result) == len(df) - 1

    def test_converts_date(self, sample_flights_df):
        """Check that it converts FlightDate to datetime."""
        result = clean_flights(sample_flights_df)
        assert pd.api.types.is_datetime64_any_dtype(result["FlightDate"])


class TestEncodeTime:
    """Tests for the encode_time function."""

    def test_extracts_hour(self, sample_flights_df):
        """Check that it extracts the hour from CRSDepTime correctly."""
        result = encode_time(sample_flights_df)
        assert "Hour" in result.columns
        assert result.loc[0, "Hour"] == 8  # 800 → 8
        assert result.loc[2, "Hour"] == 10  # 1000 → 10


class TestFillDelayNulls:
    """Tests for the fill_delay_nulls function."""

    def test_fills_with_zero(self):
        """Check that it fills delay-column nulls with 0."""
        df = pd.DataFrame({"DepDelay": [5.0, None, 3.0], "Other": [1, 2, 3]})
        result = fill_delay_nulls(df)
        assert result["DepDelay"].isna().sum() == 0
        assert result.loc[1, "DepDelay"] == 0.0


class TestFilterTopAirports:
    """Tests for the filter_top_airports function."""

    def test_filters_correctly(self, sample_flights_df):
        """Check that it filters to the busiest airports."""
        df_filtered, airports = filter_top_airports(sample_flights_df, top_n=3)
        assert len(airports) == 3
        # All flights must have origin and destination in the top airports
        assert df_filtered["Origin"].isin(airports).all()
        assert df_filtered["Dest"].isin(airports).all()
