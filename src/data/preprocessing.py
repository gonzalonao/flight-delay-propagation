"""Cleaning and preprocessing of flight data.

Handles null values, filters cancelled/diverted flights, encodes
categorical variables and prepares the data for modeling.
"""

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the flights DataFrame by removing problematic rows.

    Operations:
    - Remove cancelled and diverted flights (no delay data)
    - Remove rows where ArrDelay is null (target)
    - Convert FlightDate to datetime

    Args:
        df: Raw flights DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    n_initial = len(df)

    # Remove cancelled and diverted
    if "Cancelled" in df.columns:
        df = df[~df["Cancelled"]].copy()
    if "Diverted" in df.columns:
        df = df[~df["Diverted"]].copy()

    # Remove rows without a target
    if "ArrDelay" in df.columns:
        df = df.dropna(subset=["ArrDelay"])

    # Convert date
    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    n_final = len(df)
    logger.info(
        "Cleaning: %d → %d rows (removed %d, %.1f%%)",
        n_initial,
        n_final,
        n_initial - n_final,
        (n_initial - n_final) / max(n_initial, 1) * 100,
    )

    return df.reset_index(drop=True)


def fill_delay_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill null values in delay columns with 0.

    - DepDelay/ArrDelay: nulls usually indicate non-operated flights
      (cancelled). `clean_flights` already removes cancelled ones, but we
      fill them for safety.
    - BTS cause columns (CarrierDelay, WeatherDelay, NASDelay,
      SecurityDelay, LateAircraftDelay): the dataset defines them as NaN
      when ArrDelay < 15 min, not as "unknown delay". Filling with 0 is
      semantically correct.
    - DepDel15/ArrDel15: binary indicators; null ≡ not applicable ≡ 0.

    Args:
        df: Flights DataFrame.

    Returns:
        DataFrame with delay nulls filled.
    """
    delay_columns = [
        "DepDelay",
        "ArrDelay",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        "DepDel15",
        "ArrDel15",
    ]

    for col in delay_columns:
        if col in df.columns:
            n_nulls = df[col].isna().sum()
            if n_nulls > 0:
                df[col] = df[col].fillna(0.0)
                logger.info("  %s: %d nulls filled with 0", col, n_nulls)

    return df


def encode_time(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the hour from the CRSDepTime field (HHMM as an integer).

    The Kaggle dataset stores CRSDepTime as an integer: 1430 = 14:30.
    Extracts the hour by dividing by 100.

    Args:
        df: DataFrame with a CRSDepTime column.

    Returns:
        DataFrame with a 'Hour' column added.
    """
    if "CRSDepTime" in df.columns:
        df["Hour"] = df["CRSDepTime"] // 100
        # Clean out-of-range hours (from corrupt data)
        df["Hour"] = df["Hour"].clip(0, 23)

    return df


def filter_top_airports(
    df: pd.DataFrame,
    top_n: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """Filter the DataFrame to include only the busiest airports.

    Counts flights as origin + destination for each airport and selects the
    top_n by activity.

    Args:
        df: Flights DataFrame.
        top_n: Number of airports to keep.

    Returns:
        Tuple of (filtered DataFrame, list of airport codes).
    """
    # Count total activity per airport (origin + destination)
    origin_counts = df["Origin"].value_counts()
    dest_counts = df["Dest"].value_counts()
    total_counts = origin_counts.add(dest_counts, fill_value=0).sort_values(
        ascending=False
    )

    top_airports = total_counts.head(top_n).index.tolist()

    # Filter: both origin and destination must be in the top airports
    mask = df["Origin"].isin(top_airports) & df["Dest"].isin(top_airports)
    df_filtered = df[mask].reset_index(drop=True)

    logger.info(
        "Filtered to top %d airports: %d → %d rows (%.1f%%)",
        top_n,
        len(df),
        len(df_filtered),
        len(df_filtered) / max(len(df), 1) * 100,
    )

    return df_filtered, sorted(top_airports)


def preprocess_pipeline(
    df: pd.DataFrame,
    top_n_airports: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full preprocessing pipeline.

    Order: cleaning → null filling → hour extraction → airport filter.

    Args:
        df: Raw flights DataFrame.
        top_n_airports: Number of airports to keep.

    Returns:
        Tuple of (preprocessed DataFrame, list of airports).
    """
    logger.info("Starting preprocessing pipeline...")

    df = clean_flights(df)
    df = fill_delay_nulls(df)
    df = encode_time(df)
    df, airports = filter_top_airports(df, top_n=top_n_airports)

    logger.info("Preprocessing complete: %d rows, %d airports", len(df), len(airports))

    return df, airports
