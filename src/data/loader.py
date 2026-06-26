"""Memory-safe data loading from Parquet and CSV files.

All functions in this module are designed to handle large files (~7M rows
per year) without loading everything into memory at once. Supports
column-wise reading, sampling and year-by-year processing.
"""

import gc
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Data types optimized to reduce memory usage when reading CSVs.
# Columns are grouped into two classes by their temporal availability:
#   - Class A (schedule-side): known a priori. Safe for BOTH historical and
#     future (target window) features.
#   - Class B (actual-side): known only after the flight happens. Only
#     usable for features of the historical input window.
CSV_DTYPES: dict[str, Any] = {
    # Identifiers and schedule (Class A — known a priori)
    "FlightDate": str,
    "Airline": "category",
    "Origin": "category",
    "Dest": "category",
    "CRSDepTime": "Int16",
    "CRSArrTime": "Int16",
    "CRSElapsedTime": "Float32",
    "Distance": "Float32",
    "Month": "Int8",
    "DayOfWeek": "Int8",
    "DayofMonth": "Int8",
    # Flight status (semantically Class A, known at window close)
    "Cancelled": bool,
    "Diverted": bool,
    # Actuals (Class B — only safe in historical windows)
    "DepTime": "Float32",
    "ArrTime": "Float32",
    "DepDelay": "Float32",
    "ArrDelay": "Float32",
    "DepDel15": "Float32",
    "ArrDel15": "Float32",
    "WheelsOff": "Float32",
    "WheelsOn": "Float32",
    "TaxiOut": "Float32",
    "TaxiIn": "Float32",
    "AirTime": "Float32",
    "ActualElapsedTime": "Float32",
    # BTS delay-cause decomposition (Class B — NaN when ArrDelay < 15 min)
    "CarrierDelay": "Float32",
    "WeatherDelay": "Float32",
    "NASDelay": "Float32",
    "SecurityDelay": "Float32",
    "LateAircraftDelay": "Float32",
}


def load_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load a Parquet file with column pruning and optional sampling.

    Args:
        path: Path to the Parquet file.
        columns: List of columns to read. None reads all.
        sample_frac: Fraction of rows to sample (0.0-1.0). None reads all.
        random_seed: Seed for reproducible sampling.

    Returns:
        DataFrame with the loaded data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("Loading Parquet: %s", path.name)

    # Filter columns to those that exist in the file. Some parquet files
    # (especially older years) do not contain the BTS delay-cause columns
    # (CarrierDelay, WeatherDelay, NASDelay, SecurityDelay,
    # LateAircraftDelay) nor all the actuals — requesting them explicitly
    # raises a hard error. Better to request only what is available and warn.
    columns_to_read: list[str] | None = columns
    if columns is not None:
        available = set(pq.read_schema(path).names)
        missing = [c for c in columns if c not in available]
        columns_to_read = [c for c in columns if c in available]
        if missing:
            logger.warning(
                "  Columns not present in %s (ignored): %s",
                path.name, ", ".join(missing),
            )

    df = pd.read_parquet(path, columns=columns_to_read, engine="pyarrow")
    logger.info("  Rows loaded: %d, Columns: %d", len(df), len(df.columns))

    if sample_frac is not None and 0 < sample_frac < 1.0:
        n_before = len(df)
        df = df.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
        logger.info("  Sampling %.1f%%: %d → %d rows", sample_frac * 100, n_before, len(df))

    return df


def load_csv_chunked(
    path: str | Path,
    columns: list[str] | None = None,
    chunksize: int = 500_000,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load a large CSV file in chunks to avoid memory problems.

    Reads the file in chunks, applies per-chunk sampling if needed, and
    concatenates the final result.

    Args:
        path: Path to the CSV file.
        columns: List of columns to read. None reads all.
        chunksize: Number of rows per chunk.
        sample_frac: Fraction of rows to sample per chunk.
        random_seed: Seed for reproducible sampling.

    Returns:
        DataFrame with the loaded data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info("Loading CSV in chunks: %s", path.name)

    # Filter dtypes to only the requested columns
    usecols = columns
    dtype = {k: v for k, v in CSV_DTYPES.items() if columns is None or k in columns}

    chunks = []
    total_rows = 0
    for i, chunk in enumerate(
        pd.read_csv(path, chunksize=chunksize, usecols=usecols, dtype=dtype)
    ):
        if sample_frac is not None and 0 < sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=random_seed + i)
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info("  Chunk %d: %d rows", i + 1, len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    logger.info("  Total loaded: %d rows", len(df))
    return df


def load_flight_data(
    data_dir: str | Path,
    year: int,
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load flight data for a year, trying Parquet first, then CSV.

    This is the main loading function. It searches in the following order:
    1. Parquet: Combined_Flights_{year}.parquet
    2. CSV: Combined_Flights_{year}.csv

    Args:
        data_dir: Directory to search for files.
        year: Year to load (2018-2022).
        columns: Columns to read.
        sample_frac: Sampling fraction.
        random_seed: Random seed.

    Returns:
        DataFrame with the requested year's data.
    """
    data_dir = Path(data_dir)

    # Try Parquet first (more efficient)
    parquet_path = data_dir / f"Combined_Flights_{year}.parquet"
    if parquet_path.exists():
        return load_parquet(parquet_path, columns=columns,
                            sample_frac=sample_frac, random_seed=random_seed)

    # Fallback to combined annual CSV
    csv_path = data_dir / f"Combined_Flights_{year}.csv"
    if csv_path.exists():
        return load_csv_chunked(csv_path, columns=columns,
                                sample_frac=sample_frac, random_seed=random_seed)

    # Fallback to monthly CSVs: Flights_{year}_{month}.csv
    monthly_csvs = sorted(data_dir.glob(f"Flights_{year}_*.csv"))
    if monthly_csvs:
        logger.info("Loading %d monthly files for %d", len(monthly_csvs), year)
        dfs = []
        for csv_file in monthly_csvs:
            chunk_df = load_csv_chunked(csv_file, columns=columns,
                                        sample_frac=sample_frac,
                                        random_seed=random_seed)
            dfs.append(chunk_df)
        result = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        logger.info("Total combined from monthly CSVs: %d rows", len(result))
        return result

    raise FileNotFoundError(
        f"No data file found for {year} in {data_dir}. "
        f"Searched: {parquet_path.name}, {csv_path.name}, "
        f"and Flights_{year}_*.csv"
    )


def load_multiple_years(
    data_dir: str | Path,
    years: list[int],
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """Load data for multiple years, processing one at a time.

    WARNING: Concatenating multiple years can use a lot of memory. Use
    sample_frac to limit the size during development.

    Args:
        data_dir: Directory to search for files.
        years: List of years to load.
        columns: Columns to read.
        sample_frac: Per-year sampling fraction.
        random_seed: Random seed.
        skip_missing: If True, skips with a warning the years whose file
            does not exist. Useful when the config declares ``[2018, 2019]``
            but only one is downloaded to disk — we prefer to proceed with
            what is available instead of aborting training.

    Returns:
        Concatenated DataFrame of all years loaded successfully.

    Raises:
        FileNotFoundError: If ``skip_missing=False`` and a year is missing,
            or if ``skip_missing=True`` but no year is available.
    """
    dfs = []
    loaded_years: list[int] = []
    for year in years:
        try:
            df = load_flight_data(data_dir, year, columns=columns,
                                  sample_frac=sample_frac,
                                  random_seed=random_seed)
        except FileNotFoundError as exc:
            if not skip_missing:
                raise
            logger.warning(
                "Year %d not found in %s — skipping (skip_missing=True). "
                "Detail: %s", year, data_dir, exc,
            )
            continue
        dfs.append(df)
        loaded_years.append(year)
        logger.info("Year %d loaded: %d rows", year, len(df))

    if not dfs:
        raise FileNotFoundError(
            f"No year of {years} found in {data_dir}. "
            "Check that the Parquet/CSV files are downloaded."
        )

    result = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    logger.info(
        "Total combined: %d rows from years %s (requested: %s)",
        len(result), loaded_years, years,
    )
    return result
