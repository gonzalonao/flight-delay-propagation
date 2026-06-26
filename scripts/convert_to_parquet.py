"""Convert Kaggle dataset CSV files to Parquet format.

Reads the CSVs in chunks to avoid memory problems and saves each year as a
compressed Parquet file.

Usage:
    python scripts/convert_to_parquet.py

Only needed if you downloaded the files in CSV format. If you downloaded the
Parquet files directly from Kaggle, this script is not necessary.
"""

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Data types optimized to reduce memory usage
DTYPES = {
    "FlightDate": str,
    "Airline": "category",
    "Origin": "category",
    "Dest": "category",
    "CRSDepTime": "Int16",
    "DepDelay": "Float32",
    "ArrDelay": "Float32",
    "Distance": "Float32",
    "Cancelled": bool,
    "Diverted": bool,
    "AirTime": "Float32",
    "CarrierDelay": "Float32",
    "WeatherDelay": "Float32",
    "NASDelay": "Float32",
    "SecurityDelay": "Float32",
    "LateAircraftDelay": "Float32",
}

CHUNK_SIZE = 500_000


def convert_csv_to_parquet(csv_path: Path, output_path: Path) -> None:
    """Convert a CSV file to Parquet by reading in chunks.

    Args:
        csv_path: Path to the input CSV file.
        output_path: Path to the output Parquet file.
    """
    logger.info("Converting: %s", csv_path.name)

    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype=DTYPES)
    ):
        chunks.append(chunk)
        logger.info("  Chunk %d: %d rows read", i + 1, len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy")

    logger.info(
        "  Saved: %s (%d rows, %.1f MB)",
        output_path.name,
        len(df),
        output_path.stat().st_size / 1e6,
    )

    del df, chunks
    gc.collect()


def main() -> None:
    """Find all CSVs in data/raw/ and convert them to Parquet."""
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("Combined_Flights_*.csv"))

    if not csv_files:
        logger.warning("No CSV files found in %s", raw_dir)
        logger.info(
            "If you downloaded the Parquet files directly, "
            "this script is not necessary."
        )
        return

    for csv_path in csv_files:
        year = csv_path.stem.split("_")[-1]
        output_path = processed_dir / f"flights_{year}.parquet"

        if output_path.exists():
            logger.info("Already exists: %s (skipping)", output_path.name)
            continue

        convert_csv_to_parquet(csv_path, output_path)

    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()
