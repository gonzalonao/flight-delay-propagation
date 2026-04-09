"""Convierte archivos CSV del dataset de Kaggle a formato Parquet.

Lee los CSVs en chunks para evitar problemas de memoria y guarda
cada año como un archivo Parquet comprimido.

Uso:
    python scripts/convert_to_parquet.py

Solo es necesario si descargaste los archivos en formato CSV.
Si descargaste directamente los Parquet de Kaggle, este script no es necesario.
"""

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Tipos de datos optimizados para reducir uso de memoria
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
    """Convierte un archivo CSV a Parquet leyendo en chunks.

    Args:
        csv_path: Ruta al archivo CSV de entrada.
        output_path: Ruta al archivo Parquet de salida.
    """
    logger.info("Convirtiendo: %s", csv_path.name)

    chunks = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype=DTYPES)):
        chunks.append(chunk)
        logger.info("  Chunk %d: %d filas leídas", i + 1, len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy")

    logger.info("  Guardado: %s (%d filas, %.1f MB)", output_path.name, len(df),
                output_path.stat().st_size / 1e6)

    del df, chunks
    gc.collect()


def main() -> None:
    """Busca todos los CSVs en data/raw/ y los convierte a Parquet."""
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("Combined_Flights_*.csv"))

    if not csv_files:
        logger.warning("No se encontraron archivos CSV en %s", raw_dir)
        logger.info("Si descargaste los Parquet directamente, este script no es necesario.")
        return

    for csv_path in csv_files:
        year = csv_path.stem.split("_")[-1]
        output_path = processed_dir / f"flights_{year}.parquet"

        if output_path.exists():
            logger.info("Ya existe: %s (omitiendo)", output_path.name)
            continue

        convert_csv_to_parquet(csv_path, output_path)

    logger.info("Conversión completada.")


if __name__ == "__main__":
    main()
