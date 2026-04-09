"""Carga de datos segura en memoria desde archivos Parquet y CSV.

Todas las funciones de este módulo están diseñadas para manejar archivos
de gran tamaño (~7M filas por año) sin cargar todo en memoria a la vez.
Soporta lectura por columnas, muestreo y procesamiento año por año.
"""

import gc
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Tipos de datos optimizados para reducir uso de memoria al leer CSVs
CSV_DTYPES: dict[str, Any] = {
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
    "Month": "Int8",
    "DayOfWeek": "Int8",
    "DayofMonth": "Int8",
}


def load_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Carga un archivo Parquet con pruning de columnas y muestreo opcional.

    Args:
        path: Ruta al archivo Parquet.
        columns: Lista de columnas a leer. None lee todas.
        sample_frac: Fracción de filas a muestrear (0.0-1.0). None lee todas.
        random_seed: Semilla para muestreo reproducible.

    Returns:
        DataFrame con los datos cargados.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info("Cargando Parquet: %s", path.name)
    df = pd.read_parquet(path, columns=columns, engine="pyarrow")
    logger.info("  Filas cargadas: %d, Columnas: %d", len(df), len(df.columns))

    if sample_frac is not None and 0 < sample_frac < 1.0:
        n_before = len(df)
        df = df.sample(frac=sample_frac, random_state=random_seed).reset_index(drop=True)
        logger.info("  Muestreo %.1f%%: %d → %d filas", sample_frac * 100, n_before, len(df))

    return df


def load_csv_chunked(
    path: str | Path,
    columns: list[str] | None = None,
    chunksize: int = 500_000,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Carga un archivo CSV grande en chunks para evitar problemas de memoria.

    Lee el archivo en trozos, aplica muestreo por chunk si es necesario,
    y concatena el resultado final.

    Args:
        path: Ruta al archivo CSV.
        columns: Lista de columnas a leer. None lee todas.
        chunksize: Número de filas por chunk.
        sample_frac: Fracción de filas a muestrear por chunk.
        random_seed: Semilla para muestreo reproducible.

    Returns:
        DataFrame con los datos cargados.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    logger.info("Cargando CSV en chunks: %s", path.name)

    # Filtrar dtypes a solo las columnas solicitadas
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
        logger.info("  Chunk %d: %d filas", i + 1, len(chunk))

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    logger.info("  Total cargado: %d filas", len(df))
    return df


def load_flight_data(
    data_dir: str | Path,
    year: int,
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Carga datos de vuelos para un año, buscando Parquet primero, luego CSV.

    Esta es la función principal de carga. Busca en el siguiente orden:
    1. Parquet: Combined_Flights_{year}.parquet
    2. CSV: Combined_Flights_{year}.csv

    Args:
        data_dir: Directorio donde buscar los archivos.
        year: Año a cargar (2018-2022).
        columns: Columnas a leer.
        sample_frac: Fracción de muestreo.
        random_seed: Semilla aleatoria.

    Returns:
        DataFrame con los datos del año solicitado.
    """
    data_dir = Path(data_dir)

    # Intentar Parquet primero (más eficiente)
    parquet_path = data_dir / f"Combined_Flights_{year}.parquet"
    if parquet_path.exists():
        return load_parquet(parquet_path, columns=columns,
                            sample_frac=sample_frac, random_seed=random_seed)

    # Fallback a CSV anual combinado
    csv_path = data_dir / f"Combined_Flights_{year}.csv"
    if csv_path.exists():
        return load_csv_chunked(csv_path, columns=columns,
                                sample_frac=sample_frac, random_seed=random_seed)

    # Fallback a CSVs mensuales: Flights_{year}_{month}.csv
    monthly_csvs = sorted(data_dir.glob(f"Flights_{year}_*.csv"))
    if monthly_csvs:
        logger.info("Cargando %d archivos mensuales para %d", len(monthly_csvs), year)
        dfs = []
        for csv_file in monthly_csvs:
            chunk_df = load_csv_chunked(csv_file, columns=columns,
                                        sample_frac=sample_frac,
                                        random_seed=random_seed)
            dfs.append(chunk_df)
        result = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        logger.info("Total combinado de CSVs mensuales: %d filas", len(result))
        return result

    raise FileNotFoundError(
        f"No se encontró archivo de datos para {year} en {data_dir}. "
        f"Se buscó: {parquet_path.name}, {csv_path.name}, "
        f"y Flights_{year}_*.csv"
    )


def load_multiple_years(
    data_dir: str | Path,
    years: list[int],
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Carga datos de múltiples años, procesando uno a la vez.

    ADVERTENCIA: Concatenar múltiples años puede consumir mucha memoria.
    Usar sample_frac para limitar el tamaño durante desarrollo.

    Args:
        data_dir: Directorio donde buscar los archivos.
        years: Lista de años a cargar.
        columns: Columnas a leer.
        sample_frac: Fracción de muestreo por año.
        random_seed: Semilla aleatoria.

    Returns:
        DataFrame concatenado de todos los años solicitados.
    """
    dfs = []
    for year in years:
        df = load_flight_data(data_dir, year, columns=columns,
                              sample_frac=sample_frac, random_seed=random_seed)
        dfs.append(df)
        logger.info("Año %d cargado: %d filas", year, len(df))

    result = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    logger.info("Total combinado: %d filas de %d años", len(result), len(years))
    return result
