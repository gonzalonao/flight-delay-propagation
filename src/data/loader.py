"""Carga de datos segura en memoria desde archivos Parquet y CSV.

Todas las funciones de este módulo están diseñadas para manejar archivos
de gran tamaño (~7M filas por año) sin cargar todo en memoria a la vez.
Soporta lectura por columnas, muestreo y procesamiento año por año.
"""

import gc
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Tipos de datos optimizados para reducir uso de memoria al leer CSVs.
# Las columnas se agrupan en dos clases por su disponibilidad temporal:
#   - Class A (schedule-side): conocidas a priori. Seguras para features
#     históricas Y futuras (ventana objetivo).
#   - Class B (actual-side): conocidas solo después de que ocurre el vuelo.
#     Solo usables para features de la ventana de input histórica.
CSV_DTYPES: dict[str, Any] = {
    # Identificadores y schedule (Class A — known a priori)
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
    # Estado del vuelo (semánticamente Class A, conocido al cierre de la ventana)
    "Cancelled": bool,
    "Diverted": bool,
    # Actuales (Class B — only safe in historical windows)
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
    # BTS delay-cause decomposition (Class B — NaN cuando ArrDelay < 15 min)
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

    # Filtrar columnas a las que existen en el archivo. Algunos parquets
    # (sobre todo años antiguos) no contienen las columnas BTS de causa de
    # retraso (CarrierDelay, WeatherDelay, NASDelay, SecurityDelay,
    # LateAircraftDelay) ni todas las actuales — pedirlas explícitamente
    # provoca un error duro. Mejor pedir solo lo disponible y avisar.
    columns_to_read: list[str] | None = columns
    if columns is not None:
        available = set(pq.read_schema(path).names)
        missing = [c for c in columns if c not in available]
        columns_to_read = [c for c in columns if c in available]
        if missing:
            logger.warning(
                "  Columnas no presentes en %s (se ignoran): %s",
                path.name, ", ".join(missing),
            )

    df = pd.read_parquet(path, columns=columns_to_read, engine="pyarrow")
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
    skip_missing: bool = False,
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
        skip_missing: Si True, ignora con un warning los años cuyo archivo
            no exista. Útil cuando el config declara ``[2018, 2019]`` pero
            sólo uno está descargado en disco — preferimos seguir adelante
            con lo que haya en vez de abortar el entrenamiento.

    Returns:
        DataFrame concatenado de todos los años cargados con éxito.

    Raises:
        FileNotFoundError: Si ``skip_missing=False`` y falta algún año, o si
            ``skip_missing=True`` pero ningún año está disponible.
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
                "Año %d no encontrado en %s — se omite (skip_missing=True). "
                "Detalle: %s", year, data_dir, exc,
            )
            continue
        dfs.append(df)
        loaded_years.append(year)
        logger.info("Año %d cargado: %d filas", year, len(df))

    if not dfs:
        raise FileNotFoundError(
            f"Ningún año de {years} encontrado en {data_dir}. "
            "Verifica que los Parquet/CSV estén descargados."
        )

    result = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    logger.info(
        "Total combinado: %d filas de años %s (solicitados: %s)",
        len(result), loaded_years, years,
    )
    return result
