"""Limpieza y preprocesamiento de datos de vuelos.

Maneja valores nulos, filtra vuelos cancelados/desviados,
codifica variables categóricas y prepara los datos para modelado.
"""

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame de vuelos eliminando filas problemáticas.

    Operaciones:
    - Elimina vuelos cancelados y desviados (no tienen datos de retraso)
    - Elimina filas donde ArrDelay es nulo (target)
    - Convierte FlightDate a datetime

    Args:
        df: DataFrame crudo de vuelos.

    Returns:
        DataFrame limpio.
    """
    n_initial = len(df)

    # Eliminar cancelados y desviados
    if "Cancelled" in df.columns:
        df = df[~df["Cancelled"]].copy()
    if "Diverted" in df.columns:
        df = df[~df["Diverted"]].copy()

    # Eliminar filas sin target
    if "ArrDelay" in df.columns:
        df = df.dropna(subset=["ArrDelay"])

    # Convertir fecha
    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    n_final = len(df)
    logger.info(
        "Limpieza: %d → %d filas (eliminadas %d, %.1f%%)",
        n_initial, n_final, n_initial - n_final,
        (n_initial - n_final) / max(n_initial, 1) * 100,
    )

    return df.reset_index(drop=True)


def fill_delay_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena valores nulos en columnas de retraso con 0.

    Las columnas de tipo de retraso (CarrierDelay, WeatherDelay, etc.)
    son nulas cuando no hubo retraso de ese tipo. Se rellenan con 0.

    Args:
        df: DataFrame de vuelos.

    Returns:
        DataFrame con nulos de retraso rellenados.
    """
    delay_columns = [
        "DepDelay", "ArrDelay",
    ]

    for col in delay_columns:
        if col in df.columns:
            n_nulls = df[col].isna().sum()
            if n_nulls > 0:
                df[col] = df[col].fillna(0.0)
                logger.info("  %s: %d nulos rellenados con 0", col, n_nulls)

    return df


def encode_time(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la hora del campo CRSDepTime (formato HHMM como entero).

    El dataset de Kaggle almacena CRSDepTime como entero: 1430 = 14:30.
    Extrae la hora dividiendo entre 100.

    Args:
        df: DataFrame con columna CRSDepTime.

    Returns:
        DataFrame con columna 'Hour' añadida.
    """
    if "CRSDepTime" in df.columns:
        df["Hour"] = df["CRSDepTime"] // 100
        # Limpiar horas fuera de rango (por datos corruptos)
        df["Hour"] = df["Hour"].clip(0, 23)

    return df


def filter_top_airports(
    df: pd.DataFrame,
    top_n: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """Filtra el DataFrame para incluir solo los aeropuertos con más tráfico.

    Cuenta vuelos como origen + destino para cada aeropuerto y selecciona
    los top_n con más actividad.

    Args:
        df: DataFrame de vuelos.
        top_n: Número de aeropuertos a mantener.

    Returns:
        Tupla de (DataFrame filtrado, lista de códigos de aeropuerto).
    """
    # Contar actividad total por aeropuerto (origen + destino)
    origin_counts = df["Origin"].value_counts()
    dest_counts = df["Dest"].value_counts()
    total_counts = origin_counts.add(dest_counts, fill_value=0).sort_values(ascending=False)

    top_airports = total_counts.head(top_n).index.tolist()

    # Filtrar: tanto origen como destino deben estar en top airports
    mask = df["Origin"].isin(top_airports) & df["Dest"].isin(top_airports)
    df_filtered = df[mask].reset_index(drop=True)

    logger.info(
        "Filtrado a top %d aeropuertos: %d → %d filas (%.1f%%)",
        top_n, len(df), len(df_filtered),
        len(df_filtered) / max(len(df), 1) * 100,
    )

    return df_filtered, sorted(top_airports)


def preprocess_pipeline(
    df: pd.DataFrame,
    top_n_airports: int = 30,
) -> tuple[pd.DataFrame, list[str]]:
    """Ejecuta el pipeline completo de preprocesamiento.

    Orden: limpieza → relleno de nulos → extracción de hora → filtro aeropuertos.

    Args:
        df: DataFrame crudo de vuelos.
        top_n_airports: Número de aeropuertos a mantener.

    Returns:
        Tupla de (DataFrame preprocesado, lista de aeropuertos).
    """
    logger.info("Iniciando pipeline de preprocesamiento...")

    df = clean_flights(df)
    df = fill_delay_nulls(df)
    df = encode_time(df)
    df, airports = filter_top_airports(df, top_n=top_n_airports)

    logger.info("Preprocesamiento completado: %d filas, %d aeropuertos",
                len(df), len(airports))

    return df, airports
