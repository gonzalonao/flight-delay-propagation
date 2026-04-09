"""Ingeniería de características para modelos de predicción de retrasos.

Genera features temporales, codifica variables categóricas y prepara
las columnas necesarias para el entrenamiento de modelos tabulares.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features temporales derivadas de FlightDate y Hour.

    Genera:
    - month, day_of_week (enteros)
    - hour_sin, hour_cos (codificación cíclica de la hora)
    - dow_sin, dow_cos (codificación cíclica del día de la semana)
    - is_weekend (booleano)

    La codificación cíclica evita discontinuidades (e.g., hora 23 → 0).

    Args:
        df: DataFrame con columnas FlightDate y Hour.

    Returns:
        DataFrame con features temporales añadidas.
    """
    # Usar Month/DayOfWeek del Parquet si existen, sino extraer de FlightDate
    if "Month" in df.columns and "month" not in df.columns:
        df["month"] = df["Month"]
    elif "FlightDate" in df.columns and "month" not in df.columns:
        df["month"] = df["FlightDate"].dt.month

    if "DayOfWeek" in df.columns and "day_of_week" not in df.columns:
        df["day_of_week"] = df["DayOfWeek"]
    elif "FlightDate" in df.columns and "day_of_week" not in df.columns:
        df["day_of_week"] = df["FlightDate"].dt.dayofweek

    if "day_of_week" in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "Hour" in df.columns:
        # Codificación cíclica: transforma valores circulares a sin/cos
        df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info("Features temporales añadidas")
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Codifica variables categóricas con LabelEncoder.

    Si se proporcionan encoders existentes (del set de entrenamiento),
    los usa para transformar. Si no, crea nuevos.

    Args:
        df: DataFrame con columnas categóricas.
        categorical_cols: Lista de columnas a codificar.
        encoders: Diccionario de encoders pre-ajustados (opcional).

    Returns:
        Tupla de (DataFrame con columnas codificadas, diccionario de encoders).
    """
    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        col_encoded = f"{col}_encoded"

        if col in encoders:
            # Usar encoder existente, manejar categorías no vistas
            le = encoders[col]
            known = set(le.classes_)
            df[col_encoded] = df[col].apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
        else:
            le = LabelEncoder()
            df[col_encoded] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        logger.info("  %s: %d categorías codificadas", col, len(encoders[col].classes_))

    return df, encoders


def scale_features(
    df: pd.DataFrame,
    numerical_cols: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Normaliza columnas numéricas con StandardScaler.

    Ajusta el scaler solo en datos de entrenamiento (scaler=None).
    Para validación/test, pasar el scaler ya ajustado.

    Args:
        df: DataFrame con columnas numéricas.
        numerical_cols: Lista de columnas a normalizar.
        scaler: StandardScaler pre-ajustado (opcional).

    Returns:
        Tupla de (DataFrame con columnas normalizadas, scaler ajustado).
    """
    # Filtrar a columnas que existen en el DataFrame
    cols_present = [c for c in numerical_cols if c in df.columns]

    if not cols_present:
        logger.warning("Ninguna columna numérica encontrada para normalizar")
        return df, scaler or StandardScaler()

    if scaler is None:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
        logger.info("Scaler ajustado y aplicado a %d columnas", len(cols_present))
    else:
        df[cols_present] = scaler.transform(df[cols_present])
        logger.info("Scaler aplicado a %d columnas", len(cols_present))

    return df, scaler


def build_feature_matrix(
    df: pd.DataFrame,
    config: dict,
    encoders: dict[str, LabelEncoder] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], StandardScaler]:
    """Construye la matriz de features completa para modelos tabulares.

    Pipeline: features temporales → codificación categórica → normalización.

    Args:
        df: DataFrame preprocesado.
        config: Diccionario de configuración con sección 'features'.
        encoders: Encoders pre-ajustados (para val/test).
        scaler: Scaler pre-ajustado (para val/test).

    Returns:
        Tupla de (DataFrame con features, encoders, scaler).
    """
    features_config = config.get("features", {})

    # 1. Features temporales
    df = add_temporal_features(df)

    # 2. Codificación categórica
    categorical_cols = features_config.get("categorical", [])
    if categorical_cols:
        df, encoders = encode_categoricals(df, categorical_cols, encoders)

    # 3. Normalización
    numerical_cols = features_config.get("numerical", [])
    # Añadir features temporales a la normalización
    numerical_cols = numerical_cols + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
    ]
    df, scaler = scale_features(df, numerical_cols, scaler)

    return df, encoders, scaler
