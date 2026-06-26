"""Feature engineering for delay prediction models.

Generates temporal features, encodes categorical variables and prepares
the columns needed to train tabular models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features derived from FlightDate and Hour.

    Generates:
    - month, day_of_week (integers)
    - hour_sin, hour_cos (cyclic encoding of the hour)
    - dow_sin, dow_cos (cyclic encoding of the day of week)
    - is_weekend (boolean)

    Cyclic encoding avoids discontinuities (e.g., hour 23 → 0).

    Args:
        df: DataFrame with FlightDate and Hour columns.

    Returns:
        DataFrame with temporal features added.
    """
    # Use Month/DayOfWeek from the Parquet if present, otherwise extract from FlightDate
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
        # Cyclic encoding: transforms circular values to sin/cos
        df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info("Temporal features added")
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode categorical variables with LabelEncoder.

    If existing encoders are provided (from the training set), they are used
    to transform. Otherwise, new ones are created.

    Args:
        df: DataFrame with categorical columns.
        categorical_cols: List of columns to encode.
        encoders: Dictionary of pre-fitted encoders (optional).

    Returns:
        Tuple of (DataFrame with encoded columns, encoders dictionary).
    """
    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        col_encoded = f"{col}_encoded"

        if col in encoders:
            # Use existing encoder, handle unseen categories
            le = encoders[col]
            known = set(le.classes_)
            df[col_encoded] = df[col].apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )
        else:
            le = LabelEncoder()
            df[col_encoded] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        logger.info("  %s: %d categories encoded", col, len(encoders[col].classes_))

    return df, encoders


def scale_features(
    df: pd.DataFrame,
    numerical_cols: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize numerical columns with StandardScaler.

    Fits the scaler only on training data (scaler=None). For
    validation/test, pass the already-fitted scaler.

    Args:
        df: DataFrame with numerical columns.
        numerical_cols: List of columns to normalize.
        scaler: Pre-fitted StandardScaler (optional).

    Returns:
        Tuple of (DataFrame with normalized columns, fitted scaler).
    """
    # Filter to columns that exist in the DataFrame
    cols_present = [c for c in numerical_cols if c in df.columns]

    if not cols_present:
        logger.warning("No numerical column found to normalize")
        return df, scaler or StandardScaler()

    if scaler is None:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
        logger.info("Scaler fitted and applied to %d columns", len(cols_present))
    else:
        df[cols_present] = scaler.transform(df[cols_present])
        logger.info("Scaler applied to %d columns", len(cols_present))

    return df, scaler


def build_feature_matrix(
    df: pd.DataFrame,
    config: dict,
    encoders: dict[str, LabelEncoder] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], StandardScaler]:
    """Build the full feature matrix for tabular models.

    Pipeline: temporal features → categorical encoding → normalization.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary with a 'features' section.
        encoders: Pre-fitted encoders (for val/test).
        scaler: Pre-fitted scaler (for val/test).

    Returns:
        Tuple of (DataFrame with features, encoders, scaler).
    """
    features_config = config.get("features", {})

    # 1. Temporal features
    df = add_temporal_features(df)

    # 2. Categorical encoding
    categorical_cols = features_config.get("categorical", [])
    if categorical_cols:
        df, encoders = encode_categoricals(df, categorical_cols, encoders)

    # 3. Normalization
    numerical_cols = features_config.get("numerical", [])
    # Add temporal features to the normalization
    numerical_cols = numerical_cols + [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
    ]
    df, scaler = scale_features(df, numerical_cols, scaler)

    return df, encoders, scaler
