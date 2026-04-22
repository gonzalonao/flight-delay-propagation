"""Clases Dataset de PyTorch para datos de vuelos.

Proporciona datasets tabulares para modelos Dense/LSTM y se extenderá
en fases posteriores con datasets de grafos para los modelos GNN.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FlightDelayDataset(Dataset):
    """Dataset tabular para predicción de retraso de vuelos.

    Utilizado por los modelos DenseNN y como base para modelos más complejos.
    Almacena features y targets como tensores de PyTorch.

    Args:
        features: Array NumPy o DataFrame con las features de entrada.
        targets: Array NumPy o Series con los valores objetivo (ArrDelay).
    """

    def __init__(
        self,
        features: np.ndarray | pd.DataFrame,
        targets: np.ndarray | pd.Series,
    ) -> None:
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.Series):
            targets = targets.values

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

        logger.info(
            "Dataset creado: %d muestras, %d features",
            len(self.features), self.features.shape[1],
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def get_feature_columns(df: pd.DataFrame, target_col: str = "ArrDelay") -> list[str]:
    """Identifica las columnas de features válidas para entrenamiento.

    Excluye la columna objetivo, columnas de identificación y columnas
    categóricas originales (solo usa las versiones _encoded).

    Args:
        df: DataFrame con todas las columnas.
        target_col: Nombre de la columna objetivo.

    Returns:
        Lista de nombres de columnas de features.
    """
    # Columnas a excluir del entrenamiento
    exclude = {
        target_col, "FlightDate", "Airline", "Origin", "Dest",
        "CRSDepTime", "Cancelled", "Diverted", "Hour", "day_of_week",
        "Month", "DayOfWeek", "DayofMonth",  # Raw Parquet columns (usamos versiones procesadas)
    }

    feature_cols = [
        col for col in df.columns
        if col not in exclude and df[col].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]

    logger.info("Columnas de features seleccionadas: %d", len(feature_cols))
    return feature_cols


def create_splits(
    df: pd.DataFrame,
    config: dict,
    target_col: str = "ArrDelay",
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Divide los datos en train/val/test usando separación temporal.

    Soporta dos modos:
    - **Por fecha** (preferido cuando hay varios años): claves
      ``split.train_end`` y ``split.val_end`` como ISO ``YYYY-MM-DD``. El
      corte se hace sobre ``FlightDate``: ``< train_end`` → train;
      ``[train_end, val_end)`` → val; ``≥ val_end`` → test.
    - **Por mes** (legacy): ``split.train_months`` /
      ``split.val_months`` / ``split.test_months``. Solo válido cuando el
      dataset cubre un único año — con dos años, jan-2018 y jan-2019 caen
      en el mismo bucket.

    Args:
        df: DataFrame preprocesado con columna FlightDate.
        config: Configuración con sección 'split'.
        target_col: Nombre de la columna objetivo.

    Returns:
        Diccionario con claves 'train', 'val', 'test', cada una
        con tupla (features_df, targets_series).
    """
    split_config = config.get("split", {})
    train_end = split_config.get("train_end")
    val_end = split_config.get("val_end")

    feature_cols = get_feature_columns(df, target_col)

    if train_end is not None and val_end is not None:
        if "FlightDate" not in df.columns:
            raise ValueError(
                "Split por fecha requiere la columna FlightDate sin "
                "transformar; añádela a `data.columns` del config."
            )
        date_col = pd.to_datetime(df["FlightDate"])
        train_cutoff = pd.Timestamp(train_end)
        val_cutoff = pd.Timestamp(val_end)

        masks = {
            "train": date_col < train_cutoff,
            "val": (date_col >= train_cutoff) & (date_col < val_cutoff),
            "test": date_col >= val_cutoff,
        }
        labels = {
            "train": f"< {train_end}",
            "val": f"{train_end} ≤ d < {val_end}",
            "test": f">= {val_end}",
        }
    else:
        train_months = split_config.get("train_months", list(range(1, 10)))
        val_months = split_config.get("val_months", [10, 11])
        test_months = split_config.get("test_months", [12])

        if "Month" in df.columns:
            month_col = df["Month"]
        elif "FlightDate" in df.columns:
            month_col = df["FlightDate"].dt.month
        elif "month" in df.columns:
            # Solo válido si no está escalada (valores enteros 1-12)
            month_col = df["month"]
        else:
            raise ValueError("No se encontró columna de mes ni FlightDate")

        masks = {
            "train": month_col.isin(train_months),
            "val": month_col.isin(val_months),
            "test": month_col.isin(test_months),
        }
        labels = {
            "train": f"meses {train_months}",
            "val": f"meses {val_months}",
            "test": f"meses {test_months}",
        }

    splits = {}
    for name, mask in masks.items():
        subset = df[mask]
        splits[name] = (subset[feature_cols], subset[target_col])
        logger.info("Split %s: %d muestras (%s)", name, len(subset), labels[name])

    return splits
