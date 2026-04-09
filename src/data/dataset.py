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

    La separación temporal previene fuga de datos: se entrena con meses
    anteriores y se evalúa con meses posteriores.

    Args:
        df: DataFrame preprocesado con columna FlightDate.
        config: Configuración con sección 'split'.
        target_col: Nombre de la columna objetivo.

    Returns:
        Diccionario con claves 'train', 'val', 'test', cada una
        con tupla (features_df, targets_series).
    """
    split_config = config.get("split", {})
    train_months = split_config.get("train_months", list(range(1, 10)))
    val_months = split_config.get("val_months", [10, 11])
    test_months = split_config.get("test_months", [12])

    month_col = df["FlightDate"].dt.month if "FlightDate" in df.columns else df["month"]

    feature_cols = get_feature_columns(df, target_col)

    splits = {}
    for name, months in [("train", train_months), ("val", val_months), ("test", test_months)]:
        mask = month_col.isin(months)
        subset = df[mask]
        splits[name] = (subset[feature_cols], subset[target_col])
        logger.info("Split %s: %d muestras (meses %s)", name, len(subset), months)

    return splits
