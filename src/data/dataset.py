"""PyTorch Dataset classes for flight data.

Provides tabular datasets for the Dense/LSTM models and is extended in
later phases with graph datasets for the GNN models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FlightDelayDataset(Dataset):
    """Tabular dataset for flight-delay prediction.

    Used by the DenseNN models and as a base for more complex models.
    Stores features and targets as PyTorch tensors.

    Args:
        features: NumPy array or DataFrame with the input features.
        targets: NumPy array or Series with the target values (ArrDelay).
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
            "Dataset created: %d samples, %d features",
            len(self.features),
            self.features.shape[1],
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def get_feature_columns(df: pd.DataFrame, target_col: str = "ArrDelay") -> list[str]:
    """Identify the valid feature columns for training.

    Excludes the target column, identifier columns and the original
    categorical columns (only uses the _encoded versions).

    Args:
        df: DataFrame with all columns.
        target_col: Name of the target column.

    Returns:
        List of feature column names.
    """
    # Columns to exclude from training
    exclude = {
        target_col,
        "FlightDate",
        "Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "Cancelled",
        "Diverted",
        "Hour",
        "day_of_week",
        "Month",
        "DayOfWeek",
        "DayofMonth",  # raw Parquet cols (use processed)
    }

    feature_cols = [
        col
        for col in df.columns
        if col not in exclude
        and df[col].dtype in [np.float32, np.float64, np.int64, np.int32]
    ]

    logger.info("Selected feature columns: %d", len(feature_cols))
    return feature_cols


def create_splits(
    df: pd.DataFrame,
    config: dict,
    target_col: str = "ArrDelay",
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Split the data into train/val/test using a temporal split.

    Supports two modes:
    - **By date** (preferred when spanning several years): keys
      ``split.train_end`` and ``split.val_end`` as ISO ``YYYY-MM-DD``. The
      cut is on ``FlightDate``: ``< train_end`` → train;
      ``[train_end, val_end)`` → val; ``≥ val_end`` → test.
    - **By month** (legacy): ``split.train_months`` /
      ``split.val_months`` / ``split.test_months``. Only valid when the
      dataset covers a single year — with two years, jan-2018 and jan-2019
      fall into the same bucket.

    Args:
        df: Preprocessed DataFrame with a FlightDate column.
        config: Configuration with a 'split' section.
        target_col: Name of the target column.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each with a tuple
        (features_df, targets_series).
    """
    split_config = config.get("split", {})
    train_end = split_config.get("train_end")
    val_end = split_config.get("val_end")

    feature_cols = get_feature_columns(df, target_col)

    if train_end is not None and val_end is not None:
        if "FlightDate" not in df.columns:
            raise ValueError(
                "Date-based split requires the untransformed FlightDate "
                "column; add it to `data.columns` in the config."
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
            # Only valid if not scaled (integer values 1-12)
            month_col = df["month"]
        else:
            raise ValueError("No month column or FlightDate found")

        masks = {
            "train": month_col.isin(train_months),
            "val": month_col.isin(val_months),
            "test": month_col.isin(test_months),
        }
        labels = {
            "train": f"months {train_months}",
            "val": f"months {val_months}",
            "test": f"months {test_months}",
        }

    splits = {}
    for name, mask in masks.items():
        subset = df[mask]
        splits[name] = (subset[feature_cols], subset[target_col])
        logger.info("Split %s: %d samples (%s)", name, len(subset), labels[name])

    return splits
