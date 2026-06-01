"""Per-flight delay model (gradient-boosted regression + multi-threshold readout).

Regresses ``ArrDelay`` in minutes with a histogram GBM (native categorical +
NaN support, zero extra dependency). Classification metrics at multiple delay
thresholds are derived from the regression at inference, per the agreed
"regress minutes, threshold later" approach — so the cutoff (15/30/45/60 min)
is a reporting choice, not a retrain.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)

DEFAULT_THRESHOLDS = [15, 30, 45, 60]


class FlightDelayModel:
    """Envuelve un ``HistGradientBoostingRegressor`` con I/O y métricas."""

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        *,
        max_iter: int = 400,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 200,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.categorical_features = categorical_features or []
        self.feature_names_: list[str] | None = None
        self.regressor = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            categorical_features=self.categorical_features or None,
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FlightDelayModel":
        self.feature_names_ = list(X.columns)
        self.regressor.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.regressor.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"regressor": self.regressor,
             "categorical_features": self.categorical_features,
             "feature_names": self.feature_names_},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlightDelayModel":
        blob = joblib.load(path)
        obj = cls(categorical_features=blob["categorical_features"])
        obj.regressor = blob["regressor"]
        obj.feature_names_ = blob["feature_names"]
        return obj

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thresholds: list[int] = DEFAULT_THRESHOLDS,
    ) -> dict:
        """MAE/RMSE + recall/precision/F1 derivados a cada umbral de retraso."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        by_thr = {}
        for thr in thresholds:
            t, p = y_true >= thr, y_pred >= thr
            rec = recall_score(t, p, zero_division=0)
            prec = precision_score(t, p, zero_division=0)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            by_thr[str(thr)] = {
                "positive_rate": float(t.mean()),
                "recall": float(rec),
                "precision": float(prec),
                "f1": float(f1),
            }
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "n": int(len(y_true)),
            "by_threshold": by_thr,
        }
