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


_HISTGBM_DEFAULTS = dict(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
    min_samples_leaf=200, l2_regularization=1.0,
)
_LIGHTGBM_DEFAULTS = dict(
    n_estimators=2000, learning_rate=0.03, num_leaves=255,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
)


class FlightDelayModel:
    """Regresor de retraso per-vuelo con backend conmutable (HistGBM / LightGBM).

    ``backend='lightgbm'`` admite early stopping vía ``fit(..., eval_set=...)``.
    HistGBM es el por defecto (compatibilidad hacia atrás).
    """

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        *,
        backend: str = "histgbm",
        random_state: int = 42,
        **params,
    ) -> None:
        self.categorical_features = categorical_features or []
        self.backend = backend
        self.feature_names_: list[str] | None = None
        if backend == "lightgbm":
            from lightgbm import LGBMRegressor
            cfg = {**_LIGHTGBM_DEFAULTS, **params, "random_state": random_state}
            self.regressor = LGBMRegressor(verbose=-1, **cfg)
        elif backend == "histgbm":
            cfg = {**_HISTGBM_DEFAULTS, **params}
            self.regressor = HistGradientBoostingRegressor(
                categorical_features=self.categorical_features or None,
                random_state=random_state, **cfg,
            )
        else:
            raise ValueError(f"backend desconocido: {backend!r}")

    def fit(self, X, y, eval_set=None) -> "FlightDelayModel":
        self.feature_names_ = list(X.columns)
        if self.backend == "lightgbm":
            import lightgbm as lgb
            cat = [c for c in self.categorical_features if c in X.columns]
            callbacks = []
            if eval_set is not None:
                callbacks = [lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)]
            self.regressor.fit(
                X, y, eval_set=eval_set,
                categorical_feature=cat or "auto", callbacks=callbacks or None,
            )
        else:
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
             "feature_names": self.feature_names_,
             "backend": self.backend},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlightDelayModel":
        blob = joblib.load(path)
        obj = cls(categorical_features=blob["categorical_features"],
                  backend=blob.get("backend", "histgbm"))
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
