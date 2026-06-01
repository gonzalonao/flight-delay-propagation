"""Inferencia reutilizable para modelos de secuencia (seq2seq_gnn).

Factoriza el forward-pass + ensamblado del DataFrame de predicciones para
que tanto el notebook de Fabric (``fabric/notebooks/nb_inference.ipynb``)
como los scripts locales (``scripts/predict.py``, ``scripts/export_powerbi.py``)
compartan una única ruta de código y produzcan el mismo esquema "long".
"""

from src.inference.predictor import (
    PREDICTION_COLUMNS,
    invert_airport_map,
    predict_sequence,
    predict_to_frame,
)

__all__ = [
    "PREDICTION_COLUMNS",
    "invert_airport_map",
    "predict_sequence",
    "predict_to_frame",
]
