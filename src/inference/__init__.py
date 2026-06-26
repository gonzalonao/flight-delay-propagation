"""Reusable inference for sequence models (seq2seq_gnn).

Factors out the forward pass + prediction-DataFrame assembly so that both
the Fabric notebook (``fabric/notebooks/nb_inference.ipynb``) and the local
scripts (``scripts/predict.py``, ``scripts/export_powerbi.py``) share a
single code path and produce the same "long" schema.
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
