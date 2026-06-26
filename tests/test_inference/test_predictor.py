"""Tests for the predictions assembler (src.inference.predictor)."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.inference.predictor import (
    ACTUAL_COLUMNS,
    PREDICTION_COLUMNS,
    invert_airport_map,
    predict_to_frame,
)


class _FakeSeqModel:
    """Test model: ignores the input and returns a fixed output [N,H,3]."""

    def __init__(self, out):
        self._out = out

    def eval(self):
        return self

    def to(self, device):  # noqa: D401 - parity with nn.Module
        return self

    def __call__(self, sequence):
        return self._out


def _make_graph(timestamp):
    x = torch.zeros(3, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # y: [N=3, H=2, C=3]; channel 0 = actual arr delay, channel 2 = pct target.
    y = torch.zeros(3, 2, 3)
    y[:, :, 0] = torch.tensor([[20.0, 5.0], [30.0, 8.0], [0.0, 0.0]])
    y[:, :, 2] = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    active_mask = torch.tensor([True, True, False])
    return Data(
        x=x, edge_index=edge_index, y=y,
        active_mask=active_mask, timestamp=timestamp,
    )


def _fake_out():
    out = torch.zeros(3, 2, 3)
    out[:, :, 0] = torch.tensor([[10.0, 20.0], [5.0, 40.0], [99.0, 99.0]])  # arr delay
    out[:, :, 2] = torch.tensor([[0.0, 2.0], [-2.0, 0.0], [0.0, 0.0]])      # logits
    return out


def test_predict_to_frame_schema_and_values():
    model = _FakeSeqModel(_fake_out())
    seqs = [
        [_make_graph("2019-10-01T05:00:00")],
        [_make_graph("2019-10-01T06:00:00")],
    ]
    idx_to_iata = {0: "AAA", 1: "BBB", 2: "CCC"}
    df = predict_to_frame(model, seqs, [1, 4], idx_to_iata, torch.device("cpu"))

    # 2 active nodes x 2 horizons x 2 sequences = 8 rows; CCC excluded.
    assert len(df) == 8
    assert list(df.columns) == PREDICTION_COLUMNS + ACTUAL_COLUMNS
    assert set(df["airport_code"]) == {"AAA", "BBB"}
    assert set(df["horizon_h"]) == {1, 4}

    row = df[
        (df.airport_code == "AAA")
        & (df.horizon_h == 1)
        & (df.prediction_ts == "2019-10-01T05:00:00")
    ].iloc[0]
    assert row["predicted_arr_delay_min"] == 10.0
    assert abs(row["pct_arr_delayed_15"] - 0.5) < 1e-6   # sigmoid(0)
    assert row["target_ts"] == "2019-10-01T06:00:00"     # +1h
    assert row["actual_arr_delay_min"] == 20.0
    assert row["actual_arr_del15"] == 1

    row4 = df[
        (df.airport_code == "AAA")
        & (df.horizon_h == 4)
        & (df.prediction_ts == "2019-10-01T05:00:00")
    ].iloc[0]
    assert abs(row4["pct_arr_delayed_15"] - (1 / (1 + np.exp(-2)))) < 1e-6
    assert row4["target_ts"] == "2019-10-01T09:00:00"    # +4h


def test_predict_to_frame_reference_ts_override():
    model = _FakeSeqModel(_fake_out())  # output with 2 horizons
    g = _make_graph("2019-10-01T05:00:00")
    df = predict_to_frame(
        model, [[g]], [1, 4], {0: "AAA", 1: "BBB", 2: "CCC"},
        torch.device("cpu"), include_actuals=False,
        reference_ts=pd.Timestamp("2020-01-01T00:00:00"),
    )
    # reference_ts replaces the graph's timestamp in all rows.
    assert (df["prediction_ts"] == "2020-01-01T00:00:00").all()
    assert set(df["target_ts"]) == {"2020-01-01T01:00:00", "2020-01-01T04:00:00"}
    assert list(df.columns) == PREDICTION_COLUMNS  # without actuals columns


def test_invert_airport_map():
    assert invert_airport_map({"ATL": 0, "LAX": 1}) == {0: "ATL", 1: "LAX"}


def test_empty_sequences_return_schema():
    model = _FakeSeqModel(_fake_out())
    df = predict_to_frame(model, [], [1, 4], {0: "AAA"}, torch.device("cpu"))
    assert df.empty
    assert list(df.columns) == PREDICTION_COLUMNS + ACTUAL_COLUMNS
