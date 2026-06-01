"""Tests de las agregaciones del export de Power BI (scripts.export_powerbi)."""

import numpy as np
import pandas as pd

from scripts.export_powerbi import (
    build_agg_baseline_volume,
    build_dim_date,
    build_dim_hour,
    build_fact_airport_hour,
    build_flight_frame,
    enrich_predictions,
)


def _raw_flights():
    return pd.DataFrame(
        {
            "FlightDate": ["2019-10-01"] * 5,
            "Airline": ["X"] * 5,
            "Origin": ["A", "A", "B", "A", "A"],
            "Dest": ["B", "B", "A", "B", "B"],
            "CRSDepTime": [800, 830, 900, 2330, 800],
            "CRSArrTime": [1000, 1030, 1100, 100, 1000],
            "Distance": [500.0] * 5,
            "DepDelay": [10.0, 0.0, 5.0, 0.0, np.nan],
            "ArrDelay": [20.0, 5.0, 30.0, 0.0, np.nan],
            "DepDel15": [0.0, 0.0, 0.0, 0.0, 0.0],
            "ArrDel15": [1.0, 0.0, 1.0, 0.0, 0.0],
            "Cancelled": [False, False, False, False, True],
            "Diverted": [False] * 5,
        }
    )


def test_build_flight_frame_overnight_and_network_filter():
    fr = build_flight_frame(_raw_flights(), ["A", "B"])
    assert len(fr) == 5  # todos dentro de la red A/B
    overnight = fr[fr["CRSArrTime"] == 100].iloc[0]
    # Llegada 01:00 con salida 23:30 -> la llegada cae al día siguiente.
    assert overnight["arr_ts_hour"] == pd.Timestamp("2019-10-02 01:00:00")
    assert overnight["dep_ts_hour"] == pd.Timestamp("2019-10-01 23:00:00")


def test_fact_airport_hour_counts_and_cancellations():
    fr = build_flight_frame(_raw_flights(), ["A", "B"])
    fact = build_fact_airport_hour(fr, seats_per_flight=100)

    # Llegadas a B el 2019-10-01 10:00 -> 3 programadas (2 operadas + 1 cancelada).
    b10 = fact[
        (fact.airport_code == "B")
        & (fact.ts_hour == pd.Timestamp("2019-10-01 10:00:00"))
    ].iloc[0]
    assert b10["sched_arr"] == 3
    assert b10["arr_operated"] == 2
    assert b10["arr_cancelled"] == 1
    assert abs(b10["arr_delay_mean"] - 12.5) < 1e-6   # (20+5)/2, NaN excluido
    assert b10["arr_del15"] == 1
    assert b10["est_arr_passengers"] == 300           # 3 * 100

    # Salidas de A a las 08:00 -> 3 programadas (800, 830, 800-cancelada).
    a08 = fact[
        (fact.airport_code == "A")
        & (fact.ts_hour == pd.Timestamp("2019-10-01 08:00:00"))
    ].iloc[0]
    assert a08["sched_dep"] == 3
    assert a08["dep_operated"] == 2
    assert a08["dep_cancelled"] == 1


def test_dim_date_and_hour():
    dim_date = build_dim_date(pd.Timestamp("2019-10-01"), pd.Timestamp("2019-10-03"))
    assert len(dim_date) == 3
    assert {"date", "year", "season", "is_weekend", "day_name"}.issubset(dim_date.columns)

    dim_hour = build_dim_hour()
    assert len(dim_hour) == 24
    assert dim_hour.loc[dim_hour.hour == 0, "part_of_day"].iloc[0] == "Night"
    assert dim_hour.loc[dim_hour.hour == 14, "part_of_day"].iloc[0] == "Afternoon"


def test_enrich_predictions_attaches_volume_and_expected():
    fr = build_flight_frame(_raw_flights(), ["A", "B"])
    fact = build_fact_airport_hour(fr, seats_per_flight=None)
    baseline = build_agg_baseline_volume(fact)
    preds = pd.DataFrame(
        {
            "airport_code": ["B"],
            "horizon_h": [1],
            "predicted_arr_delay_min": [12.0],
            "pct_arr_delayed_15": [0.5],
            "target_ts": ["2019-10-01T10:00:00"],
            "prediction_ts": ["2019-10-01T09:00:00"],
        }
    )
    out = enrich_predictions(preds, fact, baseline)
    assert out.iloc[0]["sched_arr"] == 3                      # B@10:00 -> 3 programadas
    assert out.iloc[0]["expected_delayed_flights"] == 1.5     # 0.5 * 3
