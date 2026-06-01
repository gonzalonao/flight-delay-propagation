"""Tests del feature builder point-in-time (src.flight_level.features).

El foco es la **correcta observabilidad temporal**: en t_pred = salida - H solo
puede usarse información ya ocurrida. Se valida la rotación (avión entrante) a
distintos horizontes, el rollover nocturno, el estado de aeropuerto as-of (sin
fuga de horas futuras) y la guardia anti-fuga.
"""

import numpy as np
import pandas as pd
import pytest

from src.flight_level.features import (
    BLOCKED_SELF_COLUMNS,
    FEATURE_SETS,
    add_airport_state_asof,
    add_datetimes,
    add_rotation_asof,
    assemble,
    assert_no_leakage,
    build_airport_hourly_stats,
)


def _base_cols(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    for c in ["DepDel15"]:
        if c not in df:
            df[c] = 0.0
    return df


def _rotation_df():
    """Tail T1: leg1 A->B (sale 08:00 +30, llega 10:00 +40), leg2 B->C (sale 12:00).

    actual_dep leg1 = 08:30, actual_arr leg1 = 10:40, turnaround prog = 120 min.
    """
    rows = [
        dict(fid="leg1", Tail_Number="T1", Origin="A", Dest="B",
             FlightDate="2019-01-01", CRSDepTime=800, CRSArrTime=1000,
             DepDelay=30.0, ArrDelay=40.0, Distance=300.0, CRSElapsedTime=120.0,
             DayOfWeek=2, Month=1, Operating_Airline="ZZ"),
        dict(fid="leg2", Tail_Number="T1", Origin="B", Dest="C",
             FlightDate="2019-01-01", CRSDepTime=1200, CRSArrTime=1400,
             DepDelay=0.0, ArrDelay=50.0, Distance=300.0, CRSElapsedTime=120.0,
             DayOfWeek=2, Month=1, Operating_Airline="ZZ"),
    ]
    return add_datetimes(_base_cols(rows))


def _leg2(df: pd.DataFrame) -> pd.Series:
    return df[df["fid"] == "leg2"].iloc[0]


def test_overnight_rollover():
    rows = [dict(fid="o", Tail_Number="T9", Origin="A", Dest="B",
                 FlightDate="2019-01-01", CRSDepTime=2330, CRSArrTime=130,
                 DepDelay=0.0, ArrDelay=0.0, Distance=1.0, CRSElapsedTime=120.0,
                 DayOfWeek=2, Month=1, Operating_Airline="ZZ")]
    df = add_datetimes(_base_cols(rows))
    r = df.iloc[0]
    assert r["dep_dt"] == pd.Timestamp("2019-01-01 23:30")
    assert r["arr_dt"] == pd.Timestamp("2019-01-02 01:30")  # llegada al día siguiente


@pytest.mark.parametrize(
    "horizon,expected_status,expected_estimate",
    [
        (1, "arrived", 40.0),    # t_pred 11:00 -> entrante ya aterrizó -> ArrDelay
        (3, "departed", 30.0),   # t_pred 09:00 -> despegó, sin aterrizar -> DepDelay
        (4, "scheduled", np.nan),  # t_pred 08:00 -> aún no despega -> desconocido
    ],
)
def test_rotation_observability_by_horizon(horizon, expected_status, expected_estimate):
    df = add_rotation_asof(_rotation_df(), horizon)
    r = _leg2(df)
    assert r["inbound_status"] == expected_status
    assert r["sched_turnaround_min"] == 120.0          # programado: siempre conocido
    assert r["rotation_continuity"] == 1               # B->B enlaza
    if np.isnan(expected_estimate):
        assert pd.isna(r["inbound_delay_estimate"])
        assert pd.isna(r["turnaround_slack"])
    else:
        assert r["inbound_delay_estimate"] == expected_estimate
        assert r["turnaround_slack"] == 120.0 - expected_estimate


def test_first_leg_has_no_inbound():
    df = add_rotation_asof(_rotation_df(), 1)
    r = df[df["fid"] == "leg1"].iloc[0]
    assert r["is_first_leg"] == 1
    assert r["inbound_status"] == "no_inbound"
    assert pd.isna(r["inbound_delay_estimate"])


def test_airport_state_uses_only_past_completed_hour():
    """El estado de A en t_pred debe venir de la hora cerrada anterior, nunca
    de un evento posterior a t_pred (anti-fuga)."""
    rows = [
        # Evento pasado en bucket 10:00 (sale 10:30 +20): DEBE contar.
        dict(fid="past", Tail_Number="P1", Origin="A", Dest="Z",
             FlightDate="2019-01-01", CRSDepTime=1030, CRSArrTime=1200,
             DepDelay=20.0, ArrDelay=20.0, DepDel15=1.0, Distance=1.0,
             CRSElapsedTime=90.0, DayOfWeek=2, Month=1, Operating_Airline="ZZ"),
        # Evento futuro en bucket 11:00 (sale 11:30 +999): NO debe filtrarse.
        dict(fid="future", Tail_Number="P2", Origin="A", Dest="Z",
             FlightDate="2019-01-01", CRSDepTime=1130, CRSArrTime=1300,
             DepDelay=999.0, ArrDelay=999.0, DepDel15=1.0, Distance=1.0,
             CRSElapsedTime=90.0, DayOfWeek=2, Month=1, Operating_Airline="ZZ"),
        # Vuelo a predecir: sale de A a las 12:00.
        dict(fid="target", Tail_Number="P3", Origin="A", Dest="C",
             FlightDate="2019-01-01", CRSDepTime=1200, CRSArrTime=1330,
             DepDelay=0.0, ArrDelay=0.0, DepDel15=0.0, Distance=1.0,
             CRSElapsedTime=90.0, DayOfWeek=2, Month=1, Operating_Airline="ZZ"),
    ]
    df = add_datetimes(_base_cols(rows))
    hourly = build_airport_hourly_stats(df)
    # H=1 -> t_pred=11:00 -> lookup bucket = 10:00 (cerrado). El evento de 11:30
    # cae en bucket 11:00 [11:00,12:00) que termina en 12:00 > 11:00 -> excluido.
    out = add_airport_state_asof(df, hourly, horizon_h=1)
    tgt = out[out["fid"] == "target"].iloc[0]
    assert tgt["origin_dep_delay_1h"] == 20.0          # solo el evento pasado
    assert tgt["origin_dep_count_1h"] == 1


def test_leak_guard_rejects_blocked_columns():
    with pytest.raises(ValueError):
        assert_no_leakage(["Distance", "DepDelay"])     # DepDelay del propio vuelo


def test_assemble_outputs_clean_schema():
    df = _rotation_df()
    df["DepDel15"] = 0.0
    hourly = build_airport_hourly_stats(df)
    feats = FEATURE_SETS["E_all"]
    X, y = assemble(df, horizon_h=2, hourly=hourly, feature_columns=feats)
    assert list(X.columns) == feats
    assert not (set(X.columns) & BLOCKED_SELF_COLUMNS)  # ninguna columna prohibida
    assert y.name == "ArrDelay" and len(X) == len(df)
