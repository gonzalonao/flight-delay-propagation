"""Point-in-time feature builder for the per-flight delay model.

Hard rule (see ``BRANCH_NOTES.md``): for a flight predicted at horizon ``H``,
the prediction time is ``t_pred = scheduled_departure − H hours`` and **only
information observable at or before ``t_pred`` may enter the feature vector.**

Three feature families, all as-of ``t_pred``:

* **Schedule / identity** — always known in advance (route, distance, scheduled
  block time, time-of-day, calendar, operating airline).
* **Aircraft rotation** — the *prior leg of the same tail* and how much of its
  delay is observable by ``t_pred``:
    - if the inbound aircraft has *landed* by ``t_pred`` → use its ArrDelay;
    - elif it has *departed* by ``t_pred`` → use its DepDelay (proxy);
    - else → unknown (NaN); the scheduled turnaround is still known.
* **Airport state** — realized delay/volume at the origin/dest airport over the
  last fully-completed hour(s) *before* ``t_pred`` (the "airport average delay at
  that point" fallback). A bucket ``[h, h+1)`` is used only when ``h+1 ≤ t_pred``,
  so it is fully observed and cannot leak.

The flight being predicted never contributes its own post-departure fields
(``DepDelay``/``ArrDelay``/``DepTime``/…) to ``X`` — those are the target or
post-``t_pred`` and live in ``BLOCKED_SELF_COLUMNS``. The leak guard
:func:`assert_no_leakage` enforces this by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")

# Columnas crudas necesarias. DepDelay/ArrDelay de OTROS vuelos se usan como
# eventos observados as-of; nunca se usan los del propio vuelo predicho.
RAW_COLUMNS = [
    "FlightDate", "Operating_Airline", "Origin", "Dest", "Tail_Number",
    "CRSDepTime", "CRSArrTime", "Distance", "CRSElapsedTime",
    "DayOfWeek", "Month", "DepDelay", "ArrDelay", "DepDel15",
    "Cancelled", "Diverted",
]

# --- Bloques de features (lo único que entra en X) --------------------------
F_SCHEDULE = ["Distance", "CRSElapsedTime", "dep_hour", "Month", "DayOfWeek"]
F_ROUTE = ["Origin", "Dest"]                      # categóricas
F_AIRLINE = ["Operating_Airline"]                 # categórica
F_ROTATION = [
    "inbound_delay_estimate", "sched_turnaround_min", "turnaround_slack",
    "inbound_status", "is_first_leg", "rotation_continuity",
]
F_AIRPORT = [
    "origin_dep_delay_1h", "origin_dep_delay_3h", "origin_dep_count_1h",
    "origin_dep_del15_rate_1h", "dest_arr_delay_1h", "dest_arr_delay_3h",
]
# Meteo point-in-time (las columnas las crea src.flight_level.weather.add_weather_asof).
F_WEATHER = [
    "orig_wind", "orig_gust", "orig_precip", "orig_cloud", "orig_wxcat",
    "dest_wind", "dest_gust", "dest_precip", "dest_cloud", "dest_wxcat",
]
CATEGORICAL_FEATURES = set(
    F_ROUTE + F_AIRLINE + ["inbound_status", "orig_wxcat", "dest_wxcat"]
)
TARGET = "ArrDelay"

# Columnas del PROPIO vuelo que jamás pueden ser features (target / post-salida).
BLOCKED_SELF_COLUMNS = {
    "ArrDelay", "DepDelay", "DepDel15", "ArrDel15", "DepTime", "ArrTime",
    "WheelsOff", "WheelsOn", "TaxiOut", "TaxiIn", "AirTime",
    "ActualElapsedTime", "DepDelayMinutes", "ArrDelayMinutes",
    "ArrivalDelayGroups", "DepartureDelayGroups",
}

FEATURE_SETS = {
    "A_schedule": F_SCHEDULE + F_ROUTE,
    "B_schedule_airline": F_SCHEDULE + F_ROUTE + F_AIRLINE,
    "C_schedule_rotation": F_SCHEDULE + F_ROUTE + F_ROTATION,
    "D_schedule_airport": F_SCHEDULE + F_ROUTE + F_AIRPORT,
    "E_all": F_SCHEDULE + F_ROUTE + F_AIRLINE + F_ROTATION + F_AIRPORT,
    "F_all_weather": F_SCHEDULE + F_ROUTE + F_AIRLINE + F_ROTATION + F_AIRPORT + F_WEATHER,
}


# ---------------------------------------------------------------------------
# Carga / limpieza
# ---------------------------------------------------------------------------
def load_flights(years: list[int]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(RAW_DIR / f"Combined_Flights_{y}.parquet",
                        columns=RAW_COLUMNS, engine="pyarrow")
        for y in years
    ]
    return pd.concat(frames, ignore_index=True)


def filter_top_airports(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    activity = (
        df["Origin"].value_counts()
        .add(df["Dest"].value_counts(), fill_value=0)
        .sort_values(ascending=False)
    )
    top = set(activity.head(top_n).index)
    return df[df["Origin"].isin(top) & df["Dest"].isin(top)].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["Cancelled"].fillna(False) & ~df["Diverted"].fillna(False)].copy()
    df = df.dropna(subset=["ArrDelay"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tiempos programados y reales (los reales solo se usan para construir eventos
# observables / observabilidad as-of, NUNCA como features del vuelo predicho).
# ---------------------------------------------------------------------------
def _hhmm_to_minutes(s: pd.Series) -> pd.Series:
    s = s.fillna(0).astype(int)
    return (s // 100).clip(0, 23) * 60 + (s % 100).clip(0, 59)


def add_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Añade dep_dt/arr_dt programados (rollover nocturno), dep_hour y los
    instantes reales de salida/llegada (sched + delay) para uso as-of."""
    dep_min = _hhmm_to_minutes(df["CRSDepTime"])
    arr_min = _hhmm_to_minutes(df["CRSArrTime"])
    df["dep_hour"] = (dep_min // 60).astype("int16")
    df["dep_dt"] = df["FlightDate"] + pd.to_timedelta(dep_min, unit="m")
    arr_dt = df["FlightDate"] + pd.to_timedelta(arr_min, unit="m")
    arr_dt += pd.to_timedelta((arr_min < dep_min).astype(int), unit="D")
    df["arr_dt"] = arr_dt
    # Instantes REALES (eventos observados). Usados solo para as-of, no features.
    df["actual_dep_dt"] = df["dep_dt"] + pd.to_timedelta(df["DepDelay"].fillna(0), unit="m")
    df["actual_arr_dt"] = df["arr_dt"] + pd.to_timedelta(df["ArrDelay"].fillna(0), unit="m")
    return df


# ---------------------------------------------------------------------------
# Familia 1: rotación (point-in-time)
# ---------------------------------------------------------------------------
def add_rotation_asof(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Features de rotación observables en ``t_pred = dep_dt − horizon_h``."""
    df = df.sort_values(["Tail_Number", "dep_dt"], kind="stable")
    g = df.groupby("Tail_Number", sort=False)

    prev_arr_delay = g["ArrDelay"].shift(1)
    prev_dep_delay = g["DepDelay"].shift(1)
    prev_arr_dt_sched = g["arr_dt"].shift(1)
    prev_actual_dep = g["actual_dep_dt"].shift(1)
    prev_actual_arr = g["actual_arr_dt"].shift(1)
    prev_dest = g["Dest"].shift(1)

    t_pred = df["dep_dt"] - pd.Timedelta(hours=horizon_h)

    # Continuidad física de la cadena (mismo avión enlaza destino->origen).
    turnaround = (df["dep_dt"] - prev_arr_dt_sched).dt.total_seconds() / 60.0
    continuity = (prev_dest.values == df["Origin"].values)
    bad_chain = (~continuity) | (turnaround > 12 * 60) | (turnaround < 0)
    is_first = prev_arr_delay.isna()

    # Observabilidad as-of: ¿ha aterrizado / despegado la etapa entrante?
    landed = prev_actual_arr <= t_pred
    departed = prev_actual_dep <= t_pred

    # Mejor estimación del retraso entrante disponible en t_pred.
    inbound_est = np.where(
        landed, prev_arr_delay,
        np.where(departed, prev_dep_delay, np.nan),
    )
    inbound_est = pd.Series(inbound_est, index=df.index).where(~bad_chain)

    # Estado categórico del avión entrante en t_pred.
    status = np.where(
        is_first | bad_chain, "no_inbound",
        np.where(landed, "arrived", np.where(departed, "departed", "scheduled")),
    )

    turnaround = pd.Series(turnaround, index=df.index).where(~bad_chain)
    df["inbound_delay_estimate"] = inbound_est.astype("float32")
    df["sched_turnaround_min"] = turnaround.astype("float32")
    df["turnaround_slack"] = (turnaround - inbound_est).astype("float32")
    df["inbound_status"] = pd.Series(status, index=df.index).astype("category")
    df["is_first_leg"] = is_first.astype("int8")
    df["rotation_continuity"] = pd.Series(continuity, index=df.index).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Familia 2: estado del aeropuerto (point-in-time, por hora completa pasada)
# ---------------------------------------------------------------------------
def build_airport_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos realizados por (aeropuerto, hora) a partir de eventos reales.

    Devuelve un frame con índice continuo (airport, hour) y medias móviles 1h/3h
    de salidas (por Origin) y llegadas (por Dest). Cada hora agrega vuelos cuyo
    instante REAL cae en ella; al usarse solo horas ya cerradas antes de
    ``t_pred`` no hay fuga.
    """
    dep = df[["Origin", "actual_dep_dt", "DepDelay", "DepDel15"]].copy()
    dep["hour"] = dep["actual_dep_dt"].dt.floor("h")
    dep_stats = dep.groupby(["Origin", "hour"]).agg(
        dep_delay=("DepDelay", "mean"),
        dep_count=("DepDelay", "size"),
        dep_del15=("DepDel15", "mean"),
    ).rename_axis(index={"Origin": "airport"})

    arr = df[["Dest", "actual_arr_dt", "ArrDelay"]].copy()
    arr["hour"] = arr["actual_arr_dt"].dt.floor("h")
    arr_stats = arr.groupby(["Dest", "hour"]).agg(
        arr_delay=("ArrDelay", "mean"),
        arr_count=("ArrDelay", "size"),
    ).rename_axis(index={"Dest": "airport"})

    stats = dep_stats.join(arr_stats, how="outer").sort_index()

    # Media móvil 3h por aeropuerto (sobre las horas presentes; min_periods=1).
    by_ap = stats.groupby(level="airport", sort=False)
    stats["dep_delay_3h"] = by_ap["dep_delay"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    stats["arr_delay_3h"] = by_ap["arr_delay"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    return stats.reset_index()


def add_airport_state_asof(
    df: pd.DataFrame, hourly: pd.DataFrame, horizon_h: int
) -> pd.DataFrame:
    """Une el estado del aeropuerto de la última hora cerrada antes de t_pred."""
    t_pred = df["dep_dt"] - pd.Timedelta(hours=horizon_h)
    # Última hora completamente observada: [h, h+1) con h+1 <= t_pred.
    lookup = t_pred.dt.floor("h") - pd.Timedelta(hours=1)
    df = df.copy()
    df["_lookup_hour"] = lookup

    o = hourly.rename(columns={"airport": "Origin", "hour": "_lookup_hour"})
    o = o[["Origin", "_lookup_hour", "dep_delay", "dep_count", "dep_del15", "dep_delay_3h"]]
    df = df.merge(o, on=["Origin", "_lookup_hour"], how="left")

    d = hourly.rename(columns={"airport": "Dest", "hour": "_lookup_hour"})
    d = d[["Dest", "_lookup_hour", "arr_delay", "arr_delay_3h"]]
    df = df.merge(d, on=["Dest", "_lookup_hour"], how="left")

    df = df.rename(columns={
        "dep_delay": "origin_dep_delay_1h",
        "dep_delay_3h": "origin_dep_delay_3h",
        "dep_count": "origin_dep_count_1h",
        "dep_del15": "origin_dep_del15_rate_1h",
        "arr_delay": "dest_arr_delay_1h",
        "arr_delay_3h": "dest_arr_delay_3h",
    })
    return df.drop(columns="_lookup_hour")


# ---------------------------------------------------------------------------
# Ensamblado + guardia anti-fuga
# ---------------------------------------------------------------------------
def assert_no_leakage(feature_columns: list[str]) -> None:
    """Falla si alguna feature es una columna post-salida del propio vuelo."""
    leaked = set(feature_columns) & BLOCKED_SELF_COLUMNS
    if leaked:
        raise ValueError(f"Fuga de datos: features prohibidas en X: {sorted(leaked)}")


def build_work_frame(
    df: pd.DataFrame, horizon_h: int, hourly: pd.DataFrame
) -> pd.DataFrame:
    """Frame de trabajo point-in-time para un horizonte: features + columnas de
    apoyo (``dep_dt``, ``arr_dt``, ``Dest``, target), todas alineadas por fila.

    Útil para la fusión GNN↔per-vuelo, que necesita unir por fila el pronóstico
    del GNN sin que el reordenado interno de la rotación desalinee los datos.
    """
    work = add_airport_state_asof(df, hourly, horizon_h)
    work = add_rotation_asof(work, horizon_h)
    for c in CATEGORICAL_FEATURES:
        if c in work:
            work[c] = work[c].astype("category")
    return work


def assemble(
    df: pd.DataFrame,
    horizon_h: int,
    hourly: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Construye (X, y) point-in-time para un horizonte.

    ``df`` debe venir de :func:`add_datetimes`. Aplica rotación + estado de
    aeropuerto as-of, valida ausencia de fuga y devuelve solo las columnas
    pedidas (categóricas con dtype ``category``).
    """
    assert_no_leakage(feature_columns)
    work = build_work_frame(df, horizon_h, hourly)
    return work[feature_columns], work[TARGET].astype("float32")
