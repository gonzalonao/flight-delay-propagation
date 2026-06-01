"""Weather features for the per-flight model (point-in-time).

Reuses the same Open-Meteo hourly parquets the GNN consumes
(``data/processed/weather/open_meteo/{IATA}_*.parquet``) but attaches them at
**flight granularity**: for a flight predicted at ``t_pred = sched_dep − H`` we
add

* **origin** conditions *as-of* ``t_pred`` (observed "now" at the departure
  airport) — strictly point-in-time;
* **destination** conditions at the scheduled **arrival** hour, treated as a
  *forecast* (consistent with the GNN, which also uses forward weather). Weather
  forecasts are available ahead of time, so this is admissible; documented as a
  forecast assumption.

Columns per side: wind speed, wind gusts, precipitation, cloud cover, and the
categorical weather code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WEATHER_DIR = Path("data/processed/weather/open_meteo")
_SRC_COLS = ["wind_speed_10m", "wind_gusts_10m", "precipitation",
             "cloud_cover", "weather_category"]


def load_weather_long(airports: list[str]) -> pd.DataFrame:
    """Carga la meteorología horaria de los aeropuertos en formato largo.

    Returns:
        DataFrame con columnas ``airport``, ``hour`` (Timestamp horario) y los
        5 campos meteo. Aeropuertos sin fichero se omiten silenciosamente.
    """
    frames = []
    for ap in airports:
        files = sorted(WEATHER_DIR.glob(f"{ap}_*.parquet"))
        if not files:
            continue
        w = pd.read_parquet(files[0]).reset_index()
        ts_col = "timestamp" if "timestamp" in w.columns else w.columns[0]
        w = w.rename(columns={ts_col: "hour"})
        w["airport"] = ap
        frames.append(w[["airport", "hour"] + _SRC_COLS])
    long = pd.concat(frames, ignore_index=True)
    long["hour"] = pd.to_datetime(long["hour"]).dt.floor("h")
    return long


def _merge_side(work, weather, airport_col, hour_series, prefix):
    rename = {
        "wind_speed_10m": f"{prefix}_wind", "wind_gusts_10m": f"{prefix}_gust",
        "precipitation": f"{prefix}_precip", "cloud_cover": f"{prefix}_cloud",
        "weather_category": f"{prefix}_wxcat",
    }
    side = weather.rename(columns={"airport": airport_col, "hour": "_wx_hour"})
    key = pd.DataFrame({airport_col: work[airport_col].astype(str).to_numpy(),
                        "_wx_hour": hour_series.to_numpy()})
    merged = key.merge(side, on=[airport_col, "_wx_hour"], how="left")
    return {new: merged[old].to_numpy() for old, new in rename.items()}


def add_weather_asof(
    work: pd.DataFrame, weather: pd.DataFrame, horizon_h: int
) -> pd.DataFrame:
    """Añade columnas meteo point-in-time (origen as-of t_pred, destino al llegar).

    ``work`` debe traer ``dep_dt``, ``arr_dt``, ``Origin``, ``Dest``
    (de ``build_work_frame``).
    """
    t_pred = work["dep_dt"] - pd.Timedelta(hours=horizon_h)
    out = work.copy()
    for col, val in _merge_side(work, weather, "Origin", t_pred.dt.floor("h"), "orig").items():
        out[col] = val
    for col, val in _merge_side(work, weather, "Dest", work["arr_dt"].dt.floor("h"), "dest").items():
        out[col] = val
    for c in ("orig_wxcat", "dest_wxcat"):
        out[c] = out[c].astype("category")
    return out
