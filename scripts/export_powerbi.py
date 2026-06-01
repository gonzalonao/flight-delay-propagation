"""Exporta tablas planas (esquema en estrella) para el informe de Power BI.

Combina los **datos base 2018-2019** (los mismos usados en train/test) con las
**predicciones derivadas** del campeón ``seq2seq_gnn``, y escribe un conjunto de
tablas Parquet + CSV bajo ``outputs/powerbi/`` listas para importar:

    dim_airport            - metadatos de los 70 aeropuertos modelados
    dim_date               - tabla calendario (slicers)
    dim_hour               - hora del día / franja
    fact_airport_hour      - agregado por (aeropuerto, hora): vuelos, retrasos
    agg_baseline_volume    - volumen "habitual" por (aeropuerto, dow, hora)
    agg_airline            - retraso medio por aerolínea x mes
    agg_route              - retraso/volumen por ruta (origen-destino)
    agg_distance_bucket    - retraso medio por tramo de distancia
    fact_predictions       - predicciones (long) + verdad-terreno + enriquecido

Las tablas estáticas de referencia (``dim_airport``, ``agg_baseline_volume``)
también sirven para subirlas una vez a OneLake y alimentar la página live
(DirectLake). Ver ``docs/powerbi-report-spec.md``.

Uso:
    python scripts/export_powerbi.py \
        --checkpoint outputs/runs/20260514-213242/seq2seq_gnn_large.pt \
        --config configs/weekend/seq2seq_gnn_large.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_multiple_years
from src.data.preprocessing import preprocess_pipeline
from src.utils.config import load_config
from src.utils.io import get_data_dir, get_project_root, get_output_dir
from src.utils.logger import setup_logger
from src.utils.reproducibility import set_seed

# Import perezoso de predicciones: el módulo importa torch/PyG, que no hace
# falta cuando se ejecuta sólo el export base (--no-predictions).

logger = setup_logger(__name__)

DEFAULT_CONFIG = "configs/weekend/seq2seq_gnn_large.yaml"
DEFAULT_SEATS_PER_FLIGHT = 130  # estimación media de asientos/vuelo (US domestic)
AIRPORT_COORDS = "src/data/reference/airport_coords.csv"

# Columnas mínimas necesarias para las agregaciones base.
_BASE_COLS = [
    "FlightDate", "Airline", "Origin", "Dest",
    "CRSDepTime", "CRSArrTime", "Distance",
    "DepDelay", "ArrDelay", "DepDel15", "ArrDel15",
    "Cancelled", "Diverted",
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- #
# Construcción del frame de vuelos (mantiene cancelados, red top-70)
# --------------------------------------------------------------------------- #
def build_flight_frame(raw_df: pd.DataFrame, airports: list[str]) -> pd.DataFrame:
    """Frame de vuelos con timestamps de salida/llegada por hora.

    A diferencia de ``preprocess_pipeline`` (que descarta cancelados/desviados),
    aquí los conservamos para poder reportar tasas de cancelación. Restringe a
    vuelos cuyo origen Y destino están en la red de 70 aeropuertos modelada.
    """
    cols = [c for c in _BASE_COLS if c in raw_df.columns]
    df = raw_df[cols].copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # Red modelada: ambos extremos en top-70.
    df = df[df["Origin"].isin(airports) & df["Dest"].isin(airports)].copy()

    for col in ("Cancelled", "Diverted"):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
        else:
            df[col] = False

    dep = pd.to_numeric(df["CRSDepTime"], errors="coerce")
    arr = pd.to_numeric(df["CRSArrTime"], errors="coerce")
    df = df[dep.notna() & arr.notna()].copy()
    dep = dep.loc[df.index]
    arr = arr.loc[df.index]

    dep_hour = (dep // 100).clip(0, 23).astype(int)
    arr_hour = (arr // 100).clip(0, 23).astype(int)
    dep_date = df["FlightDate"].dt.normalize()
    # Rollover nocturno: si la hora programada de llegada es anterior a la de
    # salida, la llegada cae al día siguiente.
    overnight = (arr.values < dep.values)
    arr_date = dep_date + pd.to_timedelta(overnight.astype(int), unit="D")

    df["dep_ts_hour"] = dep_date + pd.to_timedelta(dep_hour, unit="h")
    df["arr_ts_hour"] = arr_date + pd.to_timedelta(arr_hour, unit="h")
    df["dep_hour"] = dep_hour.values
    df["arr_hour"] = arr_hour.values
    return df


# --------------------------------------------------------------------------- #
# fact_airport_hour
# --------------------------------------------------------------------------- #
def build_fact_airport_hour(
    flight_df: pd.DataFrame, seats_per_flight: int | None
) -> pd.DataFrame:
    """Agregado por (aeropuerto, hora) combinando salidas y llegadas."""
    f = flight_df
    operated = ~f["Cancelled"]

    # --- Salidas (agrupadas por Origin, dep_ts_hour) ---
    dep = (
        f.assign(
            _del15=f["DepDel15"].fillna(0.0),
            _op=operated.astype(int),
        )
        .groupby(["Origin", "dep_ts_hour"], observed=True)
        .agg(
            sched_dep=("Origin", "size"),
            dep_operated=("_op", "sum"),
            dep_cancelled=("Cancelled", "sum"),
            dep_delay_mean=("DepDelay", "mean"),
            dep_del15=("_del15", "sum"),
        )
        .reset_index()
        .rename(columns={"Origin": "airport_code", "dep_ts_hour": "ts_hour"})
    )

    # --- Llegadas (agrupadas por Dest, arr_ts_hour) ---
    arr = (
        f.assign(
            _del15=f["ArrDel15"].fillna(0.0),
            _op=operated.astype(int),
        )
        .groupby(["Dest", "arr_ts_hour"], observed=True)
        .agg(
            sched_arr=("Dest", "size"),
            arr_operated=("_op", "sum"),
            arr_cancelled=("Cancelled", "sum"),
            arr_delay_mean=("ArrDelay", "mean"),
            arr_delay_median=("ArrDelay", "median"),
            arr_del15=("_del15", "sum"),
        )
        .reset_index()
        .rename(columns={"Dest": "airport_code", "arr_ts_hour": "ts_hour"})
    )

    fact = arr.merge(dep, on=["airport_code", "ts_hour"], how="outer")

    count_cols = [
        "sched_arr", "arr_operated", "arr_cancelled", "arr_del15",
        "sched_dep", "dep_operated", "dep_cancelled", "dep_del15",
    ]
    for c in count_cols:
        if c in fact:
            fact[c] = fact[c].fillna(0).astype(int)

    fact["ts_hour"] = pd.to_datetime(fact["ts_hour"])
    fact["date"] = fact["ts_hour"].dt.normalize()
    fact["hour"] = fact["ts_hour"].dt.hour
    fact["day_of_week"] = fact["ts_hour"].dt.dayofweek
    fact["month"] = fact["ts_hour"].dt.month
    fact["year"] = fact["ts_hour"].dt.year

    if seats_per_flight:
        fact["est_arr_passengers"] = fact["sched_arr"] * seats_per_flight
        fact["est_dep_passengers"] = fact["sched_dep"] * seats_per_flight

    return fact.sort_values(["airport_code", "ts_hour"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# agg_baseline_volume
# --------------------------------------------------------------------------- #
def build_agg_baseline_volume(fact: pd.DataFrame) -> pd.DataFrame:
    """Volumen 'habitual' por (aeropuerto, día-de-semana, hora).

    Promedia el volumen programado horario observado a lo largo de los dos años,
    sirviendo como referencia para "tráfico vs. lo habitual" en el dashboard.
    """
    base = (
        fact.groupby(["airport_code", "day_of_week", "hour"], observed=True)
        .agg(
            baseline_sched_arr=("sched_arr", "mean"),
            baseline_sched_dep=("sched_dep", "mean"),
            baseline_arr_delay_mean=("arr_delay_mean", "mean"),
            n_observations=("sched_arr", "size"),
        )
        .reset_index()
    )
    base["day_name"] = base["day_of_week"].map(dict(enumerate(DAY_NAMES)))
    return base


# --------------------------------------------------------------------------- #
# Tablas de factores (Página 2)
# --------------------------------------------------------------------------- #
def build_agg_airline(flight_df: pd.DataFrame) -> pd.DataFrame:
    op = flight_df[~flight_df["Cancelled"]]
    g = (
        op.assign(month=op["FlightDate"].dt.month)
        .groupby(["Airline", "month"], observed=True)
        .agg(
            flights=("Airline", "size"),
            arr_delay_mean=("ArrDelay", "mean"),
            dep_delay_mean=("DepDelay", "mean"),
            arr_del15_rate=("ArrDel15", "mean"),
        )
        .reset_index()
    )
    return g


def build_agg_route(flight_df: pd.DataFrame) -> pd.DataFrame:
    op = flight_df[~flight_df["Cancelled"]]
    g = (
        op.groupby(["Origin", "Dest"], observed=True)
        .agg(
            flights=("Origin", "size"),
            arr_delay_mean=("ArrDelay", "mean"),
            arr_del15_rate=("ArrDel15", "mean"),
            mean_distance=("Distance", "mean"),
        )
        .reset_index()
    )
    return g.sort_values("flights", ascending=False).reset_index(drop=True)


def build_agg_distance_bucket(flight_df: pd.DataFrame) -> pd.DataFrame:
    op = flight_df[~flight_df["Cancelled"]].copy()
    bins = [0, 250, 500, 750, 1000, 1500, 2000, np.inf]
    labels = ["0-250", "250-500", "500-750", "750-1000", "1000-1500", "1500-2000", "2000+"]
    op["distance_bucket"] = pd.cut(op["Distance"], bins=bins, labels=labels, right=False)
    g = (
        op.groupby("distance_bucket", observed=True)
        .agg(
            flights=("Distance", "size"),
            arr_delay_mean=("ArrDelay", "mean"),
            arr_del15_rate=("ArrDel15", "mean"),
        )
        .reset_index()
    )
    return g


# --------------------------------------------------------------------------- #
# Dimensiones
# --------------------------------------------------------------------------- #
def build_dim_airport(airports: list[str]) -> pd.DataFrame:
    coords_path = get_project_root() / AIRPORT_COORDS
    coords = pd.read_csv(coords_path)
    dim = coords[coords["iata"].isin(airports)].copy()
    dim = dim.rename(columns={"iata": "airport_code"})
    missing = sorted(set(airports) - set(dim["airport_code"]))
    if missing:
        logger.warning("Aeropuertos sin coordenadas (%d): %s", len(missing), missing)
    return dim.sort_values("airport_code").reset_index(drop=True)


def build_dim_date(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(min_date.normalize(), max_date.normalize(), freq="D")
    dim = pd.DataFrame({"date": dates})
    dim["year"] = dim["date"].dt.year
    dim["month"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.strftime("%b")
    dim["day"] = dim["date"].dt.day
    dim["day_of_week"] = dim["date"].dt.dayofweek
    dim["day_name"] = dim["day_of_week"].map(dict(enumerate(DAY_NAMES)))
    dim["is_weekend"] = dim["day_of_week"] >= 5
    dim["quarter"] = dim["date"].dt.quarter
    dim["week_of_year"] = dim["date"].dt.isocalendar().week.astype(int)
    season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
              5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall",
              10: "Fall", 11: "Fall"}
    dim["season"] = dim["month"].map(season)
    return dim


def build_dim_hour() -> pd.DataFrame:
    hours = list(range(24))
    dim = pd.DataFrame({"hour": hours})
    dim["hour_label"] = dim["hour"].map(lambda h: f"{h:02d}:00")

    def part(h: int) -> str:
        if h < 6:
            return "Night"
        if h < 12:
            return "Morning"
        if h < 18:
            return "Afternoon"
        return "Evening"

    dim["part_of_day"] = dim["hour"].map(part)
    return dim


# --------------------------------------------------------------------------- #
# Enriquecimiento de predicciones
# --------------------------------------------------------------------------- #
def enrich_predictions(
    preds_df: pd.DataFrame, fact: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    """Añade volumen programado, baseline y métricas derivadas a las predicciones."""
    if preds_df.empty:
        return preds_df

    df = preds_df.copy()
    df["target_ts_dt"] = pd.to_datetime(df["target_ts"])
    df["target_hour_bucket"] = df["target_ts_dt"].dt.floor("h")
    df["dow"] = df["target_ts_dt"].dt.dayofweek
    df["hour"] = df["target_ts_dt"].dt.hour

    # Volumen programado real en la hora objetivo (desde fact_airport_hour).
    sched = fact[["airport_code", "ts_hour", "sched_arr", "sched_dep"]].rename(
        columns={"ts_hour": "target_hour_bucket"}
    )
    df = df.merge(sched, on=["airport_code", "target_hour_bucket"], how="left")

    # Baseline "habitual" para esa franja (aeropuerto, dow, hora).
    base = baseline[["airport_code", "day_of_week", "hour", "baseline_sched_arr"]].rename(
        columns={"day_of_week": "dow"}
    )
    df = df.merge(base, on=["airport_code", "dow", "hour"], how="left")

    df["sched_arr"] = df["sched_arr"].fillna(0)
    df["sched_dep"] = df["sched_dep"].fillna(0)
    df["expected_delayed_flights"] = df["pct_arr_delayed_15"] * df["sched_arr"]
    df["traffic_vs_usual"] = np.where(
        df["baseline_sched_arr"].fillna(0) > 0,
        df["sched_arr"] / df["baseline_sched_arr"] - 1.0,
        np.nan,
    )

    return df.drop(columns=["target_ts_dt", "target_hour_bucket"])


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def write_table(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    """Escribe una tabla en Parquet (canónico) y CSV (importable en Power BI)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{name}.parquet", index=False)
    df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    logger.info("  %-22s %8d filas, %2d columnas", name, len(df), df.shape[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta tablas para Power BI")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint .pt del campeón (para predicciones)")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test", "all"), help="Split a predecir")
    parser.add_argument("--out-dir", type=str, default=None, help="Directorio de salida (def: outputs/powerbi)")
    parser.add_argument("--seats-per-flight", type=int, default=None, help="Asientos/vuelo para pasajeros estimados (0 para omitir)")
    parser.add_argument("--no-predictions", action="store_true", help="Sólo tablas base (sin inferencia)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("reproducibility", {}).get("seed", 42))
    pbi_cfg = config.get("powerbi", {})

    out_dir = Path(args.out_dir or pbi_cfg.get("out_dir") or (get_output_dir() / "powerbi"))
    seats = args.seats_per_flight
    if seats is None:
        seats = pbi_cfg.get("seats_per_flight", DEFAULT_SEATS_PER_FLIGHT)
    seats = seats or None  # 0 -> None (omitir pasajeros)

    logger.info("=" * 60)
    logger.info("EXPORT POWER BI -> %s", out_dir)
    logger.info("config=%s | split=%s | seats/vuelo=%s", args.config, args.split, seats)
    logger.info("=" * 60)

    # --- Carga única del dataset crudo ---
    data_dir = get_data_dir("raw", config)
    years = config["data"].get("years", [2018])
    columns = config["data"].get("columns")
    raw_df = load_multiple_years(
        data_dir, years, columns=columns, sample_frac=None, skip_missing=True,
    )

    # Preprocesado una sola vez (misma definición de aeropuertos top-N que el
    # entrenamiento). ``pre_df`` se reutiliza para la inferencia; ``airports``
    # para acotar las agregaciones base a la red modelada.
    top_n = config.get("graph", {}).get("top_n_airports", 30)
    pre_df, airports = preprocess_pipeline(raw_df.copy(), top_n_airports=top_n)
    logger.info("Aeropuertos modelados: %d", len(airports))

    # --- Tablas base (conservan cancelados) ---
    logger.info("Construyendo tablas base...")
    flight_df = build_flight_frame(raw_df, airports)
    fact_airport_hour = build_fact_airport_hour(flight_df, seats)
    agg_baseline = build_agg_baseline_volume(fact_airport_hour)
    dim_airport = build_dim_airport(airports)
    dim_date = build_dim_date(flight_df["FlightDate"].min(), flight_df["FlightDate"].max())
    dim_hour = build_dim_hour()
    agg_airline = build_agg_airline(flight_df)
    agg_route = build_agg_route(flight_df)
    agg_distance = build_agg_distance_bucket(flight_df)

    logger.info("Escribiendo tablas base en %s ...", out_dir)
    write_table(dim_airport, "dim_airport", out_dir)
    write_table(dim_date, "dim_date", out_dir)
    write_table(dim_hour, "dim_hour", out_dir)
    write_table(fact_airport_hour, "fact_airport_hour", out_dir)
    write_table(agg_baseline, "agg_baseline_volume", out_dir)
    write_table(agg_airline, "agg_airline", out_dir)
    write_table(agg_route, "agg_route", out_dir)
    write_table(agg_distance, "agg_distance_bucket", out_dir)

    # --- Predicciones (opcional) ---
    if args.no_predictions:
        logger.info("--no-predictions: se omite la inferencia.")
        logger.info("Export base completo.")
        return

    if not args.checkpoint:
        logger.warning(
            "Sin --checkpoint: se omiten las predicciones. Pasa --checkpoint "
            "para generar fact_predictions, o usa --no-predictions."
        )
        return

    logger.info("Generando predicciones (split=%s)...", args.split)
    from scripts.predict import generate_predictions_from_df  # import perezoso

    preds_df = generate_predictions_from_df(
        pre_df, airports, config, args.checkpoint,
        split=args.split, include_actuals=True,
    )
    preds_df = enrich_predictions(preds_df, fact_airport_hour, agg_baseline)
    write_table(preds_df, "fact_predictions", out_dir)

    logger.info("Export completo: %s", out_dir)


if __name__ == "__main__":
    main()
