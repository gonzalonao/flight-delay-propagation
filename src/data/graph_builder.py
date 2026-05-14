"""Construcción de grafos de aeropuertos para modelos GNN.

Transforma datos tabulares de vuelos en grafos donde los nodos son
aeropuertos y las aristas son rutas aéreas. Genera snapshots temporales
con features agregadas por aeropuerto y targets continuos de retraso
(minutos de retraso promedio en la siguiente ventana temporal).

INVARIANTE DE FUGA TEMPORAL (no leakage)
-----------------------------------------
En el snapshot con `current_end = T`, las features y aristas de cada
nodo solo pueden depender de:

  * Cualquier vuelo con `arr_timestamp < T` (completado antes de T).
  * Las columnas de schedule (`CRS*`, `Distance`, `Airline`, `Origin`,
    `Dest`) de cualquier vuelo — son conocidas a priori y se usan tanto
    para features históricas como para features exógenas del target.

NO pueden depender de columnas Class B (post-hoc: `DepTime`, `ArrTime`,
`WheelsOff/On`, `TaxiOut/In`, `ActualElapsedTime`, `AirTime`,
`DepDelay`, `ArrDelay`, `DepDel15`, `ArrDel15`, `CarrierDelay`,
`WeatherDelay`, `NASDelay`, `SecurityDelay`, `LateAircraftDelay`,
`Cancelled`, `Diverted`) de vuelos cuyo `arr_timestamp >= T`. Las
features rich precomputadas en `_build_history_lookups` agrupan
explícitamente por hora de llegada (`arr_timestamp`) para que la
ventana histórica se cierre estrictamente antes de T.
"""

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.weather import (
    NUM_WEATHER_CATEGORIES,
    WEATHER_SCHEMA_VERSION,
    load_weather_for_airports,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Causas BTS que vienen en el dataset Combined_Flights cuando ArrDelay≥15.
# Si una columna no está cargada (Parquet antiguo), se omite del feature
# correspondiente.
BTS_CAUSE_COLUMNS = (
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
)

# Lags (en pasos de `window_delta`) usados como features históricas.
ARR_DELAY_LAGS = (1, 3, 6, 24)
ROLLING_WINDOW_HOURS = 6

# Tamaño del bloque H (meteorología) cuando ``weather_lookups`` está
# activado. Histórico = mean wind + max gust + sum precip + mean cloud
# + 5 one-hots de categoría dominante; futuro idéntico por horizonte.
# Mantenerlos iguales hace que la cuenta total sea ``9 + 9 * H``.
WEATHER_BLOCK_HIST_SIZE = 4 + NUM_WEATHER_CATEGORIES   # = 9
WEATHER_BLOCK_FUTURE_SIZE_PER_H = 4 + NUM_WEATHER_CATEGORIES  # = 9


@dataclass
class HistoryLookups:
    """Estructuras precomputadas para feature engineering por snapshot.

    Construidas una sola vez al inicio de `create_temporal_graphs`,
    permiten que cada llamada a `compute_node_features_rich` y
    `compute_dynamic_edge_attr` sea O(num_airports × num_horizons) o
    O(num_edges) sin revisitar el DataFrame entero.

    Las series indexadas por nodo usan `(airport, hour_bucket)`; las
    indexadas por arista usan `(origin, dest, hour_bucket)`. En todos
    los casos `hour_bucket = pd.Timestamp` alineado al inicio de la
    ventana.
    """

    arr_delay_by_airport_hour: pd.Series  # mean ArrDelay (Class B)
    sched_arr_count: pd.Series            # nº vuelos programados a llegar (Class A)
    sched_dep_count: pd.Series            # nº vuelos programados a salir (Class A)
    sched_arr_from_top10: pd.Series       # nº de top-10 origins → este aeropuerto
    sched_arr_mean_distance: pd.Series    # distancia media programada de inbound
    rolling_mean_arr_delay: pd.Series     # rolling 6h de ArrDelay (Class B)
    rolling_std_arr_delay: pd.Series      # rolling 6h std (Class B)
    top10_origins: list[str]              # IATAs de los 10 aeropuertos más conectados
    # Por arista (Origin, Dest, bucket):
    route_recent_arr_delay: pd.Series     # rolling 6h de ArrDelay por ruta (Class B)
    route_sched_count: pd.Series          # nº vuelos programados por ruta y dep_bucket (Class A)


@dataclass
class WeatherLookups:
    """Estructura precomputada para acceso O(1) a observaciones horarias.

    ``airport_weather`` mapea IATA → DataFrame indexado por
    ``timestamp`` (horario, redondeado al inicio de la hora) con columnas:
    ``wind_speed_10m``, ``wind_gusts_10m``, ``precipitation``,
    ``cloud_cover``, ``weather_category``. Los aeropuertos sin datos
    (sea por error de red o porque su IATA no figura en el CSV de
    coordenadas) NO aparecen en el dict — los consumidores devuelven el
    vector cero al fallar el lookup, lo que el modelo interpreta como
    "sin señal meteorológica" (equivalente a la rama ``weather_lookups
    is None`` para ese aeropuerto en concreto).

    Attributes:
        airport_weather: Dict IATA → DataFrame horario.
        enabled_params: Conjunto de claves activas {"wind", "precip_cloud",
            "category"}. Permite encender/apagar grupos individuales sin
            reconstruir la caché HTTP — los grupos apagados se sobreescriben
            con 0 en ``compute_node_features_rich``. Esto da soporte directo
            a los toggles de OPTIMIZATIONS.md.
    """

    airport_weather: dict[str, pd.DataFrame] = field(default_factory=dict)
    enabled_params: frozenset[str] = field(
        default_factory=lambda: frozenset({"wind", "precip_cloud", "category"})
    )

    def empty(self) -> bool:
        return not self.airport_weather


def _weather_window_agg(
    weather: WeatherLookups,
    iata: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    """Agrega observaciones horarias en ``[start, end)`` para un IATA.

    Devuelve un vector denso de ``WEATHER_BLOCK_HIST_SIZE`` floats:

      [0]      mean(wind_speed_10m) en km/h
      [1]      max(wind_gusts_10m) en km/h
      [2]      sum(precipitation) en mm
      [3]      mean(cloud_cover) en %
      [4..8]   one-hot de la categoría DOMINANTE (moda) en la ventana

    Las columnas correspondientes a grupos desactivados (``enabled_params``)
    se ponen a 0 — toggle a nivel de feature group para ablaciones.
    """
    out = np.zeros(WEATHER_BLOCK_HIST_SIZE, dtype=np.float32)
    df = weather.airport_weather.get(iata)
    if df is None or df.empty:
        return out

    mask = (df.index >= start) & (df.index < end)
    window = df.loc[mask]
    if window.empty:
        return out

    if "wind" in weather.enabled_params:
        wind = window["wind_speed_10m"].to_numpy(dtype="float32")
        gust = window["wind_gusts_10m"].to_numpy(dtype="float32")
        wind_valid = wind[~np.isnan(wind)]
        gust_valid = gust[~np.isnan(gust)]
        if wind_valid.size > 0:
            out[0] = float(np.mean(wind_valid))
        if gust_valid.size > 0:
            out[1] = float(np.max(gust_valid))

    if "precip_cloud" in weather.enabled_params:
        precip = window["precipitation"].to_numpy(dtype="float32")
        cloud = window["cloud_cover"].to_numpy(dtype="float32")
        precip_valid = precip[~np.isnan(precip)]
        cloud_valid = cloud[~np.isnan(cloud)]
        if precip_valid.size > 0:
            out[2] = float(np.sum(precip_valid))
        if cloud_valid.size > 0:
            out[3] = float(np.mean(cloud_valid))

    if "category" in weather.enabled_params:
        cats = window["weather_category"].to_numpy(dtype="int8")
        if cats.size > 0:
            # Moda: argmax sobre el histograma. ``np.bincount`` con
            # ``minlength`` garantiza un slot por categoría aunque la
            # ventana no contenga todas.
            hist = np.bincount(cats, minlength=NUM_WEATHER_CATEGORIES)
            dominant = int(np.argmax(hist))
            out[4 + dominant] = 1.0

    return out


def _weather_point_obs(
    weather: WeatherLookups,
    iata: str,
    timestamp: pd.Timestamp,
) -> np.ndarray:
    """Lookup horario alineado a la hora de ``timestamp``.

    Devuelve un vector denso de ``WEATHER_BLOCK_FUTURE_SIZE_PER_H`` floats
    con el mismo esquema que :func:`_weather_window_agg` excepto que las
    columnas 0..3 son la observación puntual (no agregada) y 4..8 es el
    one-hot de la categoría observada en esa hora exacta.

    Justificación de leakage: el target en T+h es el delay observado,
    no la meteorología; el feature es exógeno. Usamos la observación en
    T+h como "perfect-forecast proxy" — sobrestima la utilidad real (un
    sistema de producción usaría una previsión con error), pero ese gap
    se aborda en una ablación futura (OPTIMIZATIONS.md).
    """
    out = np.zeros(WEATHER_BLOCK_FUTURE_SIZE_PER_H, dtype=np.float32)
    df = weather.airport_weather.get(iata)
    if df is None or df.empty:
        return out

    hour = timestamp.floor("h")
    if hour not in df.index:
        return out
    row = df.loc[hour]
    if isinstance(row, pd.DataFrame):
        # Lookup que coincidió con varios timestamps (no debería pasar con
        # el floor("h") pero por defensiva nos quedamos con el primero).
        row = row.iloc[0]

    if "wind" in weather.enabled_params:
        wind_v = row.get("wind_speed_10m", np.nan)
        gust_v = row.get("wind_gusts_10m", np.nan)
        if not pd.isna(wind_v):
            out[0] = float(wind_v)
        if not pd.isna(gust_v):
            out[1] = float(gust_v)

    if "precip_cloud" in weather.enabled_params:
        precip_v = row.get("precipitation", np.nan)
        cloud_v = row.get("cloud_cover", np.nan)
        if not pd.isna(precip_v):
            out[2] = float(precip_v)
        if not pd.isna(cloud_v):
            out[3] = float(cloud_v)

    if "category" in weather.enabled_params:
        cat = row.get("weather_category", 0)
        if not pd.isna(cat):
            cat_idx = int(cat)
            if 0 <= cat_idx < NUM_WEATHER_CATEGORIES:
                out[4 + cat_idx] = 1.0

    return out


def _build_weather_lookups(
    weather_df: pd.DataFrame | None,
    airports: list[str] | None = None,
    enabled_params: frozenset[str] | None = None,
) -> WeatherLookups | None:
    """Construye ``WeatherLookups`` a partir del DataFrame multi-aeropuerto.

    Si ``weather_df`` es ``None`` o vacío, devuelve ``None`` (lo que
    desactiva el bloque H aguas abajo, manteniendo el contrato anterior
    de feature dim).

    El DataFrame de entrada tiene MultiIndex ``(iata, timestamp)`` —
    la salida es un dict ``{iata: df_indexado_por_timestamp}`` con
    ``timestamp`` redondeado a la hora (``floor("h")``) para que el
    lookup ``_weather_point_obs`` cierre por hora exacta. Hashable
    keys habilitan O(1) por airport.
    """
    if weather_df is None or weather_df.empty:
        return None

    if enabled_params is None:
        enabled_params = frozenset({"wind", "precip_cloud", "category"})

    airport_weather: dict[str, pd.DataFrame] = {}
    target_airports = set(airports) if airports is not None else None

    for iata, group in weather_df.groupby(level="iata"):
        if target_airports is not None and iata not in target_airports:
            continue
        per_airport = group.droplevel("iata").copy()
        per_airport.index = per_airport.index.floor("h")
        # Eliminar duplicados manteniendo el primer valor por hora.
        per_airport = per_airport[~per_airport.index.duplicated(keep="first")]
        airport_weather[iata] = per_airport

    return WeatherLookups(
        airport_weather=airport_weather,
        enabled_params=enabled_params,
    )


def _build_history_lookups(
    df: pd.DataFrame,
    airport_map: dict[str, int],
    window_delta: pd.Timedelta,
) -> HistoryLookups:
    """Precomputa las series por (airport, hour_bucket) usadas como
    lookups en feature engineering.

    Las features Class B (que dependen de actuales) se agregan por la
    hora *de llegada* (`arr_timestamp`) — así, al consultar `lag=k` desde
    `current_end=T`, recogemos vuelos que aterrizaron en
    [T - k·delta, T - (k-1)·delta), que están estrictamente antes de T.

    Las features Class A (schedule) se agregan por la hora *programada*
    de llegada o salida según corresponda — son válidas tanto para
    ventanas pasadas como futuras.

    Args:
        df: DataFrame completo con `arr_timestamp` y `timestamp` ya
            calculadas y ordenadas cronológicamente.
        airport_map: Mapeo IATA → índice.
        window_delta: Tamaño de ventana temporal (e.g. 1h o 2h).

    Returns:
        ``HistoryLookups`` con todas las series precomputadas.
    """
    valid_airports = set(airport_map.keys())

    # Normalizar a buckets alineados al inicio de la ventana.
    # `pd.Timestamp.floor("Nh")` redondea hacia abajo a múltiplo del
    # tamaño de la ventana.
    freq = f"{int(window_delta.total_seconds() // 3600)}h"

    df_arr = df.copy()
    df_arr["arr_bucket"] = df_arr["arr_timestamp"].dt.floor(freq)
    df_arr["dep_bucket"] = df_arr["timestamp"].dt.floor(freq)

    # ── Class B: ArrDelay agrupado por (Dest, arr_bucket) ─────────────
    arr_mask = df_arr["Dest"].isin(valid_airports) & df_arr["ArrDelay"].notna()
    arr_grp = df_arr[arr_mask].groupby(["Dest", "arr_bucket"], observed=True)
    arr_delay_by_airport_hour = arr_grp["ArrDelay"].mean().rename("arr_delay_mean")

    # Rolling 6h por aeropuerto (sobre la serie horaria) — sirve como
    # "media del comportamiento reciente" en cualquier T.
    rolling_steps = max(1, ROLLING_WINDOW_HOURS // max(1, int(window_delta.total_seconds() // 3600)))
    rolling_mean = (
        arr_delay_by_airport_hour
        .groupby(level=0, observed=True)
        .rolling(window=rolling_steps, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # Para std necesitamos una serie con todas las observaciones de cada
    # vuelo (no solo la media por bucket); aproximamos con std de las
    # medias horarias en la ventana, suficiente como señal de volatilidad.
    # Si ``rolling_steps == 1`` (caso degenerado donde la ventana cubre
    # exactamente ROLLING_WINDOW_HOURS o más), pandas rechaza
    # ``min_periods=2 > window=1``. En ese caso la std no es definible
    # con una sola observación; clampamos ``min_periods`` para que el
    # cálculo devuelva NaN (luego se rellena con 0.0). En todas las
    # configs de producción (``temporal_window_hours`` ∈ {1, 2}) el
    # clamp no se activa y la pérdida es estrictamente la misma.
    rolling_std = (
        arr_delay_by_airport_hour
        .groupby(level=0, observed=True)
        .rolling(window=rolling_steps, min_periods=min(2, rolling_steps))
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # ── Class A: schedule (Dest, arr_bucket) ─────────────────────────
    sched_arr_mask = df_arr["Dest"].isin(valid_airports)
    sched_arr_grp = df_arr[sched_arr_mask].groupby(
        ["Dest", "arr_bucket"], observed=True
    )
    sched_arr_count = sched_arr_grp.size().rename("sched_arr_count").astype("float32")
    if "Distance" in df_arr.columns:
        sched_arr_mean_distance = (
            sched_arr_grp["Distance"].mean().rename("sched_arr_mean_distance")
        )
    else:
        # Serie vacía con el mismo MultiIndex que la rama "with Distance"
        # (``["Dest", "arr_bucket"]``) → todos los lookups dan default vía
        # KeyError sin tropezar con ``IndexingError`` por una sola
        # dimensión. En producción el loader siempre incluye ``Distance``
        # (cf. ``src/data/loader.py``), así que esta rama solo fija el
        # comportamiento de los tests sintéticos sin ``Distance``.
        sched_arr_mean_distance = pd.Series(
            dtype="float32",
            name="sched_arr_mean_distance",
            index=pd.MultiIndex.from_tuples([], names=["Dest", "arr_bucket"]),
        )

    # ── Class A: schedule (Origin, dep_bucket) ───────────────────────
    sched_dep_mask = df_arr["Origin"].isin(valid_airports)
    sched_dep_grp = df_arr[sched_dep_mask].groupby(
        ["Origin", "dep_bucket"], observed=True
    )
    sched_dep_count = sched_dep_grp.size().rename("sched_dep_count").astype("float32")

    # ── Top-10 origins (estructural, una sola vez) ───────────────────
    top10_origins = (
        df_arr[df_arr["Origin"].isin(valid_airports)]
        .groupby("Origin", observed=True).size()
        .nlargest(10).index.tolist()
    )
    top10_set = set(top10_origins)
    is_from_top10 = df_arr["Origin"].isin(top10_set) & df_arr["Dest"].isin(valid_airports)
    sched_arr_from_top10 = (
        df_arr[is_from_top10]
        .groupby(["Dest", "arr_bucket"], observed=True)
        .size()
        .rename("sched_arr_from_top10")
        .astype("float32")
    )

    # ── Por ruta: ArrDelay reciente (rolling 6h) ─────────────────────
    # Class B: agrupado por (Origin, Dest, arr_bucket) → mean ArrDelay,
    # luego rolling sobre la dimensión temporal de cada ruta.
    route_arr_mask = (
        df_arr["Origin"].isin(valid_airports)
        & df_arr["Dest"].isin(valid_airports)
        & df_arr["ArrDelay"].notna()
    )
    route_arr_grp = df_arr[route_arr_mask].groupby(
        ["Origin", "Dest", "arr_bucket"], observed=True
    )
    route_arr_delay_hourly = route_arr_grp["ArrDelay"].mean().rename("route_arr_delay")
    route_recent_arr_delay = (
        route_arr_delay_hourly
        .groupby(level=[0, 1], observed=True)
        .rolling(window=rolling_steps, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # ── Por ruta: programación (Class A, dep_bucket) ─────────────────
    route_sched_mask = (
        df_arr["Origin"].isin(valid_airports)
        & df_arr["Dest"].isin(valid_airports)
    )
    route_sched_count = (
        df_arr[route_sched_mask]
        .groupby(["Origin", "Dest", "dep_bucket"], observed=True)
        .size()
        .rename("route_sched_count")
        .astype("float32")
    )

    return HistoryLookups(
        arr_delay_by_airport_hour=arr_delay_by_airport_hour,
        sched_arr_count=sched_arr_count,
        sched_dep_count=sched_dep_count,
        sched_arr_from_top10=sched_arr_from_top10,
        sched_arr_mean_distance=sched_arr_mean_distance,
        rolling_mean_arr_delay=rolling_mean,
        rolling_std_arr_delay=rolling_std,
        top10_origins=top10_origins,
        route_recent_arr_delay=route_recent_arr_delay,
        route_sched_count=route_sched_count,
    )


def _series_lookup(
    s: pd.Series,
    airport: str,
    bucket: pd.Timestamp,
    default: float = 0.0,
) -> float:
    """Lookup tolerante en una serie indexada por (airport, bucket)."""
    try:
        v = s.loc[(airport, bucket)]
    except KeyError:
        return default
    return float(v) if not pd.isna(v) else default


def _cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """Codificación sin/cos de una variable cíclica."""
    angle = 2.0 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def build_airport_mapping(airports: list[str]) -> dict[str, int]:
    """Crea un mapeo de código IATA a índice numérico.

    Args:
        airports: Lista ordenada de códigos de aeropuerto.

    Returns:
        Diccionario {código_IATA: índice}.
    """
    return {code: idx for idx, code in enumerate(sorted(airports))}


def build_edge_index(
    df: pd.DataFrame,
    airport_map: dict[str, int],
    min_flights: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]:
    """Construye la matriz de adyacencia y las features estáticas de arista.

    Crea aristas bidireccionales entre aeropuertos que tienen al menos
    `min_flights` vuelos en el dataset y devuelve un tensor con 3
    features estáticas por arista (Class A: derivables del schedule):

      Col 0  flight_count_norm        — recuento histórico normalizado
      Col 1  mean_air_time_norm       — AirTime medio histórico
      Col 2  mean_scheduled_distance  — Distance media (Class A)

    Estas tres features no cambian entre snapshots; las dos restantes
    (recent_route_delay, scheduled_flights_next_h_norm) se concatenan
    por snapshot en `compute_dynamic_edge_attr` para llegar a un
    `edge_attr` final de shape ``[num_edges, 5]``.

    Args:
        df: DataFrame con columnas Origin, Dest y opcionalmente
            AirTime / Distance.
        airport_map: Mapeo de código IATA a índice.
        min_flights: Mínimo de vuelos para crear una arista.

    Returns:
        Tupla de:
          * ``edge_index`` ``[2, num_edges]``
          * ``static_edge_attr`` ``[num_edges, 3]``
          * ``edge_pairs`` lista de ``(origin_iata, dest_iata)`` por
            cada arista en orden — usada para construir features
            dinámicas vía lookup en ``HistoryLookups``.
    """
    # Agregaciones por ruta direccional.
    agg_dict: dict = {"count": ("Origin", "size")}
    if "AirTime" in df.columns:
        agg_dict["mean_air_time"] = ("AirTime", "mean")
    if "Distance" in df.columns:
        agg_dict["mean_distance"] = ("Distance", "mean")

    route_stats = (
        df.groupby(["Origin", "Dest"], observed=True)
        .agg(**agg_dict)
        .reset_index()
    )

    # Filtrar rutas con pocos vuelos y aeropuertos no mapeados.
    route_stats = route_stats[route_stats["count"] >= min_flights]
    valid = set(airport_map.keys())
    route_stats = route_stats[
        route_stats["Origin"].isin(valid) & route_stats["Dest"].isin(valid)
    ]

    sources: list[int] = []
    targets: list[int] = []
    counts: list[float] = []
    air_times: list[float] = []
    distances: list[float] = []
    edge_pairs: list[tuple[str, str]] = []

    has_air_time = "mean_air_time" in route_stats.columns
    has_distance = "mean_distance" in route_stats.columns

    for _, row in route_stats.iterrows():
        origin, dest = row["Origin"], row["Dest"]
        src, dst = airport_map[origin], airport_map[dest]
        c = float(row["count"])
        at = float(row["mean_air_time"]) if has_air_time and not pd.isna(row["mean_air_time"]) else 0.0
        di = float(row["mean_distance"]) if has_distance and not pd.isna(row["mean_distance"]) else 0.0
        # Arista bidireccional con las mismas estadísticas (la ruta
        # X→Y y Y→X comparten distancia y tiempo medio aéreo; los
        # delays direccionales viven en `route_recent_arr_delay`).
        sources.extend([src, dst])
        targets.extend([dst, src])
        counts.extend([c, c])
        air_times.extend([at, at])
        distances.extend([di, di])
        edge_pairs.extend([(origin, dest), (dest, origin)])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    counts_t = torch.tensor(counts, dtype=torch.float32)
    air_times_t = torch.tensor(air_times, dtype=torch.float32)
    distances_t = torch.tensor(distances, dtype=torch.float32)

    if counts_t.numel() > 0:
        if counts_t.max() > 0:
            counts_t = counts_t / counts_t.max()
        if air_times_t.max() > 0:
            air_times_t = air_times_t / air_times_t.max()
        if distances_t.max() > 0:
            distances_t = distances_t / distances_t.max()

    static_edge_attr = torch.stack([counts_t, air_times_t, distances_t], dim=1)

    logger.info(
        "Grafo construido: %d nodos, %d aristas (min_flights=%d), "
        "edge_attr estático shape=%s",
        len(airport_map), edge_index.shape[1], min_flights,
        tuple(static_edge_attr.shape),
    )

    return edge_index, static_edge_attr, edge_pairs


def compute_dynamic_edge_attr(
    edge_pairs: list[tuple[str, str]],
    history_lookups: HistoryLookups,
    current_end: pd.Timestamp,
    window_delta: pd.Timedelta,
    arr_delay_norm: float = 60.0,
    sched_count_norm: float = 50.0,
) -> torch.Tensor:
    """Calcula las 2 features de arista dinámicas para un snapshot.

    Devuelve ``[num_edges, 2]`` con:

      Col 0  recent_route_delay_norm        — rolling 6h ArrDelay por
              ruta normalizado (clip ±1) — bucket inmediatamente
              anterior a ``current_end`` (Class B).
      Col 1  scheduled_flights_next_h_norm  — recuento de vuelos
              programados en la ruta para la primera ventana del target
              normalizado (Class A).

    Los normalizadores son constantes para mantener la escala estable
    entre snapshots; el ``normalize_features=true`` posterior ajusta el
    rango final.
    """
    bucket_size = f"{int(window_delta.total_seconds() // 3600)}h"
    history_bucket = (current_end - window_delta).floor(bucket_size)
    target_bucket = current_end.floor(bucket_size)

    n_edges = len(edge_pairs)
    out = torch.zeros(n_edges, 2, dtype=torch.float32)

    recent = history_lookups.route_recent_arr_delay
    sched = history_lookups.route_sched_count

    for i, (origin, dest) in enumerate(edge_pairs):
        # Recent route delay: lookup (origin, dest, history_bucket) en
        # la serie pre-rolling. Default 0 si la ruta no tuvo vuelos.
        try:
            v = recent.loc[(origin, dest, history_bucket)]
            if not pd.isna(v):
                out[i, 0] = float(v) / arr_delay_norm
        except KeyError:
            pass

        # Scheduled flights en la siguiente ventana del target.
        try:
            c = sched.loc[(origin, dest, target_bucket)]
            if not pd.isna(c):
                out[i, 1] = float(c) / sched_count_norm
        except KeyError:
            pass

    # Clip a un rango estable para que el outlier no domine.
    out[:, 0].clamp_(-1.0, 1.0)
    out[:, 1].clamp_(0.0, 1.0)
    return out


def compute_node_features(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
    delay_threshold: float = 15.0,
) -> torch.Tensor:
    """Calcula features por aeropuerto para una ventana temporal.

    Features por nodo (aeropuerto como origen):
    - avg_dep_delay: Retraso de salida promedio
    - std_dep_delay: Desviación estándar del retraso
    - pct_delayed: Porcentaje de vuelos con retraso > umbral
    - num_flights: Número de vuelos (normalizado)
    - avg_arr_delay_incoming: Retraso de llegada promedio (como destino)
    - std_arr_delay_incoming: Desviación estándar del retraso de llegada

    Args:
        window_df: DataFrame filtrado a una ventana temporal.
        airport_map: Mapeo de código IATA a índice.
        delay_threshold: Umbral de minutos para considerar un vuelo retrasado.

    Returns:
        Tensor de node features [num_nodes, num_features].
    """
    num_nodes = len(airport_map)
    num_features = 6
    features = torch.zeros(num_nodes, num_features, dtype=torch.float32)

    if window_df.empty:
        return features

    # Features como aeropuerto de ORIGEN
    origin_groups = window_df.groupby("Origin")
    for airport, group in origin_groups:
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        dep_delays = group["DepDelay"].values
        features[idx, 0] = np.nanmean(dep_delays)
        features[idx, 1] = np.nanstd(dep_delays) if len(dep_delays) > 1 else 0.0
        features[idx, 2] = np.mean(dep_delays > delay_threshold)
        features[idx, 3] = len(dep_delays)

    # Features como aeropuerto de DESTINO (retrasos de llegada)
    dest_groups = window_df.groupby("Dest")
    for airport, group in dest_groups:
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        arr_delays = group["ArrDelay"].values
        features[idx, 4] = np.nanmean(arr_delays)
        features[idx, 5] = np.nanstd(arr_delays) if len(arr_delays) > 1 else 0.0

    # Normalizar num_flights
    max_flights = features[:, 3].max()
    if max_flights > 0:
        features[:, 3] = features[:, 3] / max_flights

    # Reemplazar NaN por 0
    features = torch.nan_to_num(features, nan=0.0)

    return features


def compute_node_features_rich(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
    *,
    current_end: pd.Timestamp,
    window_delta: pd.Timedelta,
    prediction_horizons: list[int],
    history_lookups: HistoryLookups,
    delay_threshold: float = 15.0,
    weather_lookups: WeatherLookups | None = None,
) -> torch.Tensor:
    """Versión enriquecida de ``compute_node_features``.

    Devuelve un vector denso por nodo organizado en bloques con tamaños
    fijos (la dimensión total depende de ``len(prediction_horizons)`` y
    de si la meteorología está activa):

      Bloque                                      | tamaño
      --------------------------------------------|-------
      A) Aggregates current window (Class B)      | 9
      B) Volumen / disrupción (Class A+B)         | 4
      C) Causas BTS (Class B, mean per cause)     | 5
      D) Lags ArrDelay (Class B, vía lookup)      | len(ARR_DELAY_LAGS)
      E) Rolling 6h mean+std (Class B, lookup)    | 2
      F) Calendario cíclico de la ventana actual  | 6
      G) Exógenas futuras por horizonte (Class A) | 5 × len(horizons)
      H) Meteorología (opcional)                  | 9 + 9 × len(horizons)
         si ``weather_lookups is not None``       |

    Para 5 horizontes sin meteorología → 55 features por nodo. Con
    meteorología activa → 55 + 9 + 9·5 = 109 features. Si se desactiva
    el bloque H queda omitido y el feature dim vuelve al valor anterior
    sin romper modelos ya entrenados (la primera capa de cada modelo
    se construye via factory con ``input_dim`` derivado del primer
    snapshot).

    Args:
        window_df: vuelos completados en la ventana de input
            ``[current_end - window_delta, current_end)`` — usados para
            las agregaciones del bloque A, B (parcial), C.
        airport_map: IATA → índice.
        current_end: instante T que cierra la ventana de input.
        window_delta: duración de la ventana.
        prediction_horizons: horizontes futuros (e.g. [1,2,4,6,8]).
        history_lookups: estructuras precomputadas.
        delay_threshold: umbral en min para % delayed.
        weather_lookups: estructura precomputada de meteorología. ``None``
            desactiva el bloque H (default — preserva el contrato
            histórico).
    """
    num_nodes = len(airport_map)
    n_lags = len(ARR_DELAY_LAGS)
    n_horizons = len(prediction_horizons)
    feat_dim = 9 + 4 + 5 + n_lags + 2 + 6 + 5 * n_horizons
    if weather_lookups is not None:
        feat_dim += WEATHER_BLOCK_HIST_SIZE + WEATHER_BLOCK_FUTURE_SIZE_PER_H * n_horizons
    features = torch.zeros(num_nodes, feat_dim, dtype=torch.float32)

    # Class B (post-hoc) features solo pueden derivarse de vuelos cuyo
    # arr_timestamp < current_end (i.e., el vuelo ha aterrizado antes de T).
    # Ver INVARIANTE DE FUGA en el module docstring.
    if "arr_timestamp" in window_df.columns and not window_df.empty:
        completed_df = window_df[window_df["arr_timestamp"] < current_end]
    else:
        completed_df = window_df

    # ── A. Aggregates de la ventana actual (Class B → completed_df) ──
    if not completed_df.empty:
        # Por destino: ArrDelay (signal headline, alineado con el target).
        dest_g = completed_df.groupby("Dest", observed=True)
        for airport, group in dest_g:
            if airport not in airport_map:
                continue
            i = airport_map[airport]
            arr = group["ArrDelay"].values
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                features[i, 0] = float(np.mean(arr))
                features[i, 1] = float(np.std(arr)) if arr.size > 1 else 0.0
                features[i, 2] = float(np.percentile(arr, 75))
                features[i, 3] = float(np.percentile(arr, 90))
                features[i, 6] = float(np.mean(arr > delay_threshold))
                features[i, 7] = float(np.mean(arr > 60.0))
            if "TaxiIn" in group.columns:
                ti = group["TaxiIn"].dropna().values
                if ti.size > 0:
                    features[i, 8] = float(np.mean(ti))

        # Por origen: DepDelay (estado endógeno) + TaxiOut.
        ori_g = completed_df.groupby("Origin", observed=True)
        for airport, group in ori_g:
            if airport not in airport_map:
                continue
            i = airport_map[airport]
            dep = group["DepDelay"].values
            dep = dep[~np.isnan(dep)]
            if dep.size > 0:
                features[i, 4] = float(np.mean(dep))
                features[i, 5] = float(np.std(dep)) if dep.size > 1 else 0.0
            if "TaxiOut" in group.columns:
                to = group["TaxiOut"].dropna().values
                if to.size > 0:
                    # Reusa la columna 8 si TaxiIn faltó; si ambas existen,
                    # promediamos para evitar inflar la dimensión.
                    if features[i, 8] != 0.0:
                        features[i, 8] = (features[i, 8] + float(np.mean(to))) / 2.0
                    else:
                        features[i, 8] = float(np.mean(to))

    # ── B. Volumen / disrupción ─────────────────────────────────────
    # 9: num_arrivals_completed (Class B: solo vuelos ya aterrizados)
    # 10: num_departures_completed (Class A-ish: schedule del input window)
    # 11: num_cancellations  12: num_diversions  (Class A: conocidas a priori)
    if not completed_df.empty:
        for airport, group in completed_df.groupby("Dest", observed=True):
            if airport in airport_map:
                features[airport_map[airport], 9] = float(len(group))
    if not window_df.empty:
        for airport, group in window_df.groupby("Origin", observed=True):
            if airport in airport_map:
                features[airport_map[airport], 10] = float(len(group))
        if "Cancelled" in window_df.columns:
            cancelled_df = window_df[window_df["Cancelled"].astype(bool)]
            for airport, group in cancelled_df.groupby("Origin", observed=True):
                if airport in airport_map:
                    features[airport_map[airport], 11] = float(len(group))
        if "Diverted" in window_df.columns:
            diverted_df = window_df[window_df["Diverted"].astype(bool)]
            for airport, group in diverted_df.groupby("Origin", observed=True):
                if airport in airport_map:
                    features[airport_map[airport], 12] = float(len(group))
        # Normalizar arrivals/departures por máximo del snapshot.
        for col in (9, 10):
            mx = float(features[:, col].max())
            if mx > 0:
                features[:, col] = features[:, col] / mx

    # ── C. BTS causas (Class B → completed_df, por destino) ──────────
    if not completed_df.empty:
        bts_present = [c for c in BTS_CAUSE_COLUMNS if c in completed_df.columns]
        if bts_present:
            for airport, group in completed_df.groupby("Dest", observed=True):
                if airport not in airport_map:
                    continue
                i = airport_map[airport]
                for k, col in enumerate(BTS_CAUSE_COLUMNS):
                    if col in bts_present:
                        v = group[col].mean()
                        features[i, 13 + k] = 0.0 if pd.isna(v) else float(v)

    # ── D. Lags ArrDelay (lookup en serie horaria precomputada) ──────
    base_col = 13 + 5
    for lag_idx, lag in enumerate(ARR_DELAY_LAGS):
        bucket = (current_end - lag * window_delta).floor(
            f"{int(window_delta.total_seconds() // 3600)}h"
        )
        for airport, i in airport_map.items():
            features[i, base_col + lag_idx] = _series_lookup(
                history_lookups.arr_delay_by_airport_hour, airport, bucket
            )

    # ── E. Rolling 6h mean+std (lookup en bucket previo a current_end) ─
    base_col += n_lags
    rolling_bucket = (current_end - window_delta).floor(
        f"{int(window_delta.total_seconds() // 3600)}h"
    )
    for airport, i in airport_map.items():
        features[i, base_col] = _series_lookup(
            history_lookups.rolling_mean_arr_delay, airport, rolling_bucket
        )
        features[i, base_col + 1] = _series_lookup(
            history_lookups.rolling_std_arr_delay, airport, rolling_bucket
        )

    # ── F. Calendario cíclico (mismo valor para todos los nodos) ─────
    base_col += 2
    h_sin, h_cos = _cyclic_encode(current_end.hour, 24)
    dow_sin, dow_cos = _cyclic_encode(current_end.dayofweek, 7)
    m_sin, m_cos = _cyclic_encode(current_end.month - 1, 12)
    cyclic = torch.tensor([h_sin, h_cos, dow_sin, dow_cos, m_sin, m_cos],
                          dtype=torch.float32)
    features[:, base_col:base_col + 6] = cyclic.unsqueeze(0).expand(num_nodes, -1)

    # ── G. Exógenas futuras por horizonte (Class A) ──────────────────
    base_col += 6
    bucket_size = f"{int(window_delta.total_seconds() // 3600)}h"
    for h_idx, h in enumerate(prediction_horizons):
        h_start = current_end + (h - 1) * window_delta
        h_bucket = h_start.floor(bucket_size)
        col_offset = base_col + 5 * h_idx
        for airport, i in airport_map.items():
            features[i, col_offset + 0] = _series_lookup(
                history_lookups.sched_arr_count, airport, h_bucket
            )
            features[i, col_offset + 1] = _series_lookup(
                history_lookups.sched_dep_count, airport, h_bucket
            )
            features[i, col_offset + 2] = _series_lookup(
                history_lookups.sched_arr_from_top10, airport, h_bucket
            )
            features[i, col_offset + 3] = _series_lookup(
                history_lookups.sched_arr_mean_distance, airport, h_bucket
            )
        # Hora-del-día del centro del horizonte target (cíclica, 1 nº).
        # Usamos sin como única columna por horizonte para no añadir 2×5
        # más; el coseno se omite porque la hora del día es ya rica en C.
        target_center_hour = (h_start + window_delta / 2).hour
        h_sin_target, _ = _cyclic_encode(target_center_hour, 24)
        features[:, col_offset + 4] = h_sin_target

    # ── H. Meteorología (opcional, exógena) ──────────────────────────
    # H1: agregados sobre la ventana de input [current_end - W, current_end).
    # H2: observación puntual a T + (h - 0.5)·W por horizonte (centro de
    #     la ventana objetivo) — "perfect-forecast proxy". El target del
    #     modelo es el delay en T+h, no el tiempo; la meteorología es
    #     exógena y NO viola la invariante de leakage.
    if weather_lookups is not None and not weather_lookups.empty():
        base_col += 5 * n_horizons  # saltamos el bloque G ya escrito
        hist_start = current_end - window_delta
        hist_end = current_end
        for airport, i in airport_map.items():
            agg = _weather_window_agg(weather_lookups, airport, hist_start, hist_end)
            features[i, base_col : base_col + WEATHER_BLOCK_HIST_SIZE] = (
                torch.from_numpy(agg)
            )

        fut_base = base_col + WEATHER_BLOCK_HIST_SIZE
        for h_idx, h in enumerate(prediction_horizons):
            h_center = current_end + (h - 1) * window_delta + window_delta / 2
            col_offset = fut_base + WEATHER_BLOCK_FUTURE_SIZE_PER_H * h_idx
            for airport, i in airport_map.items():
                obs = _weather_point_obs(weather_lookups, airport, h_center)
                features[i, col_offset : col_offset + WEATHER_BLOCK_FUTURE_SIZE_PER_H] = (
                    torch.from_numpy(obs)
                )

    # ── normalización final: clip + nan→0 ────────────────────────────
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def compute_node_targets(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
) -> torch.Tensor:
    """Genera targets continuos: retraso promedio de LLEGADA por aeropuerto.

    Calcula el retraso promedio (en minutos) con el que aterrizan los
    vuelos en cada aeropuerto destino dentro de la ventana temporal
    futura. ArrDelay es el target natural para una GNN de propagación: el
    retraso "fluye" por las aristas X→Y como `DepDelay_X` y aterriza como
    `ArrDelay_Y`. Predecir DepDelay (versión anterior) es en gran medida
    endógeno al aeropuerto y un modelo denso lo iguala.

    Args:
        window_df: DataFrame de la ventana FUTURA (target), filtrado por
            arr_timestamp (vuelos que LLEGAN en la ventana).
        airport_map: Mapeo de código IATA a índice.

    Returns:
        Tensor continuo [num_nodes] con retraso medio de llegada (min).
    """
    num_nodes = len(airport_map)
    targets = torch.zeros(num_nodes, dtype=torch.float32)

    if window_df.empty:
        return targets

    dest_groups = window_df.groupby("Dest")
    for airport, group in dest_groups:
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        targets[idx] = group["ArrDelay"].mean()

    return torch.nan_to_num(targets, nan=0.0)


# Índices de canal estables para el tensor de target multi-tarea.
# Todo el código (modelos, losses, métricas, reporting) referencia estos
# nombres para evitar números mágicos y mantener el contrato sincronizado.
TARGET_CHANNEL_ARR_DELAY = 0  # mean ArrDelay (min) — regresión primaria
TARGET_CHANNEL_DEP_DELAY = 1  # mean DepDelay (min) — regresión auxiliar
TARGET_CHANNEL_PCT_DELAYED = 2  # fracción ArrDelay≥threshold ∈ [0, 1] — BCE
NUM_TARGET_CHANNELS = 3


def compute_node_targets_multi_channel(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
    delay_threshold: float = 15.0,
) -> torch.Tensor:
    """Versión multi-canal de :func:`compute_node_targets`.

    Devuelve ``[num_nodes, NUM_TARGET_CHANNELS]`` con los tres canales
    documentados en ``TARGET_CHANNEL_*``:

      0) mean ArrDelay (min) por aeropuerto destino — target principal.
      1) mean DepDelay (min) por aeropuerto destino — regresión auxiliar
         (multi-task regularizer).
      2) fracción de vuelos con ``ArrDelay >= delay_threshold`` ∈ [0, 1]
         por aeropuerto destino — clasificación binaria a la que apunta
         el head BCE del modelo multi-tarea.

    Todos los canales se computan sobre la **misma** ventana (los vuelos
    que llegan a Dest=airport en ``[h_start, h_end)``), por lo que NaN
    en alguno se rellena con 0 igual que en la versión single-channel.

    Args:
        window_df: DataFrame de la ventana FUTURA (target), filtrado por
            ``arr_timestamp`` (vuelos que LLEGAN en la ventana).
        airport_map: Mapeo IATA → índice.
        delay_threshold: Minutos de ArrDelay sobre los que un vuelo se
            considera "retrasado" para el canal 2.

    Returns:
        Tensor ``[num_nodes, 3]`` (canales: arr_delay, dep_delay,
        pct_arr_delayed_15).
    """
    num_nodes = len(airport_map)
    targets = torch.zeros(num_nodes, NUM_TARGET_CHANNELS, dtype=torch.float32)

    if window_df.empty:
        return targets

    has_dep = "DepDelay" in window_df.columns
    for airport, group in window_df.groupby("Dest", observed=True):
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        arr = group["ArrDelay"].to_numpy(dtype="float32", na_value=np.nan)
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size > 0:
            targets[idx, TARGET_CHANNEL_ARR_DELAY] = float(np.mean(arr_valid))
            targets[idx, TARGET_CHANNEL_PCT_DELAYED] = float(
                np.mean(arr_valid >= delay_threshold)
            )
        if has_dep:
            dep = group["DepDelay"].to_numpy(dtype="float32", na_value=np.nan)
            dep_valid = dep[~np.isnan(dep)]
            if dep_valid.size > 0:
                targets[idx, TARGET_CHANNEL_DEP_DELAY] = float(np.mean(dep_valid))

    return torch.nan_to_num(targets, nan=0.0)


def compute_multi_horizon_targets(
    df: pd.DataFrame,
    airport_map: dict[str, int],
    current_end: pd.Timestamp,
    window_delta: pd.Timedelta,
    horizons: list[int],
    delay_threshold: float = 15.0,
) -> torch.Tensor | None:
    """Genera targets multi-horizonte multi-canal.

    Para cada horizonte h en ``horizons``, computa los 3 canales de
    target sobre los vuelos que LLEGAN en la ventana
    ``[current_end + (h-1)*delta, current_end + h*delta)``. Ver
    :func:`compute_node_targets_multi_channel` para la semántica de cada
    canal.

    Args:
        df: DataFrame completo con columna ``arr_timestamp`` (preferido)
            o ``timestamp`` (fallback).
        airport_map: Mapeo IATA → índice.
        current_end: Fin de la ventana actual de features.
        window_delta: Duración de cada ventana temporal.
        horizons: Lista de horizontes (e.g., ``[1, 2, 4, 6, 8]``).
        delay_threshold: Umbral en minutos para el canal de clasificación.

    Returns:
        Tensor ``[num_nodes, num_horizons, NUM_TARGET_CHANNELS]`` o
        ``None`` si algún horizonte no tiene >=10 vuelos (proxy de "datos
        suficientes" — heredado de la versión anterior).
    """
    num_nodes = len(airport_map)
    num_horizons = len(horizons)
    targets = torch.zeros(
        num_nodes, num_horizons, NUM_TARGET_CHANNELS, dtype=torch.float32,
    )

    # La columna `arr_timestamp` (creada en `create_temporal_graphs`) es la
    # clave correcta para la ventana del target: queremos los vuelos que
    # ATERRIZAN en [h_start, h_end), no los que despegan.
    arr_col = "arr_timestamp" if "arr_timestamp" in df.columns else "timestamp"

    for h_idx, h in enumerate(horizons):
        h_start = current_end + (h - 1) * window_delta
        h_end = current_end + h * window_delta
        h_mask = (df[arr_col] >= h_start) & (df[arr_col] < h_end)
        h_df = df[h_mask]

        if len(h_df) < 10:
            return None

        targets[:, h_idx, :] = compute_node_targets_multi_channel(
            h_df, airport_map, delay_threshold=delay_threshold,
        )

    return targets


def create_temporal_graphs(
    df: pd.DataFrame,
    airport_map: dict[str, int],
    edge_index: torch.Tensor,
    edge_attr_static: torch.Tensor,
    edge_pairs: list[tuple[str, str]],
    window_hours: int = 1,
    delay_threshold: float = 15.0,
    prediction_horizons: list[int] | None = None,
    weather_lookups: WeatherLookups | None = None,
) -> list[Data]:
    """Crea snapshots de grafos temporales a partir de datos de vuelos.

    Divide los datos en ventanas temporales y genera un grafo por ventana.
    Las features de cada nodo son las estadísticas de retraso del aeropuerto
    en esa ventana. El target es el retraso promedio en ventanas futuras.

    Cuando prediction_horizons es None o [1], genera targets single-horizon
    [num_nodes] (compatible con BasicGCN). Cuando tiene múltiples valores
    (e.g., [1, 2, 3, 4, 5]), genera targets multi-horizonte [num_nodes,
    num_horizons] para MultiHorizonGAT.

    Args:
        df: DataFrame preprocesado con FlightDate y Hour.
        airport_map: Mapeo de código IATA a índice.
        edge_index: Matriz de adyacencia [2, num_edges].
        edge_attr_static: Features estáticas de arista [num_edges, 3].
        edge_pairs: Lista de ``(origin_iata, dest_iata)`` por arista.
        window_hours: Horas por ventana temporal.
        delay_threshold: Umbral de retraso en minutos.
        prediction_horizons: Lista de horizontes futuros (e.g., [1,2,3,4,5]).
            None o [1] usa modo single-horizon (retrocompatible).

    Returns:
        Lista de objetos Data de PyG (un grafo por ventana).
    """
    multi_horizon = (
        prediction_horizons is not None and len(prediction_horizons) > 1
    )

    # Crear timestamp combinando FlightDate + Hour
    df = df.copy()
    if "Hour" not in df.columns:
        df["Hour"] = (df["CRSDepTime"] // 100).clip(0, 23)

    df["timestamp"] = pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(
        df["Hour"], unit="h"
    )

    # Crear arr_timestamp basado en CRSArrTime (la hora *programada* de
    # llegada). Es la clave correcta para la ventana del target: queremos
    # los vuelos que LLEGAN al destino en [h_start, h_end), no los que
    # despegan. Si CRSArrTime < CRSDepTime suponemos vuelo nocturno y
    # sumamos un día (los vuelos que cruzan la frontera del día se
    # detectan correctamente para horarios típicos en EE.UU.).
    if "CRSArrTime" in df.columns:
        arr_hour = (df["CRSArrTime"] // 100).clip(0, 23).astype("Int16")
        crosses_midnight = (df["CRSArrTime"] < df["CRSDepTime"]).fillna(False)
        df["arr_timestamp"] = (
            pd.to_datetime(df["FlightDate"])
            + pd.to_timedelta(arr_hour.astype("Int64"), unit="h")
            + pd.to_timedelta(crosses_midnight.astype(int), unit="D")
        )
    else:
        # Fallback: si no se cargó CRSArrTime, usar la hora de salida como
        # proxy. Mantiene retrocompatibilidad con tests/datos antiguos pero
        # introduce un sesgo (subestima cuántos vuelos llegan en la ventana).
        df["arr_timestamp"] = df["timestamp"]

    # Ordenar por timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Crear ventanas temporales
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    window_delta = pd.Timedelta(hours=window_hours)

    # Calcular cuántas ventanas futuras necesitamos
    horizons_for_features = prediction_horizons if multi_horizon else [1]
    if multi_horizon:
        max_horizon = max(prediction_horizons)
        lookahead = max_horizon * window_delta
    else:
        lookahead = window_delta

    # Precomputar lookups históricos y exógenos. Se calculan una sola vez
    # sobre el dataset entero — cada snapshot consulta valores en O(N·H).
    history_lookups = _build_history_lookups(df, airport_map, window_delta)

    graphs = []
    current_time = min_time

    while current_time + window_delta + lookahead <= max_time:
        # Ventana actual (features)
        window_mask = (
            (df["timestamp"] >= current_time)
            & (df["timestamp"] < current_time + window_delta)
        )
        window_df = df[window_mask]

        if len(window_df) < 10:
            current_time += window_delta
            continue

        current_end = current_time + window_delta

        if multi_horizon:
            # Targets multi-horizonte multi-canal: [num_nodes, num_horizons, 3]
            # (arr_delay, dep_delay, pct_arr_delayed_15). El canal de
            # clasificación necesita el umbral, así que se propaga al builder.
            node_targets = compute_multi_horizon_targets(
                df, airport_map, current_end, window_delta,
                prediction_horizons,
                delay_threshold=delay_threshold,
            )
            if node_targets is None:
                current_time += window_delta
                continue
        else:
            # Targets single-horizon: [num_nodes] (retrocompatible).
            # Usar arr_timestamp por la misma razón que el caso multi-horizon.
            arr_col = "arr_timestamp" if "arr_timestamp" in df.columns else "timestamp"
            next_mask = (
                (df[arr_col] >= current_end)
                & (df[arr_col] < current_end + window_delta)
            )
            next_df = df[next_mask]

            if len(next_df) < 10:
                current_time += window_delta
                continue

            node_targets = compute_node_targets(next_df, airport_map)

        node_features = compute_node_features_rich(
            window_df,
            airport_map,
            current_end=current_end,
            window_delta=window_delta,
            prediction_horizons=horizons_for_features,
            history_lookups=history_lookups,
            delay_threshold=delay_threshold,
            weather_lookups=weather_lookups,
        )

        # Máscara de nodos con actividad (para ignorar aeropuertos sin datos)
        active_mask = node_features.abs().sum(dim=1) > 0

        # Concatenar features estáticas (3) + dinámicas (2) → [E, 5].
        edge_attr_dynamic = compute_dynamic_edge_attr(
            edge_pairs, history_lookups, current_end, window_delta,
        )
        edge_attr = torch.cat([edge_attr_static, edge_attr_dynamic], dim=1)

        # `timestamp` (ISO string) marca el inicio de la ventana de target
        # — se usa para el split cronológico por fecha en
        # `split_graphs_temporal`. Como string es compatible con el batching
        # de PyG (no se concatena, solo se preserva por grafo).
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_targets,
            active_mask=active_mask,
            timestamp=current_end.isoformat(),
        )
        graphs.append(graph)

        current_time += window_delta

    horizons_str = str(prediction_horizons) if multi_horizon else "[1]"
    logger.info(
        "Snapshots temporales creados: %d grafos (ventana=%dh, "
        "horizontes=%s, umbral=%d min)",
        len(graphs), window_hours, horizons_str, delay_threshold,
    )

    return graphs


def normalize_graph_features(
    graphs: list[Data],
    train_indices: list[int] | None = None,
) -> tuple[list[Data], dict[str, torch.Tensor]]:
    """Normaliza las node features de los grafos (zero-mean, unit-variance).

    Calcula media y desviación estándar solo sobre los nodos activos
    de los grafos de entrenamiento, y aplica la normalización a todos
    los grafos. Esto evita fuga de información del conjunto de test.

    Args:
        graphs: Lista completa de grafos PyG.
        train_indices: Índices de los grafos de entrenamiento para
            calcular estadísticas. Si None, usa todos los grafos.

    Returns:
        Tupla de (grafos normalizados, estadísticas {mean, std}).
    """
    if not graphs:
        return graphs, {"mean": torch.zeros(1), "std": torch.ones(1)}

    # Recopilar features de nodos activos para calcular estadísticas
    train_idxs = train_indices if train_indices is not None else range(len(graphs))
    all_features = []
    for i in train_idxs:
        g = graphs[i]
        mask = g.active_mask
        if mask.sum() > 0:
            all_features.append(g.x[mask])

    if not all_features:
        return graphs, {"mean": torch.zeros(1), "std": torch.ones(1)}

    stacked = torch.cat(all_features, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    # Evitar división por cero
    std = torch.clamp(std, min=1e-6)

    # Aplicar normalización a todos los grafos
    normalized = []
    for g in graphs:
        g_new = g.clone()
        g_new.x = (g.x - mean) / std
        normalized.append(g_new)

    logger.info(
        "Features normalizadas: media=%s, std=%s",
        mean.numpy().round(2), std.numpy().round(2),
    )

    return normalized, {"mean": mean, "std": std}


# Versión del esquema de features de grafo. Cualquier cambio en el contrato
# de columnas de x/edge_attr/y debe incrementar este valor para invalidar
# cachés en disco que asuman el esquema anterior.
#   v2 → v3 (W2 multi-task): los targets multi-horizonte pasan de
#         ``[N, H]`` a ``[N, H, 3]`` (canales arr_delay/dep_delay/pct_15).
#   v3 → v4 (Weather): se añade el bloque H opcional con 9 + 9·H features
#         meteorológicas (ver ``WEATHER_BLOCK_*``). Con ``weather.enabled
#         = false`` el feature dim NO cambia, pero el hash de cache incluye
#         el flag para que distintos toggles no se pisen.
GRAPH_SCHEMA_VERSION = 4


# Mapeo de grupos legibles (en config.weather.params) a las claves
# internas que reconoce ``WeatherLookups.enabled_params``. Mantener
# sincronizado con el bloque ``weather:`` de configs/default.yaml.
_WEATHER_PARAM_ALIASES = {
    "wind": "wind",
    "wind_gust": "wind",          # alias retro-compatible
    "precip": "precip_cloud",
    "precip_cloud": "precip_cloud",
    "cloud": "precip_cloud",
    "category": "category",
    "weather_code": "category",
}


def _load_weather_lookups_from_config(
    df: pd.DataFrame,
    airports: list[str],
    config: dict,
) -> WeatherLookups | None:
    """Lee config.weather y construye ``WeatherLookups`` si está activo.

    Pasos:
      1. Si ``weather.enabled`` es ``False`` (default) → ``None``.
      2. Resuelve el rango temporal de ``df`` (fechas mínima y máxima de
         ``FlightDate``). Open-Meteo es inclusivo en ``end_date`` → no
         hace falta sumar un día.
      3. Llama a ``load_weather_for_airports`` con el cache_dir
         configurado (``weather.cache_dir`` o default
         ``data/processed/weather``).
      4. Construye ``WeatherLookups`` con los grupos de params activos.
    """
    weather_config = (config.get("weather") or {})
    if not weather_config.get("enabled", False):
        return None

    flight_dates = pd.to_datetime(df["FlightDate"])
    start_date = flight_dates.min().date()
    # Ampliamos un día por seguridad: las observaciones a T+h del último
    # snapshot pueden caer en el día posterior si current_end es noche.
    end_date = flight_dates.max().date() + dt.timedelta(days=1)

    cache_dir_raw = weather_config.get("cache_dir") or "data/processed/weather"
    cache_dir = Path(cache_dir_raw)

    raw_params = weather_config.get("params") or [
        "wind", "precip_cloud", "category",
    ]
    enabled_params = frozenset(
        _WEATHER_PARAM_ALIASES.get(p, p) for p in raw_params
    )

    logger.info(
        "Meteorología activada (provider=%s, params=%s, %s..%s).",
        weather_config.get("provider", "open_meteo"),
        sorted(enabled_params), start_date, end_date,
    )

    weather_df = load_weather_for_airports(
        airports, start_date, end_date, cache_dir=cache_dir,
    )

    return _build_weather_lookups(
        weather_df, airports=airports, enabled_params=enabled_params,
    )


def _snapshot_cache_key(
    df: pd.DataFrame,
    airports: list[str],
    config: dict,
) -> str:
    """Hash estable que identifica un build de snapshots.

    Combina la versión del esquema, el subset de claves de config que
    afectan al output, los aeropuertos y un resumen del DataFrame
    (rango temporal, número de filas, columnas presentes). Si cualquiera
    de estos elementos cambia, el hash cambia y el caché se invalida.
    """
    graph_config = config.get("graph", {}) or {}
    eval_config = config.get("evaluation", {}) or {}
    weather_config = config.get("weather", {}) or {}

    relevant_graph = {
        k: graph_config.get(k)
        for k in (
            "min_route_flights",
            "temporal_window_hours",
            "prediction_horizons",
            "normalize_features",
        )
    }
    relevant_eval = {"delay_threshold_minutes": eval_config.get("delay_threshold_minutes")}
    # Subset estable de config.weather que afecta al tensor de features.
    # ``provider`` se incluye para que un futuro switch a ASOS no reuse
    # un caché generado con Open-Meteo.
    relevant_weather = {
        "enabled": bool(weather_config.get("enabled", False)),
        "provider": weather_config.get("provider", "open_meteo"),
        "params": sorted(weather_config.get("params", []) or []),
        "schema": WEATHER_SCHEMA_VERSION,
    }

    flight_dates = pd.to_datetime(df["FlightDate"])
    df_signature = {
        "rows": int(len(df)),
        "cols": sorted(map(str, df.columns)),
        "date_min": str(flight_dates.min().date()),
        "date_max": str(flight_dates.max().date()),
    }

    payload = {
        "schema": GRAPH_SCHEMA_VERSION,
        "graph": relevant_graph,
        "eval": relevant_eval,
        "weather": relevant_weather,
        "airports": sorted(airports),
        "df": df_signature,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_graph_dataset(
    df: pd.DataFrame,
    airports: list[str],
    config: dict,
) -> tuple[list[Data], dict[str, int]]:
    """Pipeline completo: de DataFrame a lista de grafos PyG.

    Si ``config['graph']['cache_dir']`` apunta a un directorio, se usa
    como caché de snapshots: el primer build escribe ``snapshots_<hash>.pt``
    y los siguientes se cargan desde disco si la firma de entradas
    (esquema + config + dataset + aeropuertos) no ha cambiado.

    Args:
        df: DataFrame preprocesado (con FlightDate, Hour, Origin, Dest, etc.).
        airports: Lista de códigos de aeropuerto filtrados.
        config: Configuración con secciones 'graph' y 'evaluation'.

    Returns:
        Tupla de (lista de grafos Data, mapeo de aeropuertos).
    """
    graph_config = config.get("graph", {})
    eval_config = config.get("evaluation", {})

    cache_dir = graph_config.get("cache_dir")
    cache_path: Path | None = None
    if cache_dir:
        cache_key = _snapshot_cache_key(df, airports, config)
        cache_path = Path(cache_dir) / f"snapshots_{cache_key}.pt"
        if cache_path.exists():
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                "Caché HIT: cargando snapshots desde %s (%.1f MB)",
                cache_path, size_mb,
            )
            payload = torch.load(cache_path, weights_only=False)
            return payload["graphs"], payload["airport_map"]
        logger.info(
            "Caché MISS: no existe %s. Generando snapshots y escribiendo "
            "caché tras el build.",
            cache_path,
        )
    else:
        logger.warning(
            "Caché de snapshots DESACTIVADO (graph.cache_dir no está en el "
            "config). Cada ejecución regenerará snapshots desde cero. Para "
            "activarlo añade `graph.cache_dir: data/processed/snapshots` "
            "al YAML."
        )

    airport_map = build_airport_mapping(airports)

    min_flights = graph_config.get("min_route_flights", 50)
    edge_index, edge_attr_static, edge_pairs = build_edge_index(
        df, airport_map, min_flights,
    )

    window_hours = graph_config.get("temporal_window_hours", 1)
    delay_threshold = eval_config.get("delay_threshold_minutes", 15)

    prediction_horizons = graph_config.get("prediction_horizons")

    # ── Meteorología (bloque H opcional) ─────────────────────────────
    # Solo se activa si ``config.weather.enabled = true``. Si la red
    # falla, ``load_weather_for_airports`` devuelve DataFrame vacío y
    # ``_build_weather_lookups`` traduce a ``WeatherLookups`` vacío →
    # el feature block H se llena con 0 sin romper el entrenamiento.
    weather_lookups = _load_weather_lookups_from_config(
        df, airports, config,
    )

    graphs = create_temporal_graphs(
        df=df,
        airport_map=airport_map,
        edge_index=edge_index,
        edge_attr_static=edge_attr_static,
        edge_pairs=edge_pairs,
        window_hours=window_hours,
        delay_threshold=delay_threshold,
        prediction_horizons=prediction_horizons,
        weather_lookups=weather_lookups,
    )

    # Normalizar features usando solo estadísticas de entrenamiento
    normalize = graph_config.get("normalize_features", False)
    if normalize and graphs:
        n = len(graphs)
        train_end = int(n * 0.7)
        train_indices = list(range(train_end))
        graphs, _ = normalize_graph_features(graphs, train_indices)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"graphs": graphs, "airport_map": airport_map, "schema": GRAPH_SCHEMA_VERSION},
            cache_path,
        )
        logger.info("Snapshots cacheados en: %s", cache_path)

    return graphs, airport_map


def split_graphs_temporal(
    graphs: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    train_end: str | None = None,
    val_end: str | None = None,
) -> dict[str, list[Data]]:
    """Divide los grafos temporales en train/val/test.

    Soporta dos modos:
    - **Por fecha** (preferido): si se pasan ``train_end`` y ``val_end``
      como strings ISO (e.g. ``"2019-07-01"``), los grafos se agrupan por
      su atributo ``timestamp``: ``< train_end`` → train; entre
      ``train_end`` y ``val_end`` → val; ``≥ val_end`` → test. Funciona
      correctamente con datasets que cubren varios años (cosa que el modo
      por mes no hacía: jan-2018 y jan-2019 caían en el mismo bucket).
    - **Por ratio** (fallback / retrocompatibilidad): si las fechas no
      están dadas, se mantiene el corte secuencial por proporción.

    Args:
        graphs: Lista de grafos PyG ordenados temporalmente.
        train_ratio: Proporción de grafos para entrenamiento (modo ratio).
        val_ratio: Proporción de grafos para validación (modo ratio).
        train_end: Fecha ISO ``YYYY-MM-DD`` que cierra el split de train.
        val_end: Fecha ISO ``YYYY-MM-DD`` que cierra el split de val.

    Returns:
        Diccionario con claves 'train', 'val', 'test'.
    """
    if train_end is not None and val_end is not None:
        # Modo por fecha. Comparación lexicográfica sobre ISO strings es
        # correcta porque el formato es ordenable.
        train_split = [g for g in graphs if g.timestamp < train_end]
        val_split = [g for g in graphs if train_end <= g.timestamp < val_end]
        test_split = [g for g in graphs if g.timestamp >= val_end]
        splits = {"train": train_split, "val": val_split, "test": test_split}
        logger.info(
            "Split por fecha: train < %s, val < %s, test ≥ %s",
            train_end, val_end, val_end,
        )
    else:
        n = len(graphs)
        cut_train = int(n * train_ratio)
        cut_val = int(n * (train_ratio + val_ratio))
        splits = {
            "train": graphs[:cut_train],
            "val": graphs[cut_train:cut_val],
            "test": graphs[cut_val:],
        }
        logger.info(
            "Split por ratio: train=%.0f%%, val=%.0f%%, test=%.0f%%",
            train_ratio * 100, val_ratio * 100,
            (1 - train_ratio - val_ratio) * 100,
        )

    for name, split in splits.items():
        logger.info("Graph split %s: %d grafos", name, len(split))

    return splits


def create_temporal_sequences(
    graphs: list[Data],
    input_window: int = 6,
) -> list[list[Data]]:
    """Agrupa grafos consecutivos en secuencias temporales para modelos LSTM.

    Crea ventanas deslizantes de tamaño ``input_window`` con paso 1 sobre
    una lista de grafos ya pertenecientes a un mismo split (train, val o
    test). El target de cada secuencia es ``sequence[-1].y`` (targets del
    último grafo).

    Debe llamarse DESPUÉS de ``split_graphs_temporal()`` sobre cada split
    por separado para evitar fuga de información entre conjuntos.

    Args:
        graphs: Lista de grafos PyG de un solo split, ordenados
            temporalmente.
        input_window: Número de snapshots consecutivos por secuencia.

    Returns:
        Lista de secuencias, donde cada secuencia es una lista de
        ``input_window`` objetos Data. Vacía si hay menos grafos que
        ``input_window``.
    """
    if len(graphs) < input_window:
        logger.warning(
            "Solo %d grafos disponibles, se necesitan %d para formar "
            "secuencias. Retornando lista vacía.",
            len(graphs), input_window,
        )
        return []

    sequences = [
        graphs[i : i + input_window]
        for i in range(len(graphs) - input_window + 1)
    ]

    logger.info(
        "Secuencias temporales creadas: %d secuencias (ventana=%d grafos)",
        len(sequences), input_window,
    )

    return sequences
