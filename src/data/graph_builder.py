"""Construction of airport graphs for GNN models.

Transforms tabular flight data into graphs where nodes are airports and
edges are air routes. Generates temporal snapshots with per-airport
aggregated features and continuous delay targets (average delay in minutes
over the next time window).

TEMPORAL LEAKAGE INVARIANT (no leakage)
----------------------------------------
In the snapshot with `current_end = T`, the features and edges of each node
may only depend on:

  * Any flight with `arr_timestamp < T` (completed before T).
  * The schedule columns (`CRS*`, `Distance`, `Airline`, `Origin`, `Dest`)
    of any flight — they are known a priori and used both for historical
    features and for exogenous features of the target.

They may NOT depend on Class B columns (post-hoc: `DepTime`, `ArrTime`,
`WheelsOff/On`, `TaxiOut/In`, `ActualElapsedTime`, `AirTime`, `DepDelay`,
`ArrDelay`, `DepDel15`, `ArrDel15`, `CarrierDelay`, `WeatherDelay`,
`NASDelay`, `SecurityDelay`, `LateAircraftDelay`, `Cancelled`, `Diverted`)
of flights whose `arr_timestamp >= T`. The rich features precomputed in
`_build_history_lookups` group explicitly by arrival hour (`arr_timestamp`)
so the historical window closes strictly before T.
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

__all__ = [
    "build_graph_dataset",
    "create_temporal_graphs",
    "create_temporal_sequences",
    "split_graphs_temporal",
    "build_airport_mapping",
    "build_edge_index",
    "normalize_graph_features",
    "ARR_DELAY_LAGS",
    "GRAPH_SCHEMA_VERSION",
]


# BTS causes that come in the Combined_Flights dataset when ArrDelay≥15.
# If a column is not loaded (old Parquet), it is omitted from the
# corresponding feature.
BTS_CAUSE_COLUMNS = (
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
)

# Lags (in steps of `window_delta`) used as historical features.
ARR_DELAY_LAGS = (1, 3, 6, 24)
ROLLING_WINDOW_HOURS = 6

# Size of block H (weather) when ``weather_lookups`` is enabled. Historical
# = mean wind + max gust + sum precip + mean cloud + 5 one-hots of the
# dominant category; future identical per horizon. Keeping them equal makes
# the total count ``9 + 9 * H``.
WEATHER_BLOCK_HIST_SIZE = 4 + NUM_WEATHER_CATEGORIES  # = 9
WEATHER_BLOCK_FUTURE_SIZE_PER_H = 4 + NUM_WEATHER_CATEGORIES  # = 9


@dataclass
class HistoryLookups:
    """Precomputed structures for per-snapshot feature engineering.

    Built once at the start of `create_temporal_graphs`, they let each call
    to `compute_node_features_rich` and `compute_dynamic_edge_attr` be
    O(num_airports × num_horizons) or O(num_edges) without revisiting the
    entire DataFrame.

    Node-indexed series use `(airport, hour_bucket)`; edge-indexed ones use
    `(origin, dest, hour_bucket)`. In all cases `hour_bucket = pd.Timestamp`
    aligned to the start of the window.
    """

    arr_delay_by_airport_hour: pd.Series  # mean ArrDelay (Class B)
    sched_arr_count: pd.Series  # # flights scheduled to arrive (Class A)
    sched_dep_count: pd.Series  # # flights scheduled to depart (Class A)
    sched_arr_from_top10: pd.Series  # # of top-10 origins → this airport
    sched_arr_mean_distance: pd.Series  # mean scheduled inbound distance
    rolling_mean_arr_delay: pd.Series  # rolling 6h of ArrDelay (Class B)
    rolling_std_arr_delay: pd.Series  # rolling 6h std (Class B)
    top10_origins: list[str]  # IATAs of the 10 most connected airports
    # Per edge (Origin, Dest, bucket):
    route_recent_arr_delay: pd.Series  # rolling 6h of ArrDelay per route (Class B)
    route_sched_count: pd.Series  # scheduled flights per route+bucket (Class A)


@dataclass
class WeatherLookups:
    """Precomputed structure for O(1) access to hourly observations.

    ``airport_weather`` maps IATA → DataFrame indexed by ``timestamp``
    (hourly, rounded to the start of the hour) with columns:
    ``wind_speed_10m``, ``wind_gusts_10m``, ``precipitation``,
    ``cloud_cover``, ``weather_category``. Airports without data (whether
    from a network error or because their IATA is not in the coordinates
    CSV) do NOT appear in the dict — consumers return the zero vector when
    the lookup fails, which the model interprets as "no weather signal"
    (equivalent to the ``weather_lookups is None`` branch for that specific
    airport).

    Attributes:
        airport_weather: Dict IATA → hourly DataFrame.
        enabled_params: Set of active keys {"wind", "precip_cloud",
            "category"}. Lets individual groups be toggled on/off without
            rebuilding the HTTP cache — disabled groups are overwritten with
            0 in ``compute_node_features_rich``. This directly supports the
            toggles in docs/ablation-log.md.
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
    """Aggregate hourly observations in ``[start, end)`` for an IATA.

    Returns a dense vector of ``WEATHER_BLOCK_HIST_SIZE`` floats:

      [0]      mean(wind_speed_10m) in km/h
      [1]      max(wind_gusts_10m) in km/h
      [2]      sum(precipitation) in mm
      [3]      mean(cloud_cover) in %
      [4..8]   one-hot of the DOMINANT category (mode) in the window

    Columns corresponding to disabled groups (``enabled_params``) are set to
    0 — feature-group-level toggle for ablations.
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
            # Mode: argmax over the histogram. ``np.bincount`` with
            # ``minlength`` guarantees a slot per category even if the
            # window does not contain all of them.
            hist = np.bincount(cats, minlength=NUM_WEATHER_CATEGORIES)
            dominant = int(np.argmax(hist))
            out[4 + dominant] = 1.0

    return out


def _weather_point_obs(
    weather: WeatherLookups,
    iata: str,
    timestamp: pd.Timestamp,
) -> np.ndarray:
    """Hourly lookup aligned to the hour of ``timestamp``.

    Returns a dense vector of ``WEATHER_BLOCK_FUTURE_SIZE_PER_H`` floats
    with the same schema as :func:`_weather_window_agg` except that columns
    0..3 are the point observation (not aggregated) and 4..8 is the one-hot
    of the category observed at that exact hour.

    Leakage rationale: the target at T+h is the observed delay, not the
    weather; the feature is exogenous. We use the observation at T+h as a
    "perfect-forecast proxy" — it overestimates the real usefulness (a
    production system would use a forecast with error), but that gap is
    addressed in a future ablation (docs/ablation-log.md).
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
        # A lookup that matched several timestamps (should not happen with
        # floor("h") but defensively we keep the first).
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
    """Build ``WeatherLookups`` from the multi-airport DataFrame.

    If ``weather_df`` is ``None`` or empty, returns ``None`` (which disables
    block H downstream, keeping the previous feature-dim contract).

    The input DataFrame has MultiIndex ``(iata, timestamp)`` — the output is
    a dict ``{iata: df_indexed_by_timestamp}`` with ``timestamp`` rounded to
    the hour (``floor("h")``) so the ``_weather_point_obs`` lookup closes on
    the exact hour. Hashable keys enable O(1) per airport.
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
        # Remove duplicates keeping the first value per hour.
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
    """Precompute the per-(airport, hour_bucket) series used as lookups in
    feature engineering.

    Class B features (which depend on actuals) are aggregated by *arrival*
    hour (`arr_timestamp`) — so, when querying `lag=k` from
    `current_end=T`, we gather flights that landed in
    [T - k·delta, T - (k-1)·delta), which are strictly before T.

    Class A features (schedule) are aggregated by the *scheduled* arrival or
    departure hour as appropriate — they are valid for both past and future
    windows.

    Args:
        df: Full DataFrame with `arr_timestamp` and `timestamp` already
            computed and sorted chronologically.
        airport_map: IATA → index map.
        window_delta: Temporal window size (e.g. 1h or 2h).

    Returns:
        ``HistoryLookups`` with all precomputed series.
    """
    valid_airports = set(airport_map.keys())

    # Normalize to buckets aligned to the start of the window.
    # `pd.Timestamp.floor("Nh")` rounds down to a multiple of the window
    # size.
    freq = f"{int(window_delta.total_seconds() // 3600)}h"

    df_arr = df.copy()
    df_arr["arr_bucket"] = df_arr["arr_timestamp"].dt.floor(freq)
    df_arr["dep_bucket"] = df_arr["timestamp"].dt.floor(freq)

    # ── Class B: ArrDelay grouped by (Dest, arr_bucket) ───────────────
    arr_mask = df_arr["Dest"].isin(valid_airports) & df_arr["ArrDelay"].notna()
    arr_grp = df_arr[arr_mask].groupby(["Dest", "arr_bucket"], observed=True)
    arr_delay_by_airport_hour = arr_grp["ArrDelay"].mean().rename("arr_delay_mean")

    # Rolling 6h per airport (over the hourly series) — serves as the
    # "mean recent behavior" at any T.
    rolling_steps = max(
        1, ROLLING_WINDOW_HOURS // max(1, int(window_delta.total_seconds() // 3600))
    )
    rolling_mean = (
        arr_delay_by_airport_hour.groupby(level=0, observed=True)
        .rolling(window=rolling_steps, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # For std we would need a series with every flight's observations (not
    # just the per-bucket mean); we approximate with the std of the hourly
    # means in the window, enough as a volatility signal. If
    # ``rolling_steps == 1`` (degenerate case where the window covers
    # exactly ROLLING_WINDOW_HOURS or more), pandas rejects
    # ``min_periods=2 > window=1``. In that case the std is undefined with a
    # single observation; we clamp ``min_periods`` so the computation
    # returns NaN (later filled with 0.0). In all production configs
    # (``temporal_window_hours`` ∈ {1, 2}) the clamp does not trigger and
    # the loss is strictly the same.
    rolling_std = (
        arr_delay_by_airport_hour.groupby(level=0, observed=True)
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
        # Empty series with the same MultiIndex as the "with Distance"
        # branch (``["Dest", "arr_bucket"]``) → all lookups return the
        # default via KeyError without tripping on ``IndexingError`` from a
        # single dimension. In production the loader always includes
        # ``Distance`` (cf. ``src/data/loader.py``), so this branch only
        # pins the behavior of the synthetic tests without ``Distance``.
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

    # ── Top-10 origins (structural, computed once) ───────────────────
    top10_origins = (
        df_arr[df_arr["Origin"].isin(valid_airports)]
        .groupby("Origin", observed=True)
        .size()
        .nlargest(10)
        .index.tolist()
    )
    top10_set = set(top10_origins)
    is_from_top10 = df_arr["Origin"].isin(top10_set) & df_arr["Dest"].isin(
        valid_airports
    )
    sched_arr_from_top10 = (
        df_arr[is_from_top10]
        .groupby(["Dest", "arr_bucket"], observed=True)
        .size()
        .rename("sched_arr_from_top10")
        .astype("float32")
    )

    # ── Per route: recent ArrDelay (rolling 6h) ──────────────────────
    # Class B: grouped by (Origin, Dest, arr_bucket) → mean ArrDelay, then
    # rolling over each route's temporal dimension.
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
        route_arr_delay_hourly.groupby(level=[0, 1], observed=True)
        .rolling(window=rolling_steps, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # ── Per route: schedule (Class A, dep_bucket) ────────────────────
    route_sched_mask = df_arr["Origin"].isin(valid_airports) & df_arr["Dest"].isin(
        valid_airports
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
    """sin/cos encoding of a cyclic variable."""
    angle = 2.0 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def build_airport_mapping(airports: list[str]) -> dict[str, int]:
    """Create a mapping from IATA code to numeric index.

    Args:
        airports: Sorted list of airport codes.

    Returns:
        Dictionary {IATA_code: index}.
    """
    return {code: idx for idx, code in enumerate(sorted(airports))}


def build_edge_index(
    df: pd.DataFrame,
    airport_map: dict[str, int],
    min_flights: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]]:
    """Build the adjacency matrix and the static edge features.

    Creates bidirectional edges between airports that have at least
    `min_flights` flights in the dataset and returns a tensor with 3 static
    features per edge (Class A: derivable from the schedule):

      Col 0  flight_count_norm        — normalized historical count
      Col 1  mean_air_time_norm       — mean historical AirTime
      Col 2  mean_scheduled_distance  — mean Distance (Class A)

    These three features do not change between snapshots; the remaining two
    (recent_route_delay, scheduled_flights_next_h_norm) are concatenated per
    snapshot in `compute_dynamic_edge_attr` to reach a final `edge_attr` of
    shape ``[num_edges, 5]``.

    Args:
        df: DataFrame with Origin, Dest and optionally AirTime / Distance
            columns.
        airport_map: IATA code → index map.
        min_flights: Minimum number of flights to create an edge.

    Returns:
        Tuple of:
          * ``edge_index`` ``[2, num_edges]``
          * ``static_edge_attr`` ``[num_edges, 3]``
          * ``edge_pairs`` list of ``(origin_iata, dest_iata)`` per edge in
            order — used to build dynamic features via lookup in
            ``HistoryLookups``.
    """
    # Aggregations per directional route.
    agg_dict: dict = {"count": ("Origin", "size")}
    if "AirTime" in df.columns:
        agg_dict["mean_air_time"] = ("AirTime", "mean")
    if "Distance" in df.columns:
        agg_dict["mean_distance"] = ("Distance", "mean")

    route_stats = (
        df.groupby(["Origin", "Dest"], observed=True).agg(**agg_dict).reset_index()
    )

    # Filter routes with few flights and unmapped airports.
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
        at = (
            float(row["mean_air_time"])
            if has_air_time and not pd.isna(row["mean_air_time"])
            else 0.0
        )
        di = (
            float(row["mean_distance"])
            if has_distance and not pd.isna(row["mean_distance"])
            else 0.0
        )
        # Bidirectional edge with the same statistics (routes X→Y and Y→X
        # share distance and mean air time; the directional delays live in
        # `route_recent_arr_delay`).
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
        "Graph built: %d nodes, %d edges (min_flights=%d), static edge_attr shape=%s",
        len(airport_map),
        edge_index.shape[1],
        min_flights,
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
    """Compute the 2 dynamic edge features for a snapshot.

    Returns ``[num_edges, 2]`` with:

      Col 0  recent_route_delay_norm        — rolling 6h ArrDelay per route,
              normalized (clip ±1) — bucket immediately before
              ``current_end`` (Class B).
      Col 1  scheduled_flights_next_h_norm  — normalized count of flights
              scheduled on the route for the first target window (Class A).

    The normalizers are constant to keep the scale stable between snapshots;
    the later ``normalize_features=true`` adjusts the final range.
    """
    bucket_size = f"{int(window_delta.total_seconds() // 3600)}h"
    history_bucket = (current_end - window_delta).floor(bucket_size)
    target_bucket = current_end.floor(bucket_size)

    n_edges = len(edge_pairs)
    out = torch.zeros(n_edges, 2, dtype=torch.float32)

    recent = history_lookups.route_recent_arr_delay
    sched = history_lookups.route_sched_count

    for i, (origin, dest) in enumerate(edge_pairs):
        # Recent route delay: lookup (origin, dest, history_bucket) in the
        # pre-rolling series. Default 0 if the route had no flights.
        try:
            v = recent.loc[(origin, dest, history_bucket)]
            if not pd.isna(v):
                out[i, 0] = float(v) / arr_delay_norm
        except KeyError:
            pass

        # Scheduled flights in the next target window.
        try:
            c = sched.loc[(origin, dest, target_bucket)]
            if not pd.isna(c):
                out[i, 1] = float(c) / sched_count_norm
        except KeyError:
            pass

    # Clip to a stable range so the outlier does not dominate.
    out[:, 0].clamp_(-1.0, 1.0)
    out[:, 1].clamp_(0.0, 1.0)
    return out


def compute_node_features(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
    delay_threshold: float = 15.0,
) -> torch.Tensor:
    """Compute per-airport features for a time window.

    Per-node features (airport as origin):
    - avg_dep_delay: Average departure delay
    - std_dep_delay: Standard deviation of the delay
    - pct_delayed: Percentage of flights with delay > threshold
    - num_flights: Number of flights (normalized)
    - avg_arr_delay_incoming: Average arrival delay (as destination)
    - std_arr_delay_incoming: Standard deviation of the arrival delay

    Args:
        window_df: DataFrame filtered to a time window.
        airport_map: IATA code → index map.
        delay_threshold: Minutes threshold to consider a flight delayed.

    Returns:
        Node features tensor [num_nodes, num_features].
    """
    num_nodes = len(airport_map)
    num_features = 6
    features = torch.zeros(num_nodes, num_features, dtype=torch.float32)

    if window_df.empty:
        return features

    # Features as ORIGIN airport
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

    # Features as DESTINATION airport (arrival delays)
    dest_groups = window_df.groupby("Dest")
    for airport, group in dest_groups:
        if airport not in airport_map:
            continue
        idx = airport_map[airport]
        arr_delays = group["ArrDelay"].values
        features[idx, 4] = np.nanmean(arr_delays)
        features[idx, 5] = np.nanstd(arr_delays) if len(arr_delays) > 1 else 0.0

    # Normalize num_flights
    max_flights = features[:, 3].max()
    if max_flights > 0:
        features[:, 3] = features[:, 3] / max_flights

    # Replace NaN with 0
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
    """Enriched version of ``compute_node_features``.

    Returns a dense per-node vector organized into fixed-size blocks (the
    total dimension depends on ``len(prediction_horizons)`` and on whether
    weather is active):

      Block                                       | size
      --------------------------------------------|-------
      A) Aggregates current window (Class B)      | 9
      B) Volume / disruption (Class A+B)          | 4
      C) BTS causes (Class B, mean per cause)     | 5
      D) ArrDelay lags (Class B, via lookup)      | len(ARR_DELAY_LAGS)
      E) Rolling 6h mean+std (Class B, lookup)    | 2
      F) Cyclic calendar of the current window    | 6
      G) Future exogenous per horizon (Class A)   | 5 × len(horizons)
      H) Weather (optional)                       | 9 + 9 × len(horizons)
         if ``weather_lookups is not None``       |

    For 5 horizons without weather → 55 features per node. With weather
    active → 55 + 9 + 9·5 = 109 features. If block H is disabled it is
    omitted and the feature dim returns to the previous value without
    breaking already-trained models (each model's first layer is built via
    the factory with ``input_dim`` derived from the first snapshot).

    Args:
        window_df: flights completed in the input window
            ``[current_end - window_delta, current_end)`` — used for the
            aggregations of blocks A, B (partial), C.
        airport_map: IATA → index.
        current_end: instant T that closes the input window.
        window_delta: window duration.
        prediction_horizons: future horizons (e.g. [1,2,4,6,8]).
        history_lookups: precomputed structures.
        delay_threshold: threshold in min for % delayed.
        weather_lookups: precomputed weather structure. ``None`` disables
            block H (default — preserves the historical contract).
    """
    num_nodes = len(airport_map)
    n_lags = len(ARR_DELAY_LAGS)
    n_horizons = len(prediction_horizons)
    feat_dim = 9 + 4 + 5 + n_lags + 2 + 6 + 5 * n_horizons
    if weather_lookups is not None:
        feat_dim += (
            WEATHER_BLOCK_HIST_SIZE + WEATHER_BLOCK_FUTURE_SIZE_PER_H * n_horizons
        )
    features = torch.zeros(num_nodes, feat_dim, dtype=torch.float32)

    # Class B (post-hoc) features can only be derived from flights whose
    # arr_timestamp < current_end (i.e., the flight landed before T).
    # See LEAKAGE INVARIANT in the module docstring.
    if "arr_timestamp" in window_df.columns and not window_df.empty:
        completed_df = window_df[window_df["arr_timestamp"] < current_end]
    else:
        completed_df = window_df

    # ── A. Aggregates of the current window (Class B → completed_df) ──
    if not completed_df.empty:
        # By destination: ArrDelay (headline signal, aligned with the target).
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

        # By origin: DepDelay (endogenous state) + TaxiOut.
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
                    # Reuse column 8 if TaxiIn was missing; if both exist,
                    # we average to avoid inflating the dimension.
                    if features[i, 8] != 0.0:
                        features[i, 8] = (features[i, 8] + float(np.mean(to))) / 2.0
                    else:
                        features[i, 8] = float(np.mean(to))

    # ── B. Volume / disruption ───────────────────────────────────────
    # 9: num_arrivals_completed (Class B: only flights that already landed)
    # 10: num_departures_completed (Class A-ish: input-window schedule)
    # 11: num_cancellations  12: num_diversions  (Class A: known a priori)
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
        # Normalize arrivals/departures by the snapshot maximum.
        for col in (9, 10):
            mx = float(features[:, col].max())
            if mx > 0:
                features[:, col] = features[:, col] / mx

    # ── C. BTS causes (Class B → completed_df, by destination) ───────
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

    # ── D. ArrDelay lags (lookup in the precomputed hourly series) ───
    base_col = 13 + 5
    for lag_idx, lag in enumerate(ARR_DELAY_LAGS):
        bucket = (current_end - lag * window_delta).floor(
            f"{int(window_delta.total_seconds() // 3600)}h"
        )
        for airport, i in airport_map.items():
            features[i, base_col + lag_idx] = _series_lookup(
                history_lookups.arr_delay_by_airport_hour, airport, bucket
            )

    # ── E. Rolling 6h mean+std (lookup in the bucket before current_end) ─
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

    # ── F. Cyclic calendar (same value for all nodes) ────────────────
    base_col += 2
    h_sin, h_cos = _cyclic_encode(current_end.hour, 24)
    dow_sin, dow_cos = _cyclic_encode(current_end.dayofweek, 7)
    m_sin, m_cos = _cyclic_encode(current_end.month - 1, 12)
    cyclic = torch.tensor(
        [h_sin, h_cos, dow_sin, dow_cos, m_sin, m_cos], dtype=torch.float32
    )
    features[:, base_col : base_col + 6] = cyclic.unsqueeze(0).expand(num_nodes, -1)

    # ── G. Future exogenous per horizon (Class A) ────────────────────
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
        # Hour-of-day at the center of the target horizon (cyclic, 1 value).
        # We use sin as the only column per horizon to avoid adding 2×5
        # more; the cosine is omitted because the hour of day is already
        # rich in C.
        target_center_hour = (h_start + window_delta / 2).hour
        h_sin_target, _ = _cyclic_encode(target_center_hour, 24)
        features[:, col_offset + 4] = h_sin_target

    # ── H. Weather (optional, exogenous) ─────────────────────────────
    # H1: aggregates over the input window [current_end - W, current_end).
    # H2: point observation at T + (h - 0.5)·W per horizon (center of the
    #     target window) — "perfect-forecast proxy". The model target is
    #     the delay at T+h, not the weather; the weather is exogenous and
    #     does NOT violate the leakage invariant.
    if weather_lookups is not None and not weather_lookups.empty():
        base_col += 5 * n_horizons  # skip the already-written block G
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
                features[
                    i, col_offset : col_offset + WEATHER_BLOCK_FUTURE_SIZE_PER_H
                ] = torch.from_numpy(obs)

    # ── final normalization: clip + nan→0 ────────────────────────────
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def compute_node_targets(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
) -> torch.Tensor:
    """Generate continuous targets: average ARRIVAL delay per airport.

    Computes the average delay (in minutes) with which flights land at each
    destination airport within the future time window. ArrDelay is the
    natural target for a propagation GNN: the delay "flows" along edges X→Y
    as `DepDelay_X` and lands as `ArrDelay_Y`. Predicting DepDelay (previous
    version) is largely endogenous to the airport and a dense model matches
    it.

    Args:
        window_df: DataFrame of the FUTURE (target) window, filtered by
            arr_timestamp (flights that ARRIVE in the window).
        airport_map: IATA code → index map.

    Returns:
        Continuous tensor [num_nodes] with mean arrival delay (min).
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


# Stable channel indices for the multi-task target tensor.
# All code (models, losses, metrics, reporting) references these names to
# avoid magic numbers and keep the contract synchronized.
TARGET_CHANNEL_ARR_DELAY = 0  # mean ArrDelay (min) — primary regression
TARGET_CHANNEL_DEP_DELAY = 1  # mean DepDelay (min) — auxiliary regression
TARGET_CHANNEL_PCT_DELAYED = 2  # fraction ArrDelay≥threshold ∈ [0, 1] — BCE
NUM_TARGET_CHANNELS = 3


def compute_node_targets_multi_channel(
    window_df: pd.DataFrame,
    airport_map: dict[str, int],
    delay_threshold: float = 15.0,
) -> torch.Tensor:
    """Multi-channel version of :func:`compute_node_targets`.

    Returns ``[num_nodes, NUM_TARGET_CHANNELS]`` with the three channels
    documented in ``TARGET_CHANNEL_*``:

      0) mean ArrDelay (min) per destination airport — primary target.
      1) mean DepDelay (min) per destination airport — auxiliary regression
         (multi-task regularizer).
      2) fraction of flights with ``ArrDelay >= delay_threshold`` ∈ [0, 1]
         per destination airport — binary classification targeted by the
         BCE head of the multi-task model.

    All channels are computed over the **same** window (flights arriving at
    Dest=airport in ``[h_start, h_end)``), so any NaN is filled with 0 just
    like in the single-channel version.

    Args:
        window_df: DataFrame of the FUTURE (target) window, filtered by
            ``arr_timestamp`` (flights that ARRIVE in the window).
        airport_map: IATA → index map.
        delay_threshold: ArrDelay minutes above which a flight is considered
            "delayed" for channel 2.

    Returns:
        Tensor ``[num_nodes, 3]`` (channels: arr_delay, dep_delay,
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
    """Generate multi-horizon multi-channel targets.

    For each horizon h in ``horizons``, computes the 3 target channels over
    the flights that ARRIVE in the window
    ``[current_end + (h-1)*delta, current_end + h*delta)``. See
    :func:`compute_node_targets_multi_channel` for the semantics of each
    channel.

    Args:
        df: Full DataFrame with an ``arr_timestamp`` column (preferred) or
            ``timestamp`` (fallback).
        airport_map: IATA → index map.
        current_end: End of the current features window.
        window_delta: Duration of each time window.
        horizons: List of horizons (e.g., ``[1, 2, 4, 6, 8]``).
        delay_threshold: Threshold in minutes for the classification channel.

    Returns:
        Tensor ``[num_nodes, num_horizons, NUM_TARGET_CHANNELS]`` or
        ``None`` if any horizon has fewer than 10 flights (proxy for
        "enough data" — inherited from the previous version).
    """
    num_nodes = len(airport_map)
    num_horizons = len(horizons)
    targets = torch.zeros(
        num_nodes,
        num_horizons,
        NUM_TARGET_CHANNELS,
        dtype=torch.float32,
    )

    # The `arr_timestamp` column (created in `create_temporal_graphs`) is
    # the correct key for the target window: we want the flights that LAND
    # in [h_start, h_end), not those that take off.
    arr_col = "arr_timestamp" if "arr_timestamp" in df.columns else "timestamp"

    for h_idx, h in enumerate(horizons):
        h_start = current_end + (h - 1) * window_delta
        h_end = current_end + h * window_delta
        h_mask = (df[arr_col] >= h_start) & (df[arr_col] < h_end)
        h_df = df[h_mask]

        if len(h_df) < 10:
            return None

        targets[:, h_idx, :] = compute_node_targets_multi_channel(
            h_df,
            airport_map,
            delay_threshold=delay_threshold,
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
    """Create temporal graph snapshots from flight data.

    Splits the data into time windows and generates one graph per window.
    Each node's features are the airport's delay statistics in that window.
    The target is the average delay in future windows.

    When prediction_horizons is None or [1], it generates single-horizon
    targets [num_nodes] (compatible with BasicGCN). When it has multiple
    values (e.g., [1, 2, 3, 4, 5]), it generates multi-horizon targets
    [num_nodes, num_horizons] for MultiHorizonGAT.

    Args:
        df: Preprocessed DataFrame with FlightDate and Hour.
        airport_map: IATA code → index map.
        edge_index: Adjacency matrix [2, num_edges].
        edge_attr_static: Static edge features [num_edges, 3].
        edge_pairs: List of ``(origin_iata, dest_iata)`` per edge.
        window_hours: Hours per time window.
        delay_threshold: Delay threshold in minutes.
        prediction_horizons: List of future horizons (e.g., [1,2,3,4,5]).
            None or [1] uses single-horizon mode (backward-compatible).

    Returns:
        List of PyG Data objects (one graph per window).
    """
    multi_horizon = prediction_horizons is not None and len(prediction_horizons) > 1

    # Create timestamp by combining FlightDate + Hour
    df = df.copy()
    if "Hour" not in df.columns:
        df["Hour"] = (df["CRSDepTime"] // 100).clip(0, 23)

    df["timestamp"] = pd.to_datetime(df["FlightDate"]) + pd.to_timedelta(
        df["Hour"], unit="h"
    )

    # Create arr_timestamp based on CRSArrTime (the *scheduled* arrival
    # hour). It is the correct key for the target window: we want the
    # flights that ARRIVE at the destination in [h_start, h_end), not those
    # that take off. If CRSArrTime < CRSDepTime we assume an overnight
    # flight and add a day (flights crossing the day boundary are detected
    # correctly for typical U.S. schedules).
    if "CRSArrTime" in df.columns:
        arr_hour = (df["CRSArrTime"] // 100).clip(0, 23).astype("Int16")
        crosses_midnight = (df["CRSArrTime"] < df["CRSDepTime"]).fillna(False)
        df["arr_timestamp"] = (
            pd.to_datetime(df["FlightDate"])
            + pd.to_timedelta(arr_hour.astype("Int64"), unit="h")
            + pd.to_timedelta(crosses_midnight.astype(int), unit="D")
        )
    else:
        # Fallback: if CRSArrTime was not loaded, use the departure hour as
        # a proxy. Keeps backward compatibility with old tests/data but
        # introduces a bias (underestimates how many flights arrive in the
        # window).
        df["arr_timestamp"] = df["timestamp"]

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Create time windows
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    window_delta = pd.Timedelta(hours=window_hours)

    # Compute how many future windows we need
    horizons_for_features = prediction_horizons if multi_horizon else [1]
    if multi_horizon:
        max_horizon = max(prediction_horizons)
        lookahead = max_horizon * window_delta
    else:
        lookahead = window_delta

    # Precompute historical and exogenous lookups. They are computed once
    # over the entire dataset — each snapshot queries values in O(N·H).
    history_lookups = _build_history_lookups(df, airport_map, window_delta)

    graphs = []
    current_time = min_time

    while current_time + window_delta + lookahead <= max_time:
        # Current window (features)
        window_mask = (df["timestamp"] >= current_time) & (
            df["timestamp"] < current_time + window_delta
        )
        window_df = df[window_mask]

        if len(window_df) < 10:
            current_time += window_delta
            continue

        current_end = current_time + window_delta

        if multi_horizon:
            # Multi-horizon multi-channel targets: [num_nodes, num_horizons, 3]
            # (arr_delay, dep_delay, pct_arr_delayed_15). The classification
            # channel needs the threshold, so it is propagated to the builder.
            node_targets = compute_multi_horizon_targets(
                df,
                airport_map,
                current_end,
                window_delta,
                prediction_horizons,
                delay_threshold=delay_threshold,
            )
            if node_targets is None:
                current_time += window_delta
                continue
        else:
            # Single-horizon targets: [num_nodes] (backward-compatible).
            # Use arr_timestamp for the same reason as the multi-horizon case.
            arr_col = "arr_timestamp" if "arr_timestamp" in df.columns else "timestamp"
            next_mask = (df[arr_col] >= current_end) & (
                df[arr_col] < current_end + window_delta
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

        # Mask of nodes with activity (to ignore airports without data)
        active_mask = node_features.abs().sum(dim=1) > 0

        # Concatenate static (3) + dynamic (2) features → [E, 5].
        edge_attr_dynamic = compute_dynamic_edge_attr(
            edge_pairs,
            history_lookups,
            current_end,
            window_delta,
        )
        edge_attr = torch.cat([edge_attr_static, edge_attr_dynamic], dim=1)

        # `timestamp` (ISO string) marks the start of the target window —
        # used for the chronological date-based split in
        # `split_graphs_temporal`. As a string it is compatible with PyG
        # batching (not concatenated, just preserved per graph).
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
        "Temporal snapshots created: %d graphs (window=%dh, "
        "horizons=%s, threshold=%d min)",
        len(graphs),
        window_hours,
        horizons_str,
        delay_threshold,
    )

    return graphs


def normalize_graph_features(
    graphs: list[Data],
    train_indices: list[int] | None = None,
) -> tuple[list[Data], dict[str, torch.Tensor]]:
    """Normalize the graphs' node features (zero-mean, unit-variance).

    Computes mean and standard deviation only over the active nodes of the
    training graphs, and applies the normalization to all graphs. This
    avoids information leakage from the test set.

    Args:
        graphs: Full list of PyG graphs.
        train_indices: Indices of the training graphs used to compute the
            statistics. If None, uses all graphs.

    Returns:
        Tuple of (normalized graphs, statistics {mean, std}).
    """
    if not graphs:
        return graphs, {"mean": torch.zeros(1), "std": torch.ones(1)}

    # Collect active-node features to compute statistics
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
    # Avoid division by zero
    std = torch.clamp(std, min=1e-6)

    # Apply normalization to all graphs
    normalized = []
    for g in graphs:
        g_new = g.clone()
        g_new.x = (g.x - mean) / std
        normalized.append(g_new)

    logger.info(
        "Features normalized: mean=%s, std=%s",
        mean.numpy().round(2),
        std.numpy().round(2),
    )

    return normalized, {"mean": mean, "std": std}


# Version of the graph feature schema. Any change to the x/edge_attr/y
# column contract must increment this value to invalidate on-disk caches
# that assume the previous schema.
#   v2 → v3 (W2 multi-task): multi-horizon targets go from ``[N, H]`` to
#         ``[N, H, 3]`` (channels arr_delay/dep_delay/pct_15).
#   v3 → v4 (Weather): the optional block H is added with 9 + 9·H weather
#         features (see ``WEATHER_BLOCK_*``). With ``weather.enabled =
#         false`` the feature dim does NOT change, but the cache hash
#         includes the flag so different toggles do not collide.
GRAPH_SCHEMA_VERSION = 4


# Mapping of human-readable groups (in config.weather.params) to the
# internal keys recognized by ``WeatherLookups.enabled_params``. Keep in
# sync with the ``weather:`` block of configs/default.yaml.
_WEATHER_PARAM_ALIASES = {
    "wind": "wind",
    "wind_gust": "wind",  # backward-compatible alias
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
    """Read config.weather and build ``WeatherLookups`` if active.

    Steps:
      1. If ``weather.enabled`` is ``False`` (default) → ``None``.
      2. Resolve the temporal range of ``df`` (min and max ``FlightDate``).
         Open-Meteo is inclusive on ``end_date`` → no need to add a day.
      3. Call ``load_weather_for_airports`` with the configured cache_dir
         (``weather.cache_dir`` or the default ``data/processed/weather``).
      4. Build ``WeatherLookups`` with the active param groups.
    """
    weather_config = config.get("weather") or {}
    if not weather_config.get("enabled", False):
        return None

    flight_dates = pd.to_datetime(df["FlightDate"])
    start_date = flight_dates.min().date()
    # Extend by a day for safety: the T+h observations of the last snapshot
    # may fall on the following day if current_end is at night.
    end_date = flight_dates.max().date() + dt.timedelta(days=1)

    cache_dir_raw = weather_config.get("cache_dir") or "data/processed/weather"
    cache_dir = Path(cache_dir_raw)

    raw_params = weather_config.get("params") or [
        "wind",
        "precip_cloud",
        "category",
    ]
    enabled_params = frozenset(_WEATHER_PARAM_ALIASES.get(p, p) for p in raw_params)

    logger.info(
        "Weather enabled (provider=%s, params=%s, %s..%s).",
        weather_config.get("provider", "open_meteo"),
        sorted(enabled_params),
        start_date,
        end_date,
    )

    weather_df = load_weather_for_airports(
        airports,
        start_date,
        end_date,
        cache_dir=cache_dir,
    )

    return _build_weather_lookups(
        weather_df,
        airports=airports,
        enabled_params=enabled_params,
    )


def _snapshot_cache_key(
    df: pd.DataFrame,
    airports: list[str],
    config: dict,
) -> str:
    """Stable hash that identifies a snapshot build.

    Combines the schema version, the subset of config keys that affect the
    output, the airports and a summary of the DataFrame (temporal range,
    row count, columns present). If any of these elements changes, the hash
    changes and the cache is invalidated.
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
    relevant_eval = {
        "delay_threshold_minutes": eval_config.get("delay_threshold_minutes")
    }
    # Stable subset of config.weather that affects the feature tensor.
    # ``provider`` is included so a future switch to ASOS does not reuse a
    # cache generated with Open-Meteo.
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
) -> tuple[list[Data], dict[str, int], dict[str, torch.Tensor] | None]:
    """Full pipeline: from DataFrame to a list of PyG graphs.

    If ``config['graph']['cache_dir']`` points to a directory, it is used as
    a snapshot cache: the first build writes ``snapshots_<hash>.pt`` and
    subsequent ones load from disk if the input signature (schema + config +
    dataset + airports) has not changed.

    Args:
        df: Preprocessed DataFrame (with FlightDate, Hour, Origin, Dest, etc.).
        airports: List of filtered airport codes.
        config: Configuration with 'graph' and 'evaluation' sections.

    Returns:
        Tuple of (list of Data graphs, airport map, normalization
        statistics {mean, std} or None if normalize_features=False). The
        normalization statistics are needed at inference time to scale the
        features the same way as during training; save them alongside the
        checkpoint with ``torch.save(norm_stats, path)``.
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
                "Cache HIT: loading snapshots from %s (%.1f MB)",
                cache_path,
                size_mb,
            )
            payload = torch.load(cache_path, weights_only=False)
            return payload["graphs"], payload["airport_map"], payload.get("norm_stats")
        logger.info(
            "Cache MISS: %s does not exist. Generating snapshots and writing "
            "the cache after the build.",
            cache_path,
        )
    else:
        logger.warning(
            "Snapshot cache DISABLED (graph.cache_dir is not in the config). "
            "Every run will regenerate snapshots from scratch. To enable it, "
            "add `graph.cache_dir: data/processed/snapshots` to the YAML."
        )

    airport_map = build_airport_mapping(airports)

    min_flights = graph_config.get("min_route_flights", 50)
    edge_index, edge_attr_static, edge_pairs = build_edge_index(
        df,
        airport_map,
        min_flights,
    )

    window_hours = graph_config.get("temporal_window_hours", 1)
    delay_threshold = eval_config.get("delay_threshold_minutes", 15)

    prediction_horizons = graph_config.get("prediction_horizons")

    # ── Weather (optional block H) ───────────────────────────────────
    # Only enabled if ``config.weather.enabled = true``. If the network
    # fails, ``load_weather_for_airports`` returns an empty DataFrame and
    # ``_build_weather_lookups`` translates it to an empty ``WeatherLookups``
    # → feature block H is filled with 0 without breaking training.
    weather_lookups = _load_weather_lookups_from_config(
        df,
        airports,
        config,
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

    # Normalize features using only training statistics. The statistics are
    # returned so they can be persisted alongside the checkpoint (needed to
    # scale features the same way at inference time).
    norm_stats: dict[str, torch.Tensor] | None = None
    normalize = graph_config.get("normalize_features", False)
    if normalize and graphs:
        n = len(graphs)
        train_end = int(n * 0.7)
        train_indices = list(range(train_end))
        graphs, norm_stats = normalize_graph_features(graphs, train_indices)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "graphs": graphs,
                "airport_map": airport_map,
                "schema": GRAPH_SCHEMA_VERSION,
                "norm_stats": norm_stats,
            },
            cache_path,
        )
        logger.info("Snapshots cached at: %s", cache_path)

    return graphs, airport_map, norm_stats


def split_graphs_temporal(
    graphs: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    train_end: str | None = None,
    val_end: str | None = None,
) -> dict[str, list[Data]]:
    """Split the temporal graphs into train/val/test.

    Supports two modes:
    - **By date** (preferred): if ``train_end`` and ``val_end`` are passed
      as ISO strings (e.g. ``"2019-07-01"``), the graphs are grouped by
      their ``timestamp`` attribute: ``< train_end`` → train; between
      ``train_end`` and ``val_end`` → val; ``≥ val_end`` → test. Works
      correctly with datasets spanning several years (which the by-month
      mode did not: jan-2018 and jan-2019 fell into the same bucket).
    - **By ratio** (fallback / backward compatibility): if the dates are
      not given, the sequential proportional cut is kept.

    Args:
        graphs: List of temporally ordered PyG graphs.
        train_ratio: Proportion of graphs for training (ratio mode).
        val_ratio: Proportion of graphs for validation (ratio mode).
        train_end: ISO date ``YYYY-MM-DD`` that closes the train split.
        val_end: ISO date ``YYYY-MM-DD`` that closes the val split.

    Returns:
        Dictionary with keys 'train', 'val', 'test'.
    """
    if train_end is not None and val_end is not None:
        # Date mode. Lexicographic comparison over ISO strings is correct
        # because the format is sortable.
        train_split = [g for g in graphs if g.timestamp < train_end]
        val_split = [g for g in graphs if train_end <= g.timestamp < val_end]
        test_split = [g for g in graphs if g.timestamp >= val_end]
        splits = {"train": train_split, "val": val_split, "test": test_split}
        logger.info(
            "Date split: train < %s, val < %s, test ≥ %s",
            train_end,
            val_end,
            val_end,
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
            "Ratio split: train=%.0f%%, val=%.0f%%, test=%.0f%%",
            train_ratio * 100,
            val_ratio * 100,
            (1 - train_ratio - val_ratio) * 100,
        )

    for name, split in splits.items():
        logger.info("Graph split %s: %d graphs", name, len(split))

    return splits


def create_temporal_sequences(
    graphs: list[Data],
    input_window: int = 6,
) -> list[list[Data]]:
    """Group consecutive graphs into temporal sequences for LSTM models.

    Creates sliding windows of size ``input_window`` with step 1 over a list
    of graphs already belonging to the same split (train, val or test). Each
    sequence's target is ``sequence[-1].y`` (targets of the last graph).

    Must be called AFTER ``split_graphs_temporal()`` on each split
    separately to avoid information leakage between sets.

    Args:
        graphs: List of PyG graphs from a single split, temporally ordered.
        input_window: Number of consecutive snapshots per sequence.

    Returns:
        List of sequences, where each sequence is a list of
        ``input_window`` Data objects. Empty if there are fewer graphs than
        ``input_window``.
    """
    if len(graphs) < input_window:
        logger.warning(
            "Only %d graphs available, %d are needed to form sequences. "
            "Returning an empty list.",
            len(graphs),
            input_window,
        )
        return []

    sequences = [
        graphs[i : i + input_window] for i in range(len(graphs) - input_window + 1)
    ]

    logger.info(
        "Temporal sequences created: %d sequences (window=%d graphs)",
        len(sequences),
        input_window,
    )

    return sequences
