"""Weather data fetcher for Open-Meteo Historical (ERA5).

Provides hourly weather observations for each airport (identified by IATA)
over the dataset's temporal range. The resulting columns are injected as
exogenous features in the graph builder
(``graph_builder.compute_node_features_rich``).

Source
------
Open-Meteo Historical Weather API (ERA5 reanalysis):
    https://archive-api.open-meteo.com/v1/archive

- Free, no API key.
- ERA5 reanalysis on a ~25 km grid — uniform coverage for the project's 70
  airports (unlike Meteostat, which depends on METAR/ASOS stations and
  suffers gaps at secondary hubs).
- Forecast endpoint with the same schema → future train/serve symmetry.

Canonical schema
-----------------
For each airport and clock hour we return the following columns:

  wind_speed_10m   km/h  mean wind speed at 10 m
  wind_gusts_10m   km/h  max gust at 10 m
  precipitation    mm    rain + liquid-equivalent snow
  cloud_cover      %     cloud cover
  weather_code     int   WMO code (see ``bucket_wmo_code``)
  weather_category int   compact bucket 0..4 (see ``bucket_wmo_code``)

Cache
-----
``data/processed/weather/open_meteo/{iata}_{start}_{end}.parquet`` indexed
by the hourly timestamp. The cache is per airport and requested range: if
the range changes, the file changes. The leakage invariant does NOT depend
on this cache — the temporal cut is done in ``graph_builder``, not here.

Failure policy
--------------
``load_weather_for_airports`` logs a warning and skips an airport if its
coordinate is missing from ``airport_coords.csv`` or if Open-Meteo returns
an HTTP error. The feature builder fills the gaps with 0.0, so training
does NOT break: it degrades to "this airport has no weather signal" without
propagating the exception.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Proxy pool (optional, to work around Open-Meteo 429 rate-limits).
#
# Activation: define one of these two environment variables
#
#   OPEN_METEO_PROXY_URL   URL to the Webshare (or other provider) endpoint
#                          that returns the proxy list in
#                          "host:port:user:pass" format — one line per proxy.
#   OPEN_METEO_PROXY_FILE  Local path to a .txt with the same format.
#
# If both are defined, OPEN_METEO_PROXY_URL wins. The pool is loaded only
# once (lazy + memoized) on the first call that needs it. If neither is
# defined, the module behaves as before (direct HTTP, no retries) and the
# tests keep passing unchanged.
# ---------------------------------------------------------------------------

# Maximum number of proxies to try after a 429/URLError before giving up.
# 5 is enough: on Webshare's free plan the proxies rotate IP per request,
# and the 100 delivered proxies rarely go down as a block.
_MAX_PROXY_RETRIES = 5

_PROXY_POOL: list[str] | None = None
_PROXY_POOL_LOADED = False


def _load_proxy_pool() -> list[str]:
    """Return the list of proxies (lazy + memoized).

    Reads ``OPEN_METEO_PROXY_URL`` or ``OPEN_METEO_PROXY_FILE``. Each valid
    line must be ``host:port:user:pass`` (basic auth) or ``host:port`` (no
    auth). The result is shuffled once so we do not always hammer the first
    proxy.
    """
    global _PROXY_POOL, _PROXY_POOL_LOADED
    if _PROXY_POOL_LOADED:
        return _PROXY_POOL or []
    _PROXY_POOL_LOADED = True

    url = os.environ.get("OPEN_METEO_PROXY_URL")
    path = os.environ.get("OPEN_METEO_PROXY_FILE")
    text: str | None = None

    if url:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "flight-delay-prop/0.1"}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                text = response.read().decode("utf-8")
            logger.info("Proxy pool downloaded from OPEN_METEO_PROXY_URL.")
        except Exception as exc:  # pragma: no cover - network-dependent
            logger.warning(
                "OPEN_METEO_PROXY_URL defined but the download failed: %s. "
                "Continuing without proxies (direct requests).",
                exc,
            )
    elif path:
        try:
            text = Path(path).read_text(encoding="utf-8")
            logger.info("Proxy pool loaded from OPEN_METEO_PROXY_FILE: %s", path)
        except Exception as exc:  # pragma: no cover - FS-dependent
            logger.warning(
                "OPEN_METEO_PROXY_FILE defined (%s) but the read failed: %s. "
                "Continuing without proxies.",
                path,
                exc,
            )

    if not text:
        _PROXY_POOL = []
        return _PROXY_POOL

    pool: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) == 4:
            host, port, user, pwd = parts
            pool.append(f"http://{user}:{pwd}@{host}:{port}")
        elif len(parts) == 2:
            host, port = parts
            pool.append(f"http://{host}:{port}")
        # Other formats are silently ignored (commented lines, headers, etc.).

    random.shuffle(pool)
    _PROXY_POOL = pool
    if pool:
        logger.info("Proxy pool ready: %d proxies available.", len(pool))
    else:
        logger.warning(
            "Proxy pool empty after parsing (wrong format? expected "
            "host:port:user:pass per line)."
        )
    return pool


# Default path to the coordinates CSV (versioned with the code, not the
# data: top-70 U.S. airports).
_DEFAULT_COORDS_PATH = Path(__file__).parent / "reference" / "airport_coords.csv"

# Open-Meteo public endpoint. The free plan allows ~10k calls/day and each
# call covers the full requested range, so 70 airports × 2 years = 70
# calls — very comfortable.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Parameters requested from the API (stable order — affects the cache key).
_HOURLY_PARAMS = (
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
)

# Version of the weather feature schema. Any change to the list of returned
# columns or to the bucketing in ``bucket_wmo_code`` must increment this
# value so that ``_snapshot_cache_key`` (in graph_builder) invalidates
# previous caches that assumed the old schema.
WEATHER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AirportCoord:
    """Geographic coordinates of an airport."""

    iata: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float


def load_airport_coords(
    path: Path | None = None,
) -> dict[str, AirportCoord]:
    """Load the coordinates CSV into an IATA → :class:`AirportCoord` map.

    Args:
        path: Path to the CSV. If ``None``, uses the bundled
            ``src/data/reference/airport_coords.csv`` (top-70 U.S.).

    Returns:
        Dictionary ``{iata: AirportCoord}``.
    """
    csv_path = Path(path) if path is not None else _DEFAULT_COORDS_PATH
    df = pd.read_csv(csv_path)
    return {
        row.iata: AirportCoord(
            iata=row.iata,
            name=row.name,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            elevation_m=float(row.elevation_m),
        )
        for row in df.itertuples(index=False)
    }


# Official WMO weather codes (see Open-Meteo docs):
#   0       Clear sky
#   1-3     Increasing cloudiness
#   45,48   Fog
#   51-57   Drizzle (incl. freezing)
#   61-67   Rain (incl. freezing)
#   71-77   Snow / snow grains
#   80-82   Rain showers
#   85-86   Snow showers
#   95      Thunderstorm
#   96, 99  Thunderstorm with hail
#
# Bucketing into 5 categories (canonical for air operations):
#   0  clear/cloudy   → minimal impact
#   1  fog            → forced IFR, reduces capacity
#   2  rain/drizzle   → wet, possible IFR
#   3  snow           → closures, deicing
#   4  thunderstorm   → ground stops, holds — the #1 cause of NAS delays
_WMO_BUCKET_MAP: dict[int, int] = {}
for code in range(0, 4):
    _WMO_BUCKET_MAP[code] = 0
for code in (45, 48):
    _WMO_BUCKET_MAP[code] = 1
for code in list(range(51, 58)) + list(range(61, 68)) + list(range(80, 83)):
    _WMO_BUCKET_MAP[code] = 2
for code in list(range(71, 78)) + [85, 86]:
    _WMO_BUCKET_MAP[code] = 3
for code in (95, 96, 99):
    _WMO_BUCKET_MAP[code] = 4

NUM_WEATHER_CATEGORIES = 5


def bucket_wmo_code(code: int | float) -> int:
    """Map a WMO code to bucket 0..4 (see the module docstring).

    Unknown codes or ``NaN`` fall to bucket 0 (clear/cloudy) for
    conservativeness — that is the operationally no-impact bucket.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return 0
    try:
        return _WMO_BUCKET_MAP.get(int(code), 0)
    except (TypeError, ValueError):
        return 0


def _cache_path(cache_dir: Path, iata: str, start: dt.date, end: dt.date) -> Path:
    """Deterministic on-disk parquet path for an (iata, range)."""
    fname = f"{iata}_{start.isoformat()}_{end.isoformat()}.parquet"
    return cache_dir / fname


def _open_meteo_request_url(
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
) -> str:
    """Build the full URL with query string for the archive endpoint."""
    params = {
        "latitude": f"{latitude:.4f}",
        "longitude": f"{longitude:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(_HOURLY_PARAMS),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{OPEN_METEO_ARCHIVE_URL}?{qs}"


def _fetch_json(url: str, timeout: float = 30.0, *, proxy: str | None = None) -> dict:
    """HTTP GET with stdlib (no new dependency). Raises ``HTTPError``.

    If ``proxy`` is passed (format ``http://user:pass@host:port``), the
    request is routed through that proxy instead of going out directly.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "flight-delay-prop/0.1"})
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        opener = urllib.request.build_opener(handler)
        with opener.open(req, timeout=timeout) as response:
            body = response.read()
    else:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
    return json.loads(body.decode("utf-8"))


def _parse_open_meteo_response(payload: dict) -> pd.DataFrame:
    """Convert the Open-Meteo JSON response into an hourly DataFrame.

    Expected structure::

        {
          "hourly": {
            "time": ["2018-01-01T00:00", ...],
            "wind_speed_10m": [12.4, ...],
            "wind_gusts_10m": [...],
            "precipitation": [...],
            "cloud_cover": [...],
            "weather_code": [...]
          }
        }

    Returns a DataFrame indexed by ``timestamp`` (datetime64[ns]) with one
    column per parameter and the derived ``weather_category``.
    """
    hourly = payload.get("hourly") or {}
    if not hourly or "time" not in hourly:
        # Empty or malformed response — return an empty DataFrame with the
        # expected columns so everything downstream is pd.NA.
        idx = pd.DatetimeIndex([], name="timestamp")
        return pd.DataFrame(
            {p: pd.Series(dtype="float32") for p in _HOURLY_PARAMS}
            | {"weather_category": pd.Series(dtype="int8")},
            index=idx,
        )

    times = pd.to_datetime(hourly["time"])
    data = {p: hourly.get(p, []) for p in _HOURLY_PARAMS}
    df = pd.DataFrame(data, index=times)
    df.index.name = "timestamp"

    # Compact types: float32 for the numeric ones, int8 for the code.
    for col in ("wind_speed_10m", "wind_gusts_10m", "precipitation", "cloud_cover"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["weather_code"] = pd.to_numeric(df["weather_code"], errors="coerce").astype(
        "Int16"
    )

    # Derived categorical bucket (vectorized to avoid a per-row apply).
    df["weather_category"] = (
        df["weather_code"].map(_WMO_BUCKET_MAP).fillna(0).astype("int8")
    )
    return df


def fetch_open_meteo(
    latitude: float,
    longitude: float,
    start_date: dt.date,
    end_date: dt.date,
    *,
    cache_path: Path | None = None,
    timeout: float = 30.0,
    _http_fetcher=_fetch_json,
) -> pd.DataFrame:
    """Fetch hourly observations from Open-Meteo (with on-disk cache).

    Args:
        latitude, longitude: Decimal coordinates of the point.
        start_date, end_date: Range closed on both sides (Open-Meteo
            includes the full ``end_date``).
        cache_path: If passed, parquet is persisted/read at that path. The
            existence of ``cache_path`` skips the HTTP call.
        timeout: Seconds before the HTTP request fails.
        _http_fetcher: Injectable function for tests (transport mock).

    Returns:
        Hourly DataFrame with columns
        ``[wind_speed_10m, wind_gusts_10m, precipitation, cloud_cover,
        weather_code, weather_category]`` indexed by ``timestamp``. If the
        request fails, returns an empty DataFrame with those columns.
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            cached.index.name = "timestamp"
            return cached
        except Exception as exc:  # pragma: no cover - corruption recovery
            logger.warning(
                "Corrupt weather cache at %s (%s) — retrying HTTP.",
                cache_path,
                exc,
            )

    url = _open_meteo_request_url(latitude, longitude, start_date, end_date)

    # List of proxies to try: first direct egress (None), then up to
    # _MAX_PROXY_RETRIES proxies from the pool. If there is no pool (env
    # vars undefined, or tests with an injected _http_fetcher that does not
    # support proxies), only the direct egress is attempted — identical
    # behavior to the original.
    using_default_fetcher = _http_fetcher is _fetch_json
    proxies_to_try: list[str | None] = [None]
    if using_default_fetcher:
        pool = _load_proxy_pool()
        if pool:
            # We pick a different sub-sample on each call to spread load
            # across the 100 proxies without remembering state.
            proxies_to_try.extend(
                random.sample(pool, k=min(_MAX_PROXY_RETRIES, len(pool)))
            )

    payload: dict | None = None
    last_exc: Exception | None = None
    for attempt_idx, proxy in enumerate(proxies_to_try):
        if attempt_idx > 0:
            # Gentle backoff (0.5–1.0 s) before each retry, avoids
            # hammering the server even through proxies.
            time.sleep(0.5 + random.random() * 0.5)
        try:
            if using_default_fetcher:
                payload = _fetch_json(url, timeout=timeout, proxy=proxy)
            else:
                # Tests inject a fetcher with the old signature (no proxy).
                payload = _http_fetcher(url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if proxy is None and len(proxies_to_try) > 1:
                logger.warning(
                    "Direct Open-Meteo failed (lat=%.4f, lon=%.4f): %s. "
                    "Retrying via proxy pool (%d available).",
                    latitude,
                    longitude,
                    exc,
                    len(proxies_to_try) - 1,
                )
            continue
        else:
            if proxy is not None:
                logger.info(
                    "Open-Meteo OK via proxy on attempt %d (lat=%.4f, lon=%.4f).",
                    attempt_idx + 1,
                    latitude,
                    longitude,
                )
            break

    if payload is None:
        logger.warning(
            "Open-Meteo failed after %d attempt(s) (lat=%.4f, lon=%.4f, %s..%s): %s. "
            "Returning empty DataFrame; the builder will fill with 0.",
            len(proxies_to_try),
            latitude,
            longitude,
            start_date,
            end_date,
            last_exc,
        )
        return _parse_open_meteo_response({})

    df = _parse_open_meteo_response(payload)

    if cache_path is not None and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

    return df


def load_weather_for_airports(
    iata_codes: Iterable[str],
    start_date: dt.date,
    end_date: dt.date,
    *,
    coords: dict[str, AirportCoord] | None = None,
    cache_dir: Path | None = None,
    timeout: float = 30.0,
    _http_fetcher=_fetch_json,
) -> pd.DataFrame:
    """Download (or read from cache) weather data for several airports.

    Args:
        iata_codes: Iterable of IATAs to cover.
        start_date, end_date: Common (inclusive) range for all.
        coords: IATA→AirportCoord map. If ``None``, loaded from the bundled
            CSV. Airports without an entry are skipped with a warning.
        cache_dir: Root directory for the on-disk parquets. ``None``
            disables the cache (downloads every time — only recommended for
            tests).
        timeout: Seconds before each HTTP request fails.
        _http_fetcher: Injectable for tests.

    Returns:
        DataFrame with MultiIndex ``(iata, timestamp)`` and columns
        ``[wind_speed_10m, wind_gusts_10m, precipitation, cloud_cover,
        weather_code, weather_category]``. Airports without data do not
        appear in the index; the caller (graph_builder) treats their
        absence as "no signal" and fills with 0.
    """
    if coords is None:
        coords = load_airport_coords()

    cache_path_root: Path | None = None
    if cache_dir is not None:
        cache_path_root = Path(cache_dir) / "open_meteo"
        cache_path_root.mkdir(parents=True, exist_ok=True)

    iata_list = sorted(set(iata_codes))
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for iata in iata_list:
        coord = coords.get(iata)
        if coord is None:
            missing.append(iata)
            continue

        cp = (
            _cache_path(cache_path_root, iata, start_date, end_date)
            if cache_path_root is not None
            else None
        )
        df = fetch_open_meteo(
            coord.latitude,
            coord.longitude,
            start_date,
            end_date,
            cache_path=cp,
            timeout=timeout,
            _http_fetcher=_http_fetcher,
        )
        if df.empty:
            continue

        df = df.copy()
        df["iata"] = iata
        df = df.set_index("iata", append=True).swaplevel(0, 1)
        df.index.set_names(["iata", "timestamp"], inplace=True)
        frames.append(df)

    if missing:
        logger.warning(
            "No coordinates in airport_coords.csv for %d airports: %s. "
            "Their weather features will be 0 (no signal).",
            len(missing),
            ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
        )

    if not frames:
        logger.warning(
            "load_weather_for_airports: 0 airports returned data "
            "(%d requested). Block H of the feature tensor will stay at 0.",
            len(iata_list),
        )
        return pd.DataFrame(
            columns=list(_HOURLY_PARAMS) + ["weather_category"],
            index=pd.MultiIndex.from_tuples([], names=["iata", "timestamp"]),
        )

    out = pd.concat(frames, axis=0)
    out = out.sort_index()
    logger.info(
        "Weather loaded: %d airports × %d hours (range %s..%s).",
        out.index.get_level_values("iata").nunique(),
        len(out.index.get_level_values("timestamp").unique()),
        start_date,
        end_date,
    )
    return out
