"""Fetcher de datos meteorológicos para Open-Meteo Historical (ERA5).

Provee observaciones horarias de tiempo para cada aeropuerto (identificado
por IATA) en el rango temporal del dataset. Las columnas resultantes se
inyectan como features exógenas en el constructor de grafos
(``graph_builder.compute_node_features_rich``).

Fuente
------
Open-Meteo Historical Weather API (Reanalysis ERA5):
    https://archive-api.open-meteo.com/v1/archive

- Gratuita, sin API key.
- Reanalysis ERA5 sobre cuadrícula ~25 km — cobertura uniforme para los 70
  aeropuertos del proyecto (a diferencia de Meteostat, que depende de
  estaciones METAR/ASOS y sufre huecos en hubs secundarios).
- Endpoint forecast con el mismo esquema → simetría train/serve futura.

Esquema canónico
----------------
Por cada aeropuerto y hora natural devolvemos las siguientes columnas:

  wind_speed_10m   km/h  velocidad media del viento a 10 m
  wind_gusts_10m   km/h  ráfaga máxima a 10 m
  precipitation    mm    lluvia + nieve líquida equivalente
  cloud_cover      %     cobertura nubosa
  weather_code     int   código WMO (ver ``bucket_wmo_code``)
  weather_category int   bucket compacto 0..4 (ver ``bucket_wmo_code``)

Caché
-----
``data/processed/weather/open_meteo/{iata}_{start}_{end}.parquet`` indexado
por la marca temporal horaria. La caché es por aeropuerto y rango
solicitado: si el rango cambia, el fichero cambia. La invariante de
leakage NO depende de esta caché — el corte temporal se hace en
``graph_builder``, no aquí.

Política ante fallos
--------------------
``load_weather_for_airports`` registra un warning y omite el aeropuerto
si su coordenada no figura en ``airport_coords.csv`` o si Open-Meteo
devuelve un error HTTP. El builder de features rellena los huecos con
0.0, así que el entrenamiento NO se rompe: degrada a "este aeropuerto no
tiene señal meteorológica" sin propagar la excepción.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Ruta por defecto al CSV de coordenadas (versionado con el código,
# no con los datos: aeropuertos top-70 EE.UU.).
_DEFAULT_COORDS_PATH = Path(__file__).parent / "reference" / "airport_coords.csv"

# Endpoint público de Open-Meteo. El plan gratuito permite ~10k llamadas/día
# y cada llamada cubre el rango completo solicitado, así que 70 aeropuertos
# × 2 años = 70 llamadas — muy holgado.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Parámetros pedidos al API (orden estable — afecta a la firma del cache).
_HOURLY_PARAMS = (
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
)

# Versión del esquema de weather features. Cualquier cambio en la lista de
# columnas devueltas o en el bucketing de ``bucket_wmo_code`` debe
# incrementar este valor para que ``_snapshot_cache_key`` (en
# graph_builder) invalide cachés anteriores que asumieran el esquema viejo.
WEATHER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AirportCoord:
    """Coordenadas geográficas de un aeropuerto."""

    iata: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float


def load_airport_coords(
    path: Path | None = None,
) -> dict[str, AirportCoord]:
    """Carga el CSV de coordenadas en un mapeo IATA → :class:`AirportCoord`.

    Args:
        path: Ruta al CSV. Si ``None``, usa el bundled
            ``src/data/reference/airport_coords.csv`` (top-70 EE.UU.).

    Returns:
        Diccionario ``{iata: AirportCoord}``.
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


# WMO weather codes oficiales (ver Open-Meteo docs):
#   0       Cielo despejado
#   1-3     Nubosidad creciente
#   45,48   Niebla
#   51-57   Llovizna (incluye congelante)
#   61-67   Lluvia (incluye congelante)
#   71-77   Nieve / granos de nieve
#   80-82   Chubascos de lluvia
#   85-86   Chubascos de nieve
#   95      Tormenta
#   96, 99  Tormenta con granizo
#
# Bucketing en 5 categorías (canonical para operaciones aéreas):
#   0  clear/cloudy   → impacto mínimo
#   1  fog            → IFR forzado, reduce capacidad
#   2  rain/drizzle   → mojado, posible IFR
#   3  snow           → cierres, deicing
#   4  thunderstorm   → ground stops, holds — la causa #1 de delays NAS
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
    """Mapea un código WMO al bucket 0..4 (ver módulo docstring).

    Códigos desconocidos o ``NaN`` caen a bucket 0 (clear/cloudy) por
    conservadurismo — ese es el bucket sin impacto operativo.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return 0
    try:
        return _WMO_BUCKET_MAP.get(int(code), 0)
    except (TypeError, ValueError):
        return 0


def _cache_path(cache_dir: Path, iata: str, start: dt.date, end: dt.date) -> Path:
    """Ruta determinista del parquet en disco para un (iata, rango)."""
    fname = f"{iata}_{start.isoformat()}_{end.isoformat()}.parquet"
    return cache_dir / fname


def _open_meteo_request_url(
    latitude: float,
    longitude: float,
    start: dt.date,
    end: dt.date,
) -> str:
    """Construye la URL completa con query string para el archive endpoint."""
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


def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    """HTTP GET con stdlib (sin nueva dependencia). Levanta ``HTTPError``."""
    req = urllib.request.Request(url, headers={"User-Agent": "flight-delay-prop/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def _parse_open_meteo_response(payload: dict) -> pd.DataFrame:
    """Convierte la respuesta JSON de Open-Meteo en un DataFrame horario.

    Estructura esperada::

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

    Devuelve un DataFrame indexado por ``timestamp`` (datetime64[ns]) con
    una columna por parámetro y la categoría derivada ``weather_category``.
    """
    hourly = payload.get("hourly") or {}
    if not hourly or "time" not in hourly:
        # Respuesta vacía o malformada — devolvemos DataFrame vacío con
        # las columnas esperadas para que aguas abajo todo sea pd.NA.
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

    # Tipos compactos: float32 para los numéricos, int8 para el código.
    for col in ("wind_speed_10m", "wind_gusts_10m", "precipitation", "cloud_cover"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df["weather_code"] = pd.to_numeric(df["weather_code"], errors="coerce").astype(
        "Int16"
    )

    # Bucket categórico derivado (vectorizado para evitar apply por fila).
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
    """Obtiene observaciones horarias de Open-Meteo (con caché en disco).

    Args:
        latitude, longitude: Coordenadas decimales del punto.
        start_date, end_date: Rango cerrado por ambos lados (Open-Meteo
            incluye ``end_date`` completo).
        cache_path: Si se pasa, se persiste/lee parquet en esa ruta. La
            existencia de ``cache_path`` salta la llamada HTTP.
        timeout: Segundos antes de fallar la petición HTTP.
        _http_fetcher: Función inyectable para tests (mock del transport).

    Returns:
        DataFrame horario con columnas
        ``[wind_speed_10m, wind_gusts_10m, precipitation, cloud_cover,
        weather_code, weather_category]`` indexado por ``timestamp``.
        Si la petición falla, devuelve DataFrame vacío con esas columnas.
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            cached.index.name = "timestamp"
            return cached
        except Exception as exc:  # pragma: no cover - corruption recovery
            logger.warning(
                "Caché meteorológica corrupta en %s (%s) — reintentando HTTP.",
                cache_path, exc,
            )

    url = _open_meteo_request_url(latitude, longitude, start_date, end_date)
    try:
        payload = _http_fetcher(url, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning(
            "Open-Meteo falló (lat=%.4f, lon=%.4f, %s..%s): %s. "
            "Devolviendo DataFrame vacío; el builder rellenará con 0.",
            latitude, longitude, start_date, end_date, exc,
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
    """Descarga (o lee de caché) datos meteorológicos para varios aeropuertos.

    Args:
        iata_codes: Iterable de IATAs a cubrir.
        start_date, end_date: Rango (inclusivo) común para todos.
        coords: Mapeo IATA→AirportCoord. Si ``None``, se carga del CSV
            bundled. Aeropuertos sin entrada se omiten con warning.
        cache_dir: Directorio raíz para los parquets en disco. ``None``
            desactiva la caché (descarga cada vez — solo recomendable
            para tests).
        timeout: Segundos antes de fallar cada petición HTTP.
        _http_fetcher: Inyectable para tests.

    Returns:
        DataFrame con MultiIndex ``(iata, timestamp)`` y columnas
        ``[wind_speed_10m, wind_gusts_10m, precipitation, cloud_cover,
        weather_code, weather_category]``. Aeropuertos sin datos no
        aparecen en el índice; el caller (graph_builder) trata su ausencia
        como "no señal" y rellena con 0.
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
            coord.latitude, coord.longitude, start_date, end_date,
            cache_path=cp, timeout=timeout, _http_fetcher=_http_fetcher,
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
            "Sin coordenadas en airport_coords.csv para %d aeropuertos: %s. "
            "Sus features meteorológicas serán 0 (no-señal).",
            len(missing), ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
        )

    if not frames:
        logger.warning(
            "load_weather_for_airports: 0 aeropuertos devolvieron datos "
            "(%d solicitados). El bloque H del feature tensor quedará en 0.",
            len(iata_list),
        )
        return pd.DataFrame(
            columns=list(_HOURLY_PARAMS) + ["weather_category"],
            index=pd.MultiIndex.from_tuples([], names=["iata", "timestamp"]),
        )

    out = pd.concat(frames, axis=0)
    out = out.sort_index()
    logger.info(
        "Meteorología cargada: %d aeropuertos × %d horas (rango %s..%s).",
        out.index.get_level_values("iata").nunique(),
        len(out.index.get_level_values("timestamp").unique()),
        start_date, end_date,
    )
    return out
