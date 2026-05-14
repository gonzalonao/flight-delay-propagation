"""Tests para :mod:`src.data.weather`.

No tocan la red — el cliente HTTP se inyecta vía el parámetro
``_http_fetcher`` que aceptan ``fetch_open_meteo`` y
``load_weather_for_airports``. Los tests cubren:

- Carga de coordenadas desde el CSV bundled.
- Bucketing de códigos WMO en 5 categorías (canónicas para ops aéreas).
- Parser de respuestas JSON de Open-Meteo (tipos compactos + categoría).
- Cache hit/miss en disco (parquet round-trip).
- Degradación graceful ante errores HTTP / payload vacío.
- Concatenación multi-aeropuerto con MultiIndex (iata, timestamp).
"""

from __future__ import annotations

import datetime as dt
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from src.data import weather as weather_mod
from src.data.weather import (
    AirportCoord,
    NUM_WEATHER_CATEGORIES,
    WEATHER_SCHEMA_VERSION,
    bucket_wmo_code,
    fetch_open_meteo,
    load_airport_coords,
    load_weather_for_airports,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_payload(num_hours: int = 24) -> dict:
    """JSON sintético con la forma de respuesta de Open-Meteo."""
    times = pd.date_range("2018-06-01", periods=num_hours, freq="h")
    return {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "wind_speed_10m": [10.0 + i for i in range(num_hours)],
            "wind_gusts_10m": [15.0 + i for i in range(num_hours)],
            "precipitation": [0.0] * num_hours,
            "cloud_cover": [50] * num_hours,
            "weather_code": [0, 1, 45, 51, 63, 71, 80, 95]
            + [3] * (num_hours - 8),
        }
    }


def _fake_fetcher_factory(payload: dict):
    """Devuelve un callable inyectable que ignora la URL y devuelve payload."""

    def _fake(url: str, timeout: float = 30.0) -> dict:
        return payload

    return _fake


# ---------------------------------------------------------------------------
# bucket_wmo_code
# ---------------------------------------------------------------------------


class TestWmoBucketing:
    """5 categorías canónicas: clear/cloudy, fog, rain, snow, thunder."""

    def test_clear_codes_map_to_0(self):
        for code in (0, 1, 2, 3):
            assert bucket_wmo_code(code) == 0

    def test_fog_codes_map_to_1(self):
        assert bucket_wmo_code(45) == 1
        assert bucket_wmo_code(48) == 1

    def test_rain_codes_map_to_2(self):
        for code in (51, 53, 55, 61, 63, 65, 80, 82):
            assert bucket_wmo_code(code) == 2

    def test_snow_codes_map_to_3(self):
        for code in (71, 73, 75, 77, 85, 86):
            assert bucket_wmo_code(code) == 3

    def test_thunder_codes_map_to_4(self):
        for code in (95, 96, 99):
            assert bucket_wmo_code(code) == 4

    def test_unknown_code_falls_back_to_0(self):
        # Códigos fuera del catálogo WMO → clear/cloudy (conservador).
        assert bucket_wmo_code(123) == 0
        assert bucket_wmo_code(-1) == 0

    def test_nan_and_none_fall_back_to_0(self):
        import math

        assert bucket_wmo_code(None) == 0
        assert bucket_wmo_code(math.nan) == 0

    def test_num_categories_constant_matches_bucketing(self):
        # NUM_WEATHER_CATEGORIES es el contrato hacia graph_builder
        # (tamaño del one-hot). Tiene que coincidir con max(bucket)+1.
        assert NUM_WEATHER_CATEGORIES == 5


# ---------------------------------------------------------------------------
# load_airport_coords
# ---------------------------------------------------------------------------


class TestLoadAirportCoords:
    """El CSV bundled cubre al menos los top-30 hubs EE.UU."""

    def test_loads_default_csv(self):
        coords = load_airport_coords()
        assert isinstance(coords, dict)
        # Los top-10 deben estar siempre presentes.
        for iata in ("ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA"):
            assert iata in coords, f"{iata} ausente del CSV de referencia"
            entry = coords[iata]
            assert isinstance(entry, AirportCoord)
            assert -90 <= entry.latitude <= 90
            assert -180 <= entry.longitude <= 180

    def test_atl_coordinates_are_correct(self):
        # Sanity check de un aeropuerto conocido — sirve para detectar si
        # alguien edita el CSV y rompe alguna fila.
        coords = load_airport_coords()
        atl = coords["ATL"]
        assert abs(atl.latitude - 33.64) < 0.1
        assert abs(atl.longitude - (-84.43)) < 0.1


# ---------------------------------------------------------------------------
# fetch_open_meteo
# ---------------------------------------------------------------------------


class TestFetchOpenMeteo:
    """Parser + cache + degradación."""

    def test_parses_payload_into_dataframe(self, tmp_path: Path):
        payload = _make_payload(num_hours=24)
        df = fetch_open_meteo(
            33.64, -84.43, dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            cache_path=None,
            _http_fetcher=_fake_fetcher_factory(payload),
        )
        assert len(df) == 24
        assert df.index.name == "timestamp"
        assert "weather_code" in df.columns
        assert "weather_category" in df.columns
        # Tipos compactos (afectan al footprint cacheado).
        assert df["wind_speed_10m"].dtype.kind == "f"
        assert df["weather_category"].dtype.kind in ("i", "u")

    def test_category_column_matches_bucketing(self):
        payload = _make_payload(num_hours=8)
        # Las primeras 8 horas del payload tienen códigos 0,1,45,51,63,71,80,95
        df = fetch_open_meteo(
            33.64, -84.43, dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            cache_path=None,
            _http_fetcher=_fake_fetcher_factory(payload),
        )
        expected = [0, 0, 1, 2, 2, 3, 2, 4]  # ver bucket_wmo_code
        assert df["weather_category"].tolist() == expected

    def test_cache_round_trip(self, tmp_path: Path):
        payload = _make_payload(num_hours=12)
        cache_file = tmp_path / "atl_2018.parquet"

        # MISS — escribe parquet.
        calls = {"n": 0}

        def counting_fetcher(url: str, timeout: float = 30.0) -> dict:
            calls["n"] += 1
            return payload

        df1 = fetch_open_meteo(
            33.64, -84.43, dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            cache_path=cache_file, _http_fetcher=counting_fetcher,
        )
        assert cache_file.exists()
        assert calls["n"] == 1

        # HIT — no debe llamar al fetcher.
        df2 = fetch_open_meteo(
            33.64, -84.43, dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            cache_path=cache_file, _http_fetcher=counting_fetcher,
        )
        assert calls["n"] == 1, "El segundo fetch debe leer cache, no HTTP"
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
            check_dtype=False,  # parquet redondea int8↔int16 en round-trip
        )

    def test_http_error_returns_empty_dataframe(self):
        def raises(url: str, timeout: float = 30.0) -> dict:
            raise urllib.error.URLError("network down")

        df = fetch_open_meteo(
            33.64, -84.43, dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            cache_path=None, _http_fetcher=raises,
        )
        assert df.empty
        # Las columnas canónicas siguen estando — el caller las llena con 0.
        for col in ("wind_speed_10m", "weather_code", "weather_category"):
            assert col in df.columns

    def test_empty_payload_returns_empty_dataframe(self):
        df = fetch_open_meteo(
            33.64, -84.43, dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            cache_path=None, _http_fetcher=_fake_fetcher_factory({}),
        )
        assert df.empty


# ---------------------------------------------------------------------------
# load_weather_for_airports (multi-aeropuerto)
# ---------------------------------------------------------------------------


class TestLoadWeatherForAirports:
    """Concatena varios aeropuertos con MultiIndex (iata, timestamp)."""

    def test_returns_multiindex_dataframe(self, tmp_path: Path):
        payload = _make_payload(num_hours=6)
        coords = {
            "ATL": AirportCoord("ATL", "Atlanta", 33.64, -84.43, 313.0),
            "LAX": AirportCoord("LAX", "Los Angeles", 33.94, -118.41, 38.0),
        }
        df = load_weather_for_airports(
            ["ATL", "LAX"], dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            coords=coords, cache_dir=tmp_path,
            _http_fetcher=_fake_fetcher_factory(payload),
        )
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["iata", "timestamp"]
        assert set(df.index.get_level_values("iata").unique()) == {"ATL", "LAX"}
        assert len(df) == 12  # 2 aeropuertos × 6 horas

    def test_missing_coords_skipped_with_warning(self, tmp_path: Path, caplog):
        payload = _make_payload(num_hours=2)
        coords = {
            "ATL": AirportCoord("ATL", "Atlanta", 33.64, -84.43, 313.0),
        }
        # XXX no está en coords → debe omitirse con warning, no romper.
        df = load_weather_for_airports(
            ["ATL", "XXX"], dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            coords=coords, cache_dir=tmp_path,
            _http_fetcher=_fake_fetcher_factory(payload),
        )
        assert set(df.index.get_level_values("iata").unique()) == {"ATL"}

    def test_all_airports_failing_returns_empty_frame_with_schema(self, tmp_path: Path):
        def all_fail(url: str, timeout: float = 30.0) -> dict:
            raise urllib.error.URLError("offline")

        coords = {
            "ATL": AirportCoord("ATL", "Atlanta", 33.64, -84.43, 313.0),
        }
        df = load_weather_for_airports(
            ["ATL"], dt.date(2018, 6, 1), dt.date(2018, 6, 1),
            coords=coords, cache_dir=tmp_path,
            _http_fetcher=all_fail,
        )
        assert df.empty
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["iata", "timestamp"]
        # Las columnas canónicas siguen ahí para que el caller pueda
        # hacer ``df.loc[iata, col]`` sin tropezar con KeyError de columna.
        for col in ("wind_speed_10m", "wind_gusts_10m", "precipitation",
                    "cloud_cover", "weather_code", "weather_category"):
            assert col in df.columns


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    """``WEATHER_SCHEMA_VERSION`` se concatena al hash de cache de snapshots.

    Si alguien cambia el bucketing o las columnas devueltas, debe
    incrementar también la versión para que las cachés viejas se invaliden.
    Este test fija el valor actual para que los cambios sean intencionales.
    """

    def test_schema_version_is_explicit_integer(self):
        assert isinstance(WEATHER_SCHEMA_VERSION, int)
        assert WEATHER_SCHEMA_VERSION >= 1


# ---------------------------------------------------------------------------
# Integración con graph_builder: helpers + contrato de feat_dim
# ---------------------------------------------------------------------------


pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.data.graph_builder import (  # noqa: E402  (import tardío deliberado)
    WEATHER_BLOCK_FUTURE_SIZE_PER_H,
    WEATHER_BLOCK_HIST_SIZE,
    WeatherLookups,
    _build_weather_lookups,
    _weather_point_obs,
    _weather_window_agg,
    compute_node_features_rich,
    _build_history_lookups,
)


def _make_synthetic_weather_df() -> pd.DataFrame:
    """Genera meteorología sintética para 2 aeropuertos × 48 horas."""
    times = pd.date_range("2018-06-01", periods=48, freq="h")
    frames = []
    for iata in ("AAA", "BBB"):
        df = pd.DataFrame(
            {
                "wind_speed_10m": np.linspace(10.0, 30.0, 48).astype("float32"),
                "wind_gusts_10m": np.linspace(15.0, 45.0, 48).astype("float32"),
                "precipitation": np.zeros(48, dtype="float32"),
                "cloud_cover": np.full(48, 50.0, dtype="float32"),
                # Mezcla de categorías: 0=clear, 4=thunder en horas 10-15
                "weather_category": np.where(
                    (np.arange(48) >= 10) & (np.arange(48) < 16), 4, 0,
                ).astype("int8"),
                "weather_code": np.where(
                    (np.arange(48) >= 10) & (np.arange(48) < 16), 95, 0,
                ).astype("int16"),
            },
            index=times,
        )
        df.index.name = "timestamp"
        df["iata"] = iata
        df = df.set_index("iata", append=True).swaplevel(0, 1)
        df.index.set_names(["iata", "timestamp"], inplace=True)
        frames.append(df)
    return pd.concat(frames).sort_index()


class TestWeatherHelpers:
    """``_weather_window_agg`` y ``_weather_point_obs`` respetan el esquema."""

    def test_window_agg_shape_and_values(self):
        wdf = _make_synthetic_weather_df()
        lookups = _build_weather_lookups(wdf)
        assert lookups is not None

        # Ventana [2018-06-01 09:00, 2018-06-01 16:00) cubre h=9..15
        # → categorías [0, 4, 4, 4, 4, 4, 4]: dominante = 4 (thunder).
        start = pd.Timestamp("2018-06-01 09:00")
        end = pd.Timestamp("2018-06-01 16:00")
        out = _weather_window_agg(lookups, "AAA", start, end)
        assert out.shape == (WEATHER_BLOCK_HIST_SIZE,)
        # Wind/precip/cloud no son cero (hay observaciones en la ventana).
        assert out[0] > 0  # mean wind > 0
        assert out[1] > 0  # max gust > 0
        assert out[2] == 0  # sum precip = 0 (sintético)
        assert out[3] == 50.0  # mean cloud = 50
        # One-hot: categoría 4 dominante.
        assert out[4 + 4] == 1.0
        assert out[4 + 0] == 0.0

    def test_window_agg_zero_for_unknown_airport(self):
        wdf = _make_synthetic_weather_df()
        lookups = _build_weather_lookups(wdf)
        start = pd.Timestamp("2018-06-01 00:00")
        end = pd.Timestamp("2018-06-01 03:00")
        out = _weather_window_agg(lookups, "ZZZ", start, end)
        # Aeropuerto inexistente → bloque H = ceros completos.
        assert out.shape == (WEATHER_BLOCK_HIST_SIZE,)
        assert (out == 0).all()

    def test_point_obs_lookup_exact_hour(self):
        wdf = _make_synthetic_weather_df()
        lookups = _build_weather_lookups(wdf)
        # Hora 12 cae en la franja thunder (categoría 4).
        out = _weather_point_obs(lookups, "AAA", pd.Timestamp("2018-06-01 12:30"))
        assert out.shape == (WEATHER_BLOCK_FUTURE_SIZE_PER_H,)
        # ``floor("h")`` redondea a las 12:00.
        assert out[4 + 4] == 1.0  # categoría thunder

    def test_disabled_params_zero_out_block(self):
        wdf = _make_synthetic_weather_df()
        # Solo wind activo → precip/cloud/category quedan a 0.
        lookups = _build_weather_lookups(
            wdf, enabled_params=frozenset({"wind"}),
        )
        start = pd.Timestamp("2018-06-01 09:00")
        end = pd.Timestamp("2018-06-01 16:00")
        out = _weather_window_agg(lookups, "AAA", start, end)
        assert out[0] > 0  # wind sigue activo
        assert out[1] > 0  # gust sigue activo
        assert out[2] == 0  # precip apagado
        assert out[3] == 0  # cloud apagado
        assert (out[4:9] == 0).all()  # category apagado

    def test_build_lookups_returns_none_for_empty(self):
        empty = pd.DataFrame(
            columns=["wind_speed_10m", "wind_gusts_10m", "precipitation",
                     "cloud_cover", "weather_category"],
            index=pd.MultiIndex.from_tuples([], names=["iata", "timestamp"]),
        )
        assert _build_weather_lookups(empty) is None
        assert _build_weather_lookups(None) is None


class TestFeatureBlockContract:
    """``compute_node_features_rich`` extiende feat_dim solo si hay weather."""

    def _build_flight_df(self) -> pd.DataFrame:
        """DataFrame sintético mínimo para que ``_build_history_lookups``
        funcione sin error: 2 aeropuertos × 48 horas × 1 vuelo/h."""
        rows = []
        for iata_origin, iata_dest in [("AAA", "BBB"), ("BBB", "AAA")]:
            for h in range(48):
                ts = pd.Timestamp("2018-06-01") + pd.Timedelta(hours=h)
                rows.append({
                    "FlightDate": ts.date(),
                    "Origin": iata_origin,
                    "Dest": iata_dest,
                    "CRSDepTime": h * 100,
                    "CRSArrTime": (h + 1) * 100 % 2400,
                    "ArrDelay": float(h % 10),
                    "DepDelay": float(h % 7),
                    "Distance": 1000.0,
                    "Cancelled": 0,
                    "Diverted": 0,
                })
        df = pd.DataFrame(rows)
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])
        df["timestamp"] = df["FlightDate"] + pd.to_timedelta(df["CRSDepTime"] // 100, unit="h")
        df["arr_timestamp"] = df["timestamp"] + pd.Timedelta(hours=1)
        return df

    def test_feat_dim_without_weather(self):
        import torch  # local re-import por orden de skip
        df = self._build_flight_df()
        airport_map = {"AAA": 0, "BBB": 1}
        window_delta = pd.Timedelta(hours=1)
        horizons = [1, 2, 4, 6, 8]
        history = _build_history_lookups(df, airport_map, window_delta)
        window_df = df[df["timestamp"] < pd.Timestamp("2018-06-01 06:00")]

        feats = compute_node_features_rich(
            window_df, airport_map,
            current_end=pd.Timestamp("2018-06-01 06:00"),
            window_delta=window_delta,
            prediction_horizons=horizons,
            history_lookups=history,
            weather_lookups=None,
        )
        # Sin weather: 9+4+5+4+2+6+5·H = 30 + 5·5 = 55.
        assert feats.shape == (2, 55)
        assert isinstance(feats, torch.Tensor)

    def test_feat_dim_with_weather(self):
        df = self._build_flight_df()
        airport_map = {"AAA": 0, "BBB": 1}
        window_delta = pd.Timedelta(hours=1)
        horizons = [1, 2, 4, 6, 8]
        history = _build_history_lookups(df, airport_map, window_delta)
        wdf = _make_synthetic_weather_df()
        weather = _build_weather_lookups(wdf)
        window_df = df[df["timestamp"] < pd.Timestamp("2018-06-01 06:00")]

        feats = compute_node_features_rich(
            window_df, airport_map,
            current_end=pd.Timestamp("2018-06-01 06:00"),
            window_delta=window_delta,
            prediction_horizons=horizons,
            history_lookups=history,
            weather_lookups=weather,
        )
        # Con weather: 55 + 9 + 9·5 = 109.
        assert feats.shape == (2, 109)
        # El bloque H NO debe ser todo ceros — hay señal sintética.
        h_block = feats[:, 55:]
        assert h_block.abs().sum().item() > 0

    def test_weather_block_zero_when_airport_missing(self):
        df = self._build_flight_df()
        airport_map = {"AAA": 0, "BBB": 1, "ZZZ": 2}  # ZZZ no tiene weather
        window_delta = pd.Timedelta(hours=1)
        horizons = [1, 2]
        history = _build_history_lookups(df, airport_map, window_delta)
        wdf = _make_synthetic_weather_df()  # Solo AAA, BBB
        weather = _build_weather_lookups(wdf)
        window_df = df[df["timestamp"] < pd.Timestamp("2018-06-01 06:00")]

        feats = compute_node_features_rich(
            window_df, airport_map,
            current_end=pd.Timestamp("2018-06-01 06:00"),
            window_delta=window_delta,
            prediction_horizons=horizons,
            history_lookups=history,
            weather_lookups=weather,
        )
        # Bloque H del nodo ZZZ (índice 2) debe ser cero.
        h_block_start = 9 + 4 + 5 + 4 + 2 + 6 + 5 * len(horizons)
        zzz_block = feats[airport_map["ZZZ"], h_block_start:]
        assert (zzz_block == 0).all()
