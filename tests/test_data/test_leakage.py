"""Tests de no-leakage temporal en features de grafo.

Garantiza la invariante documentada en `src/data/graph_builder.py`: en un
snapshot con ``current_end = T``, ninguna feature (de nodo o arista) puede
depender de columnas Class B (post-hoc) de vuelos cuyo ``arr_timestamp >= T``.

Estrategia: construimos un dataset sintético en el que un único vuelo
"futuro" lleva un ``ArrDelay`` distintivo (un sentinel grande). Para todos
los snapshots cuyo ``current_end`` sea anterior al ``arr_timestamp`` de ese
vuelo, ninguna feature del aeropuerto destino del vuelo debe acercarse al
valor sentinel: si lo hiciera, habría leakage.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.graph_builder import build_graph_dataset

# Sentinel grande y único: si aparece (incluso atenuado) en una feature de
# un snapshot cuyo current_end está antes del arr_timestamp del vuelo
# futuro, hay leakage. Se elige un valor varios órdenes de magnitud por
# encima de cualquier escala legítima del dataset (Distance hasta ~5000
# millas, ActualElapsedTime hasta ~600 min, retrasos típicos <300 min)
# para que la propagación del sentinel sea inconfundible y no colisione
# con features Class A inocuas como ``scheduled_mean_distance_in_hN``.
LEAK_SENTINEL = 1_000_000.0


def _make_synthetic_df(
    future_arr_delay: float = LEAK_SENTINEL,
    flights_per_route_per_hour: int = 5,
) -> pd.DataFrame:
    """Crea un DataFrame sintético con un vuelo "futuro" sentinelado.

    Para cada hora del 2018-01-01 hay ``flights_per_route_per_hour`` vuelos
    AAA→BBB y la misma cantidad BBB→CCC. Los AAA→BBB que despegan a las
    18:00 llevan ``future_arr_delay``; el resto, ArrDelay=5 min. El
    volumen por hora (≥10 vuelos) garantiza que el filtro por defecto de
    ``create_temporal_graphs`` (mínimo 10 vuelos por ventana) no descarte
    snapshots ni horizontes.
    """
    rows = []
    for h in range(24):
        for k in range(flights_per_route_per_hour):
            ad = future_arr_delay if h == 18 else 5.0
            rows.append({
                "FlightDate": pd.Timestamp("2018-01-01"),
                "Airline": "AA",
                "Origin": "AAA",
                "Dest": "BBB",
                "CRSDepTime": h * 100 + k,
                "CRSArrTime": ((h + 1) % 24) * 100 + k,
                "CRSElapsedTime": 60,
                "Distance": 500,
                "AirTime": 50,
                "DepDelay": 0.0,
                "ArrDelay": ad,
                "TaxiOut": 10.0,
                "TaxiIn": 5.0,
                "Cancelled": False,
                "Diverted": False,
                "DepDel15": False,
                "ArrDel15": ad >= 15.0,
            })
            rows.append({
                "FlightDate": pd.Timestamp("2018-01-01"),
                "Airline": "AA",
                "Origin": "BBB",
                "Dest": "CCC",
                "CRSDepTime": h * 100 + k,
                "CRSArrTime": ((h + 2) % 24) * 100 + k,
                "CRSElapsedTime": 120,
                "Distance": 700,
                "AirTime": 100,
                "DepDelay": 0.0,
                "ArrDelay": 5.0,
                "TaxiOut": 10.0,
                "TaxiIn": 5.0,
                "Cancelled": False,
                "Diverted": False,
                "DepDel15": False,
                "ArrDel15": False,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_config() -> dict:
    return {
        "graph": {
            "top_n_airports": 3,
            "min_route_flights": 1,
            "temporal_window_hours": 1,
            "prediction_horizons": [1, 2],
            "normalize_features": False,  # sin normalizar para inspeccionar valores
        },
        "evaluation": {"delay_threshold_minutes": 15},
    }


def test_node_features_do_not_leak_future_arr_delay(synthetic_config):
    """Snapshots con current_end ≤ sentinel_arr_ts no contienen su ArrDelay."""
    df = _make_synthetic_df()
    airports = ["AAA", "BBB", "CCC"]
    graphs, airport_map = build_graph_dataset(df, airports, synthetic_config)
    assert len(graphs) > 0

    # Sentinel: vuelo AAA→BBB que sale a las 18:00 → arr_ts=19:00.
    sentinel_arr_ts = pd.Timestamp("2018-01-01 19:00")
    bbb_idx = airport_map["BBB"]

    for g in graphs:
        # g.timestamp == current_end (ver create_temporal_graphs).
        current_end = pd.Timestamp(g.timestamp)
        if current_end > sentinel_arr_ts:
            continue
        features_bbb = g.x[bbb_idx].numpy()
        # Cualquier feature derivada del sentinel ArrDelay=1e6 estaría en
        # ese orden de magnitud (>=1e5 incluso después de promediar). Las
        # features legítimas máximas son las exógenas Class A que pueden
        # llevar Distance/ElapsedTime crudos (escala ~1e3); el umbral
        # 1e5 las excluye sin sacrificar sensibilidad al leak.
        assert np.all(np.abs(features_bbb) < 1e5), (
            f"Posible leakage en BBB en snapshot current_end={current_end}: "
            f"features={features_bbb}"
        )


def test_edge_attr_do_not_leak_future_arr_delay(synthetic_config):
    """recent_route_delay no incluye al vuelo futuro sentinelado."""
    df = _make_synthetic_df()
    airports = ["AAA", "BBB", "CCC"]
    graphs, airport_map = build_graph_dataset(df, airports, synthetic_config)
    assert len(graphs) > 0

    sentinel_arr_ts = pd.Timestamp("2018-01-01 19:00")
    aaa = airport_map["AAA"]
    bbb = airport_map["BBB"]

    for g in graphs:
        current_end = pd.Timestamp(g.timestamp)
        if current_end > sentinel_arr_ts:
            continue
        ea = g.edge_attr  # [E, 5]
        ei = g.edge_index  # [2, E]
        for e in range(ei.shape[1]):
            if ei[0, e].item() == aaa and ei[1, e].item() == bbb:
                # col 3 = recent_route_delay (normalizado por /60, clipped ±1).
                # Sin sentinel: 5/60 ≈ 0.083. Con leakage sat. en ±1.
                v = float(ea[e, 3])
                assert abs(v) < 0.5, (
                    f"Posible leakage en arista AAA→BBB en "
                    f"current_end={current_end}: recent_route_delay={v}"
                )


def test_targets_do_use_future_window(synthetic_config):
    """Sanidad: el target SÍ captura al vuelo futuro (no es leakage)."""
    df = _make_synthetic_df()
    airports = ["AAA", "BBB", "CCC"]
    graphs, airport_map = build_graph_dataset(df, airports, synthetic_config)

    # Sentinel arriba a BBB a las 19:00. Para el snapshot con
    # current_end=18:00, target h=2 cubre [19:00, 20:00) y captura el vuelo.
    bbb = airport_map["BBB"]
    found = False
    for g in graphs:
        current_end = pd.Timestamp(g.timestamp)
        if current_end == pd.Timestamp("2018-01-01 18:00"):
            y = g.y  # [N, num_horizons]
            # h_idx=1 corresponde a horizons[1]=2.
            assert y[bbb, 1].item() > 1e5, (
                f"Target h=2 de BBB no captura el vuelo sentinel: y={y[bbb]}"
            )
            found = True
            break
    assert found, "No se generó snapshot con current_end=18:00"
