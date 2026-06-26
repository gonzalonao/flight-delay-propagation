"""Temporal no-leakage tests on graph features.

Guarantees the invariant documented in `src/data/graph_builder.py`: in a
snapshot with ``current_end = T``, no feature (node or edge) may depend on
Class B (post-hoc) columns of flights whose ``arr_timestamp >= T``.

Strategy: we build a synthetic dataset in which a single "future" flight
carries a distinctive ``ArrDelay`` (a large sentinel). For every snapshot
whose ``current_end`` is before that flight's ``arr_timestamp``, no feature
of the flight's destination airport may come close to the sentinel value:
if it did, there would be leakage.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.graph_builder import build_graph_dataset

# Large, unique sentinel: if it appears (even attenuated) in a feature of a
# snapshot whose current_end is before the future flight's arr_timestamp,
# there is leakage. We pick a value several orders of magnitude above any
# legitimate dataset scale (Distance up to ~5000 miles, ActualElapsedTime
# up to ~600 min, typical delays <300 min) so the sentinel's propagation is
# unmistakable and does not collide with harmless Class A features like
# ``scheduled_mean_distance_in_hN``.
LEAK_SENTINEL = 1_000_000.0


def _make_synthetic_df(
    future_arr_delay: float = LEAK_SENTINEL,
    flights_per_route_per_hour: int = 5,
) -> pd.DataFrame:
    """Create a synthetic DataFrame with one sentinelled "future" flight.

    For each hour of 2018-01-01 there are ``flights_per_route_per_hour``
    AAA→BBB flights and the same number BBB→CCC. The AAA→BBB ones departing
    at 18:00 carry ``future_arr_delay``; the rest, ArrDelay=5 min. The
    per-hour volume (≥10 flights) guarantees that the default filter in
    ``create_temporal_graphs`` (minimum 10 flights per window) does not
    discard snapshots or horizons.
    """
    rows = []
    for h in range(24):
        for k in range(flights_per_route_per_hour):
            ad = future_arr_delay if h == 18 else 5.0
            rows.append(
                {
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
                }
            )
            rows.append(
                {
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
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_config() -> dict:
    return {
        "graph": {
            "top_n_airports": 3,
            "min_route_flights": 1,
            "temporal_window_hours": 1,
            "prediction_horizons": [1, 2],
            "normalize_features": False,  # unnormalized to inspect values
        },
        "evaluation": {"delay_threshold_minutes": 15},
    }


def test_node_features_do_not_leak_future_arr_delay(synthetic_config):
    """Snapshots with current_end ≤ sentinel_arr_ts do not contain its ArrDelay."""
    df = _make_synthetic_df()
    airports = ["AAA", "BBB", "CCC"]
    graphs, airport_map, _norm_stats = build_graph_dataset(
        df, airports, synthetic_config
    )
    assert len(graphs) > 0

    # Sentinel: AAA→BBB flight departing at 18:00 → arr_ts=19:00.
    sentinel_arr_ts = pd.Timestamp("2018-01-01 19:00")
    bbb_idx = airport_map["BBB"]

    for g in graphs:
        # g.timestamp == current_end (see create_temporal_graphs).
        current_end = pd.Timestamp(g.timestamp)
        if current_end > sentinel_arr_ts:
            continue
        features_bbb = g.x[bbb_idx].numpy()
        # Any feature derived from the sentinel ArrDelay=1e6 would be in
        # that order of magnitude (>=1e5 even after averaging). The maximum
        # legitimate features are the Class A exogenous ones that may carry
        # raw Distance/ElapsedTime (scale ~1e3); the 1e5 threshold excludes
        # them without sacrificing sensitivity to the leak.
        assert np.all(np.abs(features_bbb) < 1e5), (
            f"Possible leakage in BBB in snapshot current_end={current_end}: "
            f"features={features_bbb}"
        )


def test_edge_attr_do_not_leak_future_arr_delay(synthetic_config):
    """recent_route_delay does not include the sentinelled future flight."""
    df = _make_synthetic_df()
    airports = ["AAA", "BBB", "CCC"]
    graphs, airport_map, _norm_stats = build_graph_dataset(
        df, airports, synthetic_config
    )
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
                # col 3 = recent_route_delay (normalized by /60, clipped ±1).
                # Without sentinel: 5/60 ≈ 0.083. With leakage saturated at ±1.
                v = float(ea[e, 3])
                assert abs(v) < 0.5, (
                    f"Possible leakage in edge AAA→BBB at "
                    f"current_end={current_end}: recent_route_delay={v}"
                )


def test_targets_do_use_future_window(synthetic_config):
    """Sanity: the target DOES capture the future flight (not leakage)."""
    df = _make_synthetic_df()
    airports = ["AAA", "BBB", "CCC"]
    graphs, airport_map, _norm_stats = build_graph_dataset(
        df, airports, synthetic_config
    )

    # Sentinel arrives at BBB at 19:00. For the snapshot with
    # current_end=18:00, target h=2 covers [19:00, 20:00) and captures the flight.
    bbb = airport_map["BBB"]
    found = False
    for g in graphs:
        current_end = pd.Timestamp(g.timestamp)
        if current_end == pd.Timestamp("2018-01-01 18:00"):
            y = g.y  # W2: [N, num_horizons, 3]
            # h_idx=1 corresponds to horizons[1]=2; channel 0 = arr_delay
            # (the channel where the sentinel ArrDelay=1e6 lives).
            assert y[bbb, 1, 0].item() > 1e5, (
                f"Target h=2 (arr) of BBB does not capture the sentinel: y={y[bbb, 1]}"
            )
            # Channel 2 (pct_arr_delayed_15) ∈ [0, 1] — not contaminated.
            assert 0.0 <= y[bbb, 1, 2].item() <= 1.0
            found = True
            break
    assert found, "No snapshot generated with current_end=18:00"
