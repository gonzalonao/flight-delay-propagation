"""Test del adjuntado meteo point-in-time (src.flight_level.weather)."""

import pandas as pd

from src.flight_level.weather import add_weather_asof


def test_weather_asof_origin_tpred_and_dest_arrival():
    work = pd.DataFrame({
        "Origin": ["A"], "Dest": ["B"],
        "dep_dt": [pd.Timestamp("2019-01-01 12:00")],
        "arr_dt": [pd.Timestamp("2019-01-01 15:00")],
    })
    weather = pd.DataFrame({
        "airport": ["A", "A", "B", "B"],
        "hour": [pd.Timestamp("2019-01-01 10:00"), pd.Timestamp("2019-01-01 11:00"),
                 pd.Timestamp("2019-01-01 14:00"), pd.Timestamp("2019-01-01 15:00")],
        "wind_speed_10m": [1.0, 2.0, 3.0, 4.0],
        "wind_gusts_10m": [10.0, 20.0, 30.0, 40.0],
        "precipitation": [0.0, 0.0, 5.0, 0.0],
        "cloud_cover": [50.0, 60.0, 70.0, 80.0],
        "weather_category": [0, 1, 2, 3],
    })
    # H=1 -> t_pred=11:00 -> origen usa hora 11:00 (wind 2.0, wxcat 1).
    # Destino usa la hora de llegada 15:00 (wind 4.0, wxcat 3).
    out = add_weather_asof(work, weather, horizon_h=1)
    assert out["orig_wind"].iloc[0] == 2.0
    assert int(out["orig_wxcat"].iloc[0]) == 1
    assert out["dest_wind"].iloc[0] == 4.0
    assert int(out["dest_wxcat"].iloc[0]) == 3
