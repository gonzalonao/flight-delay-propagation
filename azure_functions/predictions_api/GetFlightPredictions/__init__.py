"""
GET /api/predictions?airport=ATL&horizon=2

Reads the predictions_latest Delta table from OneLake and returns the
most recent predicted arrival delay for the requested airport and horizon.

Expected env vars:
    ONELAKE_WORKSPACE_ID  – Fabric workspace GUID
    ONELAKE_LAKEHOUSE_ID  – Lakehouse GUID (or name)

Query params:
    airport  – IATA code (e.g. ATL), case-insensitive
    horizon  – hours ahead; must be one of {1, 2, 4, 6, 8}
               (matches the model's prediction_horizons in production.yaml)

Returns JSON:
    {"airport": "ATL", "horizon": 2, "predicted_arr_delay_min": 14.3,
     "prediction_ts": "2026-05-19T15:00:00+00:00"}
"""

import json
import logging
import os

import azure.functions as func

logger = logging.getLogger(__name__)

WORKSPACE_ID   = os.environ["ONELAKE_WORKSPACE_ID"]
LAKEHOUSE_ID   = os.environ["ONELAKE_LAKEHOUSE_ID"]
# Actual hours ahead, matching configs/production.yaml graph.prediction_horizons.
# Delta-table columns are named arr_delay_h{H} where H ∈ this set.
VALID_HORIZONS = {1, 2, 4, 6, 8}


def _table_uri() -> str:
    return (
        f"abfss://{WORKSPACE_ID}@onelake.dfs.fabric.microsoft.com"
        f"/{LAKEHOUSE_ID}.Lakehouse/Tables/predictions_latest"
    )


def _read_prediction(airport: str, horizon: int):
    from azure.identity import DefaultAzureCredential
    from deltalake import DeltaTable

    token = DefaultAzureCredential().get_token("https://storage.azure.com/.default").token
    storage_options = {
        "bearer_token": token,
        "use_fabric_endpoint": "true",
    }

    dt = DeltaTable(_table_uri(), storage_options=storage_options)
    df = dt.to_pandas(filters=[("airport", "=", airport)])

    if df.empty:
        return None

    col = f"arr_delay_h{horizon}"
    if col not in df.columns:
        return None

    row = df.sort_values("prediction_ts", ascending=False).iloc[0]
    return {
        "airport": row["airport"],
        "horizon": horizon,
        "predicted_arr_delay_min": round(float(row[col]), 1),
        "prediction_ts": str(row["prediction_ts"]),
    }


def _bad_request(msg: str) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"error": msg}),
        status_code=400,
        mimetype="application/json",
    )


def main(req: func.HttpRequest) -> func.HttpResponse:
    airport = req.params.get("airport", "").strip().upper()
    if not airport:
        return _bad_request("Missing required parameter: airport")

    horizon_raw = req.params.get("horizon", "").strip()
    if not horizon_raw:
        return _bad_request("Missing required parameter: horizon")

    try:
        horizon = int(horizon_raw)
    except ValueError:
        return _bad_request("horizon must be an integer")

    if horizon not in VALID_HORIZONS:
        return _bad_request(f"horizon must be one of {sorted(VALID_HORIZONS)}")

    try:
        result = _read_prediction(airport, horizon)
    except Exception as exc:
        logger.exception("Failed to read predictions table")
        return func.HttpResponse(
            json.dumps({"error": "Failed to read predictions", "detail": str(exc)}),
            status_code=500,
            mimetype="application/json",
        )

    if result is None:
        return func.HttpResponse(
            json.dumps({"error": f"No predictions found for airport {airport}"}),
            status_code=404,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps(result),
        status_code=200,
        mimetype="application/json",
    )
