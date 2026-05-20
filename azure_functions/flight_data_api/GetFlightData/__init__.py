"""
POST /api/v1/flights/ingest

Reads the equivalent 2022 hour from the historical parquet on OneLake,
then writes class_b actuals, class_a schedule (+8 h), and rolling_buffer.
Trims rolling_buffer partitions older than BUFFER_HOURS.

Uses ``azure-storage-file-datalake`` (DataLakeServiceClient) to talk to
OneLake.  The earlier ``adlfs`` approach silently routed requests to the
wrong endpoint (``onelake.blob.core.windows.net``).

Expected env vars:
    ONELAKE_WORKSPACE_ID  - Fabric workspace name (e.g. TFM_Flight_Prediction)
    ONELAKE_LAKEHOUSE_ID  - Lakehouse name (e.g. FlightData_Lakehouse)
    BUFFER_HOURS          - rolling window size (default 168)
    SOURCE_YEAR           - historical year to remap to (default 2022)

Request body (JSON) or query param:
    timestamp  - ISO-8601 UTC datetime; defaults to current hour if omitted

Returns JSON:
    {"status": "ok", "rows_ingested": N, "timestamp": "...", "equivalent_2022": "...",
     "buffer_partitions_deleted": K}
"""

import io
import json
import logging
import os

import azure.functions as func
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

WORKSPACE_ID = os.environ["ONELAKE_WORKSPACE_ID"]
LAKEHOUSE_ID = os.environ["ONELAKE_LAKEHOUSE_ID"]
BUFFER_HOURS = int(os.environ.get("BUFFER_HOURS", "168"))
SOURCE_YEAR = int(os.environ.get("SOURCE_YEAR", "2022"))

ONELAKE_URL = "https://onelake.dfs.fabric.microsoft.com"
_LH = f"{LAKEHOUSE_ID}.Lakehouse/Files"
SOURCE_PATH = f"{_LH}/raw/historical_2022/Combined_Flights_2022.parquet"
CLASS_B_ROOT = f"{_LH}/live_feed/class_b"
CLASS_A_ROOT = f"{_LH}/live_feed/class_a_schedule"
BUFFER_ROOT = f"{_LH}/live_feed/rolling_buffer"

CLASS_A_COLS = [
    "FlightDate", "Airline", "Origin", "Dest",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance",
    "Month", "DayOfWeek", "DayofMonth", "Cancelled", "Diverted",
]
CLASS_B_EXTRA = [
    "DepTime", "ArrTime", "DepDelay", "ArrDelay",
    "DepDel15", "ArrDel15", "WheelsOff", "WheelsOn",
    "TaxiOut", "TaxiIn", "AirTime", "ActualElapsedTime",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]
CLASS_B_COLS = CLASS_A_COLS + CLASS_B_EXTRA

# Module-level cache: survives across invocations in the same Function host
_source_data = None


def _get_workspace_client():
    from azure.identity import DefaultAzureCredential
    from azure.storage.filedatalake import DataLakeServiceClient

    service = DataLakeServiceClient(
        account_url=ONELAKE_URL, credential=DefaultAzureCredential()
    )
    return service.get_file_system_client(WORKSPACE_ID)


def _download_source(ws) -> bytes:
    global _source_data
    if _source_data is not None:
        logger.info("Using cached source parquet (%d bytes)", len(_source_data))
        return _source_data
    fc = ws.get_file_client(SOURCE_PATH)
    _source_data = fc.download_file().readall()
    logger.info("Downloaded source parquet: %d bytes", len(_source_data))
    return _source_data


def _read_filtered(data: bytes, columns: list, filters: list) -> pa.Table:
    schema = pq.read_schema(io.BytesIO(data))
    available = set(schema.names)
    cols = [c for c in columns if c in available]
    return pq.read_table(io.BytesIO(data), columns=cols, filters=filters)


def _write_parquet(table: pa.Table, ws, path: str) -> None:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    fc = ws.get_file_client(path)
    fc.upload_data(buf.getvalue(), overwrite=True)


def _parse_timestamp(req: func.HttpRequest) -> pd.Timestamp:
    try:
        body = req.get_json() if req.get_body() else {}
    except ValueError:
        body = {}
    ts_str = req.params.get("timestamp") or body.get("timestamp")
    if not ts_str:
        return pd.Timestamp.utcnow().floor("h")
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").floor("h")


def _error(msg: str, status: int = 500) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"status": "error", "message": msg}),
        status_code=status,
        mimetype="application/json",
    )


def main(req: func.HttpRequest) -> func.HttpResponse:
    now_utc = _parse_timestamp(req)
    equiv_dt = now_utc.replace(year=SOURCE_YEAR)

    hour_start = equiv_dt.hour * 100
    hour_end = (equiv_dt.hour + 1) * 100

    logger.info("Ingesting hour %s -> equiv %s: %s", now_utc, SOURCE_YEAR, equiv_dt)

    try:
        ws = _get_workspace_client()
    except Exception as exc:
        logger.exception("Failed to initialise workspace client")
        return _error(f"Auth/client init failed: {exc}")

    # -- Read source parquet (download once, filter in-memory) ----------------
    try:
        source = _download_source(ws)
        table_b = _read_filtered(
            source,
            CLASS_B_COLS,
            [
                ("Month", "=", equiv_dt.month),
                ("DayofMonth", "=", equiv_dt.day),
                ("CRSDepTime", ">=", hour_start),
                ("CRSDepTime", "<", hour_end),
            ],
        )
    except Exception as exc:
        logger.exception("Failed to read source parquet")
        return _error(f"Parquet read failed: {exc}")

    rows = len(table_b)
    ts_path = f"{now_utc.year}/{now_utc.month:02d}/{now_utc.day:02d}/{now_utc.hour:02d}"
    logger.info("Read %d rows for %s", rows, equiv_dt)

    # -- Write class_b + rolling_buffer ---------------------------------------
    try:
        _write_parquet(table_b, ws, f"{CLASS_B_ROOT}/{ts_path}/flights.parquet")
        _write_parquet(table_b, ws, f"{BUFFER_ROOT}/{ts_path}/flights.parquet")
    except Exception as exc:
        logger.exception("Failed to write class_b / buffer")
        return _error(f"Write class_b failed: {exc}")

    # -- Write class_a schedule (current + next 8 hours) ----------------------
    try:
        sched_start = equiv_dt.hour * 100
        sched_end = min((equiv_dt.hour + 9) * 100, 2400)

        table_a = _read_filtered(
            source,
            CLASS_A_COLS,
            [
                ("Month", "=", equiv_dt.month),
                ("DayofMonth", "=", equiv_dt.day),
                ("CRSDepTime", ">=", sched_start),
                ("CRSDepTime", "<", sched_end),
            ],
        )

        if equiv_dt.hour + 9 >= 24:
            next_day = equiv_dt + pd.Timedelta(days=1)
            overflow_end = ((equiv_dt.hour + 9) % 24) * 100
            table_a_overflow = _read_filtered(
                source,
                CLASS_A_COLS,
                [
                    ("Month", "=", next_day.month),
                    ("DayofMonth", "=", next_day.day),
                    ("CRSDepTime", ">=", 0),
                    ("CRSDepTime", "<", overflow_end),
                ],
            )
            table_a = pa.concat_tables([table_a, table_a_overflow])

        _write_parquet(table_a, ws, f"{CLASS_A_ROOT}/{ts_path}/schedule.parquet")
    except Exception as exc:
        logger.exception("Failed to write class_a schedule")
        return _error(f"Write class_a failed: {exc}")

    # -- Trim rolling buffer --------------------------------------------------
    cutoff = now_utc - pd.Timedelta(hours=BUFFER_HOURS)
    deleted = 0
    try:
        paths = list(ws.get_paths(path=BUFFER_ROOT, recursive=True))
        seen_hours = set()
        for p in paths:
            rel = p.name[len(BUFFER_ROOT) :].strip("/")
            parts = rel.split("/")
            if len(parts) >= 4:
                hour_key = "/".join(parts[:4])
                seen_hours.add(hour_key)

        for hour_key in sorted(seen_hours):
            parts = hour_key.split("/")
            try:
                partition_ts = pd.Timestamp(
                    year=int(parts[0]),
                    month=int(parts[1]),
                    day=int(parts[2]),
                    hour=int(parts[3]),
                    tz="UTC",
                )
            except (ValueError, IndexError):
                continue
            if partition_ts < cutoff:
                dc = ws.get_directory_client(f"{BUFFER_ROOT}/{hour_key}")
                dc.delete_directory()
                deleted += 1
    except Exception as exc:
        logger.warning("Buffer trim failed (non-fatal): %s", exc)

    logger.info("Trim removed %d partitions older than %s", deleted, cutoff)

    return func.HttpResponse(
        json.dumps({
            "status": "ok",
            "timestamp": now_utc.isoformat(),
            "equivalent_2022": equiv_dt.isoformat(),
            "rows_ingested": rows,
            "buffer_partitions_deleted": deleted,
        }),
        status_code=200,
        mimetype="application/json",
    )
