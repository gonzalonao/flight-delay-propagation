"""
POST /api/v1/flights/ingest

Reads the equivalent 2022 hour from the historical parquet on OneLake,
then writes class_b actuals, class_a schedule (+8 h), and rolling_buffer.
Trims rolling_buffer partitions older than BUFFER_HOURS.

Expected env vars:
    ONELAKE_WORKSPACE_ID  – Fabric workspace GUID
    ONELAKE_LAKEHOUSE_ID  – Lakehouse GUID (or name)
    BUFFER_HOURS          – rolling window size (default 168)
    SOURCE_YEAR           – historical year to remap to (default 2022)

Request body (JSON) or query param:
    timestamp  – ISO-8601 UTC datetime; defaults to current hour if omitted

Returns JSON:
    {"status": "ok", "rows_ingested": N, "timestamp": "...", "equivalent_2022": "...",
     "buffer_partitions_deleted": K}
"""

import json
import logging
import os
from pathlib import PurePosixPath

import azure.functions as func
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

WORKSPACE_ID  = os.environ["ONELAKE_WORKSPACE_ID"]
LAKEHOUSE_ID  = os.environ["ONELAKE_LAKEHOUSE_ID"]
BUFFER_HOURS  = int(os.environ.get("BUFFER_HOURS", "168"))
SOURCE_YEAR   = int(os.environ.get("SOURCE_YEAR", "2022"))

_LH           = f"{LAKEHOUSE_ID}.Lakehouse/Files"
SOURCE_PARQUET = f"{WORKSPACE_ID}/{_LH}/raw/historical_2022/Combined_Flights_2022.parquet"
CLASS_B_ROOT   = f"{WORKSPACE_ID}/{_LH}/live_feed/class_b"
CLASS_A_ROOT   = f"{WORKSPACE_ID}/{_LH}/live_feed/class_a_schedule"
BUFFER_ROOT    = f"{WORKSPACE_ID}/{_LH}/live_feed/rolling_buffer"

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


def _get_fs():
    import adlfs
    from azure.identity import DefaultAzureCredential
    return adlfs.AzureBlobFileSystem(
        account_name="onelake",
        credential=DefaultAzureCredential(),
        custom_domain="onelake.dfs.fabric.microsoft.com",
    )


def _write_parquet(table: pa.Table, fs, path: str) -> None:
    parent = str(PurePosixPath(path).parent)
    fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as fh:
        pq.write_table(table, fh, compression="snappy")


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
    now_utc  = _parse_timestamp(req)
    equiv_dt = now_utc.replace(year=SOURCE_YEAR)

    hour_start = equiv_dt.hour * 100
    hour_end   = (equiv_dt.hour + 1) * 100

    logger.info("Ingesting hour %s → equiv %s: %s", now_utc, SOURCE_YEAR, equiv_dt)

    try:
        fs = _get_fs()
    except Exception as exc:
        logger.exception("Failed to initialise filesystem")
        return _error(f"Auth/filesystem init failed: {exc}")

    # ── Read source parquet (filter pushdown, never loads full file) ──────
    try:
        schema        = pq.read_schema(SOURCE_PARQUET, filesystem=fs)
        available     = set(schema.names)
        b_cols        = [c for c in CLASS_B_COLS if c in available]
        a_cols        = [c for c in CLASS_A_COLS  if c in available]

        table_b = pq.read_table(
            SOURCE_PARQUET,
            filesystem=fs,
            columns=b_cols,
            filters=[
                ("Month",      "=",  equiv_dt.month),
                ("DayofMonth", "=",  equiv_dt.day),
                ("CRSDepTime", ">=", hour_start),
                ("CRSDepTime", "<",  hour_end),
            ],
        )
    except Exception as exc:
        logger.exception("Failed to read source parquet")
        return _error(f"Parquet read failed: {exc}")

    rows    = len(table_b)
    ts_path = f"{now_utc.year}/{now_utc.month:02d}/{now_utc.day:02d}/{now_utc.hour:02d}"
    logger.info("Read %d rows for %s", rows, equiv_dt)

    # ── Write class_b + rolling_buffer ────────────────────────────────────
    try:
        _write_parquet(table_b, fs, f"{CLASS_B_ROOT}/{ts_path}/flights.parquet")
        _write_parquet(table_b, fs, f"{BUFFER_ROOT}/{ts_path}/flights.parquet")
    except Exception as exc:
        logger.exception("Failed to write class_b / buffer")
        return _error(f"Write class_b failed: {exc}")

    # ── Write class_a schedule (current + next 8 hours) ──────────────────
    try:
        sched_start = equiv_dt.hour * 100
        sched_end   = min((equiv_dt.hour + 9) * 100, 2400)

        table_a_main = pq.read_table(
            SOURCE_PARQUET,
            filesystem=fs,
            columns=a_cols,
            filters=[
                ("Month",      "=",  equiv_dt.month),
                ("DayofMonth", "=",  equiv_dt.day),
                ("CRSDepTime", ">=", sched_start),
                ("CRSDepTime", "<",  sched_end),
            ],
        )

        if equiv_dt.hour + 9 >= 24:
            next_day     = equiv_dt + pd.Timedelta(days=1)
            overflow_end = ((equiv_dt.hour + 9) % 24) * 100
            table_a_overflow = pq.read_table(
                SOURCE_PARQUET,
                filesystem=fs,
                columns=a_cols,
                filters=[
                    ("Month",      "=",  next_day.month),
                    ("DayofMonth", "=",  next_day.day),
                    ("CRSDepTime", ">=", 0),
                    ("CRSDepTime", "<",  overflow_end),
                ],
            )
            table_a = pa.concat_tables([table_a_main, table_a_overflow])
        else:
            table_a = table_a_main

        _write_parquet(table_a, fs, f"{CLASS_A_ROOT}/{ts_path}/schedule.parquet")
    except Exception as exc:
        logger.exception("Failed to write class_a schedule")
        return _error(f"Write class_a failed: {exc}")

    # ── Trim rolling buffer ───────────────────────────────────────────────
    cutoff  = now_utc - pd.Timedelta(hours=BUFFER_HOURS)
    deleted = 0
    try:
        for year_dir in fs.ls(BUFFER_ROOT, detail=False):
            for month_dir in fs.ls(year_dir, detail=False):
                for day_dir in fs.ls(month_dir, detail=False):
                    for hour_dir in fs.ls(day_dir, detail=False):
                        try:
                            parts = hour_dir.rstrip("/").split("/")
                            # path tail is .../YYYY/MM/DD/HH
                            partition_ts = pd.Timestamp(
                                year=int(parts[-4]),
                                month=int(parts[-3]),
                                day=int(parts[-2]),
                                hour=int(parts[-1]),
                                tz="UTC",
                            )
                            if partition_ts < cutoff:
                                fs.rm(hour_dir, recursive=True)
                                deleted += 1
                        except (ValueError, IndexError):
                            pass
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
