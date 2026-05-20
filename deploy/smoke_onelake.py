"""Smoke-test OneLake access before deploying GetFlightData to Azure Functions.

Uses ``azure-storage-file-datalake`` (the official Azure SDK), which is the
pattern we'll refactor the Function to. ``adlfs`` was tried first and silently
routed to the wrong endpoint (``onelake.blob.core.windows.net``, which doesn't
exist) — hence the misleading ``AccountIsDisabled`` error.

Confirms (locally, against ``az login``):

  1. ``DefaultAzureCredential`` authenticates against OneLake.
  2. The signed-in user has read permission on the Lakehouse.
  3. PyArrow can read ``Combined_Flights_2022.parquet`` with filter pushdown
     against an in-memory buffer downloaded from OneLake.

Run:
    pip install azure-storage-file-datalake azure-identity pyarrow
    az login                              # only needed once per machine
    python deploy/smoke_onelake.py
"""

import io
import sys
import time

WORKSPACE = "TFM_Flight_Prediction"
LAKEHOUSE = "FlightData_Lakehouse"
SOURCE_PATH = (
    f"{LAKEHOUSE}.Lakehouse/Files/raw/historical_2022/Combined_Flights_2022.parquet"
)
ONELAKE_URL = "https://onelake.dfs.fabric.microsoft.com"


def main() -> int:
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.filedatalake import DataLakeServiceClient
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"ERROR: missing dep ({exc}).", file=sys.stderr)
        print(
            "Run: pip install azure-storage-file-datalake azure-identity pyarrow",
            file=sys.stderr,
        )
        return 2

    print("[1/4] Building credential + service client...")
    credential = DefaultAzureCredential()
    service = DataLakeServiceClient(account_url=ONELAKE_URL, credential=credential)

    print(f"[2/4] Listing workspace {WORKSPACE}/{LAKEHOUSE}.Lakehouse/Files/")
    workspace = service.get_file_system_client(WORKSPACE)
    try:
        entries = list(
            workspace.get_paths(path=f"{LAKEHOUSE}.Lakehouse/Files/", recursive=False)
        )
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        print()
        print("  Likely causes:")
        print("    - Your account lacks Member/Contributor on the Fabric workspace.")
        print(
            "      Fix: Fabric portal → TFM_Flight_Prediction → Manage access → add"
        )
        print("      your account as Member.")
        print(
            "    - The Lakehouse doesn't exist or the name doesn't match exactly."
        )
        return 1
    print(f"  OK — found {len(entries)} entries under Files/")
    for e in entries[:10]:
        print(f"    {e.name}{'/' if e.is_directory else ''}")

    print(f"[3/4] Downloading source parquet: {SOURCE_PATH}")
    file_client = workspace.get_file_client(SOURCE_PATH)
    t0 = time.perf_counter()
    try:
        downloader = file_client.download_file()
        data = downloader.readall()
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        print("  Check that Combined_Flights_2022.parquet is at the expected path.")
        return 1
    dt = time.perf_counter() - t0
    mb = len(data) / 1e6
    print(f"  Downloaded {mb:.1f} MB in {dt:.1f}s ({mb / dt:.1f} MB/s)")

    print("[4/4] PyArrow filter pushdown on the in-memory parquet...")
    buf = io.BytesIO(data)
    t0 = time.perf_counter()
    table = pq.read_table(
        buf,
        columns=["FlightDate", "Origin", "Dest", "CRSDepTime", "DepDelay", "ArrDelay"],
        filters=[
            ("Month", "=", 5),
            ("DayofMonth", "=", 20),
            ("CRSDepTime", ">=", 1400),
            ("CRSDepTime", "<", 1500),
        ],
    )
    dt = time.perf_counter() - t0
    print(f"  Filtered read in {dt:.2f}s → {len(table)} rows")
    if len(table):
        print(f"  Sample row: {table.to_pandas().iloc[0].to_dict()}")
    else:
        print(
            "  WARN: 0 rows — not a failure, try a different Month/DayofMonth filter"
        )

    print()
    print(
        "OK — azure-storage-file-datalake + OneLake works. Safe to refactor the Function."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
