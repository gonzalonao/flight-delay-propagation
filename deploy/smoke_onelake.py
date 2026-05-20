"""Smoke-test OneLake access before deploying GetFlightData to Azure Functions.

Confirms (locally, against ``az login``):

  1. ``DefaultAzureCredential`` can authenticate against OneLake.
  2. The signed-in user has read permission on ``TFM_Flight_Prediction /
     FlightData_Lakehouse``.
  3. PyArrow can read ``Combined_Flights_2022.parquet`` with filter pushdown
     through fsspec/adlfs.

If this passes, the Function App will work end-to-end as long as its
Managed Identity is granted the same role on the Lakehouse. If this fails,
we refactor the Function to use ``azure-storage-file-datalake`` directly.

Run:
    pip install adlfs azure-identity pyarrow
    az login                              # only needed once per machine
    python deploy/smoke_onelake.py
"""

import sys
import time

WORKSPACE = "TFM_Flight_Prediction"
LAKEHOUSE = "FlightData_Lakehouse"
SOURCE = (
    f"{WORKSPACE}/{LAKEHOUSE}.Lakehouse/Files/raw/historical_2022/"
    f"Combined_Flights_2022.parquet"
)


def main() -> int:
    try:
        import adlfs
    except ImportError:
        print("ERROR: adlfs not installed. Run: pip install adlfs", file=sys.stderr)
        return 2

    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        print(
            "ERROR: azure-identity not installed. Run: pip install azure-identity",
            file=sys.stderr,
        )
        return 2

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow not installed. Run: pip install pyarrow", file=sys.stderr)
        return 2

    print("[1/4] Building credential...")
    credential = DefaultAzureCredential()

    print("[2/4] Building filesystem (account=onelake, host=fabric.microsoft.com)...")
    # This is the exact pattern used by azure_functions/flight_data_api.
    # If this fails, we'll know to refactor before deploying.
    try:
        fs = adlfs.AzureBlobFileSystem(
            account_name="onelake",
            credential=credential,
            custom_domain="onelake.dfs.fabric.microsoft.com",
        )
    except TypeError as exc:
        # custom_domain may not be a valid kwarg on this adlfs version.
        print(f"  custom_domain kwarg rejected: {exc}")
        print("  Trying without custom_domain (uses storage.azure.com fallback)...")
        fs = adlfs.AzureBlobFileSystem(account_name="onelake", credential=credential)

    print(f"[3/4] Listing workspace root: {WORKSPACE}/")
    try:
        entries = fs.ls(WORKSPACE, detail=False)
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        print("  Likely cause: signed-in user lacks read access on the Lakehouse,")
        print("  OR adlfs + OneLake aren't talking. Check 'az account show' first.")
        return 1
    print(f"  OK — found {len(entries)} entries")
    for e in entries[:5]:
        print(f"    {e}")

    print(f"[4/4] Reading parquet schema + filtered rows from {SOURCE}")
    t0 = time.perf_counter()
    schema = pq.read_schema(SOURCE, filesystem=fs)
    t_schema = time.perf_counter() - t0
    print(f"  Schema read in {t_schema:.1f}s; {len(schema.names)} columns")

    t0 = time.perf_counter()
    table = pq.read_table(
        SOURCE,
        filesystem=fs,
        columns=["FlightDate", "Origin", "Dest", "CRSDepTime", "DepDelay", "ArrDelay"],
        filters=[
            ("Month", "=", 5),
            ("DayofMonth", "=", 20),
            ("CRSDepTime", ">=", 1400),
            ("CRSDepTime", "<", 1500),
        ],
    )
    t_read = time.perf_counter() - t0
    print(f"  Filtered read in {t_read:.1f}s; got {len(table)} rows")

    if len(table) == 0:
        print(
            "  WARN: 0 rows matched the test filter. "
            "Not necessarily a failure — try a different date/hour."
        )
    else:
        print("  Sample row:")
        print(f"    {table.to_pandas().iloc[0].to_dict()}")

    print()
    print("OK — adlfs + OneLake pattern works. Safe to deploy GetFlightData as-is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
