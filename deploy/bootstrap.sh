#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — one-time Azure setup for flight-delay-propagation
#
# Prerequisites:
#   - az CLI installed and logged in (az login)
#   - Azure ML extension: az extension add -n ml
#   - Azure CLI variables below filled in
#   - Microsoft Fabric workspace created manually in the Fabric portal
#   - OneLake ADLS endpoint available (shown in Fabric workspace settings)
#
# Usage:
#   chmod +x deploy/bootstrap.sh
#   ./deploy/bootstrap.sh
# =============================================================================
set -euo pipefail

# ── Configuration — fill these in before running ────────────────────────────
AML_RESOURCE_GROUP="flight-delay-rg"
AML_WORKSPACE="flight-delay-aml"
AML_LOCATION="eastus"

ONELAKE_ACCOUNT="<your-onelake-account>.dfs.fabric.microsoft.com"
ONELAKE_CONTAINER="<your-workspace-name>"          # Fabric workspace name
ONELAKE_LAKEHOUSE="<your-lakehouse-name>.Lakehouse" # Lakehouse name in Fabric

CHECKPOINT_PATH="outputs/best_seq2seq_gnn.pt"      # Local training checkpoint
AIRPORT_MAP_PATH="outputs/airport_map.json"         # Saved from training run
FEATURE_STATS_PATH="outputs/feature_stats.pt"       # Saved from training run
HISTORICAL_DATA_DIR="data/raw"                      # Local raw parquet files
# ── End configuration ────────────────────────────────────────────────────────

echo "=== [1/6] Creating Azure ML workspace (if it doesn't exist) ==="
az ml workspace create \
  --name "$AML_WORKSPACE" \
  --resource-group "$AML_RESOURCE_GROUP" \
  --location "$AML_LOCATION" \
  --no-wait || echo "Workspace already exists, skipping."

echo "=== [2/6] Creating GPU compute cluster ==="
az ml compute create \
  --file aml/compute/training_cluster.yml \
  --resource-group "$AML_RESOURCE_GROUP" \
  --workspace-name "$AML_WORKSPACE" || echo "Cluster already exists, skipping."

echo "=== [3/6] Building and registering AzureML environment ==="
echo "  This takes ~20 minutes on first build. Subsequent runs reuse the image."
az ml environment create \
  --file aml/environments/flight-delay-prod.yml \
  --resource-group "$AML_RESOURCE_GROUP" \
  --workspace-name "$AML_WORKSPACE"

echo "=== [4/6] Preparing inference checkpoint (stripping optimizer state) ==="
python deploy/prep_inference_checkpoint.py \
  --input "$CHECKPOINT_PATH" \
  --output "outputs/best_seq2seq_gnn_inference.pt" \
  --airport-map "$AIRPORT_MAP_PATH" \
  --metadata "outputs/metadata.json"

echo "=== [5/6] Uploading data and model artifacts to OneLake ==="
BASE="https://${ONELAKE_ACCOUNT}/${ONELAKE_CONTAINER}/${ONELAKE_LAKEHOUSE}/Files"

# Upload 2022 historical data (source of truth for the fake ingestion pipeline).
echo "  Uploading Combined_Flights_2022.parquet ..."
az storage blob upload \
  --account-name "${ONELAKE_ACCOUNT%%.*}" \
  --container-name "${ONELAKE_CONTAINER}" \
  --name "${ONELAKE_LAKEHOUSE}/Files/raw/historical_2022/Combined_Flights_2022.parquet" \
  --file "${HISTORICAL_DATA_DIR}/Combined_Flights_2022.parquet" \
  --auth-mode login

# Upload 2018-2021 historical data for training.
for YEAR in 2018 2019 2020 2021; do
  FILE="${HISTORICAL_DATA_DIR}/Combined_Flights_${YEAR}.parquet"
  if [ -f "$FILE" ]; then
    echo "  Uploading Combined_Flights_${YEAR}.parquet ..."
    az storage blob upload \
      --account-name "${ONELAKE_ACCOUNT%%.*}" \
      --container-name "${ONELAKE_CONTAINER}" \
      --name "${ONELAKE_LAKEHOUSE}/Files/raw/historical_2018_2021/Combined_Flights_${YEAR}.parquet" \
      --file "$FILE" \
      --auth-mode login
  else
    echo "  WARNING: ${FILE} not found, skipping."
  fi
done

# Upload champion model artifacts (all 4 files atomically via a staging path).
echo "  Uploading champion model artifacts ..."
for FILE in \
    "outputs/best_seq2seq_gnn_inference.pt:models/champion/best_seq2seq_gnn_inference.pt" \
    "${AIRPORT_MAP_PATH}:models/champion/airport_map.json" \
    "${FEATURE_STATS_PATH}:models/champion/feature_stats.pt" \
    "outputs/metadata.json:models/champion/metadata.json"; do
  LOCAL="${FILE%%:*}"
  REMOTE="${FILE##*:}"
  if [ -f "$LOCAL" ]; then
    az storage blob upload \
      --account-name "${ONELAKE_ACCOUNT%%.*}" \
      --container-name "${ONELAKE_CONTAINER}" \
      --name "${ONELAKE_LAKEHOUSE}/Files/${REMOTE}" \
      --file "$LOCAL" \
      --auth-mode login
  else
    echo "  WARNING: ${LOCAL} not found, skipping."
  fi
done

echo "=== [6/6] Backfilling 7-day rolling buffer ==="
echo "  This will run nb_ingest_live_feed 168 times to populate lag features."
echo "  Run this step manually in Fabric after importing the notebooks:"
echo ""
echo "    In Fabric, open nb_ingest_live_feed.ipynb and run:"
echo "    for i in range(168, 0, -1):"
echo "        ts = pd.Timestamp.utcnow().floor('h') - pd.Timedelta(hours=i)"
echo "        run_notebook(backfill_ts=ts)"
echo ""
echo "  Or trigger the pipeline pl_fake_ingestion manually 168 times."
echo ""
echo "=== Bootstrap complete ==="
echo ""
echo "Next steps:"
echo "  1. Import fabric/notebooks/*.ipynb into your Fabric workspace."
echo "  2. Import fabric/pipelines/*.json into Fabric Data Factory."
echo "  3. Update ADLS paths in the notebooks to match your workspace."
echo "  4. Enable the pl_fake_ingestion and pl_hourly_predict pipeline triggers."
echo "  5. Verify predictions/latest Delta table is populated after the first hour."
echo "  6. Connect Power BI to predictions/latest via DirectLake."
