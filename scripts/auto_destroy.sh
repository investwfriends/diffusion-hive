#!/usr/bin/env bash
# auto_destroy.sh — Monitor local puller and destroy pod when target reached.
#
# Usage:
#   bash scripts/auto_destroy.sh <puller_pid> <pod_uuid> <api_key>

set -euo pipefail

PULLER_PID="$1"
POD_UUID="$2"
API_KEY="$3"
LOCAL_DIR="/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/cloud_shards"
LOG="$LOCAL_DIR/pull.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-DESTROY] $*" | tee -a "$LOG" >&2; }

log "Monitoring puller PID $PULLER_PID for pod $POD_UUID..."

while kill -0 "$PULLER_PID" 2>/dev/null; do
  sleep 30
done

log "Puller PID $PULLER_PID has exited. Checking local bytes..."

local_bytes=$(find "$LOCAL_DIR" -maxdepth 1 -name 'shard_*.pt' -type f -print0 2>/dev/null \
  | xargs -0 stat -f%z 2>/dev/null | awk '{s+=$1} END {print s+0}')

local_gb=$(python3 -c "print(f'{int($local_bytes)/1024**3:.2f}')")
log "Local shards total: $local_gb GB"

log "Calling QuickPod Destroy API for CPU pod $POD_UUID..."
RESPONSE=$(curl -s "https://api.quickpod.org/update/api/destroypod_cpu?pod_uuid=$POD_UUID" \
  -H "X-API-Key: $API_KEY")

log "Destroy API Response: $RESPONSE"
log "Auto-destroy complete."