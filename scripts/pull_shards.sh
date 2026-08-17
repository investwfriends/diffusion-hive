#!/usr/bin/env bash
# pull_shards.sh — Minimal rsync-only puller.
#
# Each cycle: spawn ONE rsync (= one SSH) that pulls all new shards
# and removes them from remote after verified transfer. rsync's
# --ignore-existing skips files we already have. We bound each cycle
# with gtimeout so a wedged sshd can't freeze the puller.
#
# Stops when total local shard bytes >= TARGET_BYTES; signals the
# remote generator to stop.
#
# Env overrides:
#   REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_SHARD_DIR,
#   LOCAL_DIR, TARGET_BYTES, POLL_SECONDS

set -uo pipefail

REMOTE_HOST="${REMOTE_HOST:-98.191.113.4}"
REMOTE_PORT="${REMOTE_PORT:-5920}"
REMOTE_USER="${REMOTE_USER:-9e86f08b-b084-4aa4-a8f3-effd1a601513}"
REMOTE_SHARD_DIR="${REMOTE_SHARD_DIR:-/DiffusionHive/shards}"
LOCAL_DIR="${LOCAL_DIR:-/Users/beshir.aissi/Desktop/Random/DiffusionHive/data/cloud_shards}"
TARGET_BYTES="${TARGET_BYTES:-5368709120}"
POLL_SECONDS="${POLL_SECONDS:-20}"
RSYNC_TIMEOUT="${RSYNC_TIMEOUT:-90}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10
          -o BatchMode=yes
          -o ServerAliveInterval=10 -o ServerAliveCountMax=2
          -o ControlMaster=no -o ControlPath=none)

mkdir -p "$LOCAL_DIR"
LOG="$LOCAL_DIR/pull.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&2; }

local_bytes() {
  find "$LOCAL_DIR" -maxdepth 1 -name 'shard_*.pt' -type f -print0 2>/dev/null \
    | xargs -0 stat -f%z 2>/dev/null | awk '{s+=$1} END {print s+0}'
}

fmt_bytes() {
  python3 -c "n=int('$1' or 0);
print(f'{n/1024**3:.2f} GB' if n>=1024**3 else f'{n/1024**2:.1f} MB' if n>=1024**2 else f'{n/1024:.1f} KB' if n>0 else '0 B')"
}

do_pull() {
  # One rsync per cycle. If remote path has no matches, rsync exits 23 (fine).
  gtimeout "$RSYNC_TIMEOUT" rsync -az --ignore-existing --remove-source-files --partial \
    -e "ssh -p $REMOTE_PORT ${SSH_OPTS[*]}" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SHARD_DIR/shard_*.pt" \
    "$LOCAL_DIR/" 2>>"$LOG"
  return $?
}

log "pull_shards (rsync-only) start → $LOCAL_DIR (target $(fmt_bytes "$TARGET_BYTES"))"

while true; do
  have=$(local_bytes)
  nshards=$(find "$LOCAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l | tr -d ' ')

  if [ "$have" -ge "$TARGET_BYTES" ]; then
    log "TARGET reached ($(fmt_bytes "$have")) — stopping + signaling remote generator"
    gtimeout 30 ssh -n "${SSH_OPTS[@]}" -p "$REMOTE_PORT" \
        "$REMOTE_USER@$REMOTE_HOST" \
        'pkill -f "python3.12.*gen_data"; true' 2>/dev/null
    exit 0
  fi

  log "local: $(fmt_bytes "$have") / $(fmt_bytes "$TARGET_BYTES") ($nshards shards) — rsync cycle"

  do_pull
  rc=$?
  if [ "$rc" = "0" ] || [ "$rc" = "23" ] || [ "$rc" = "124" ]; then
    new_total=$(local_bytes)
    new_count=$(find "$LOCAL_DIR" -maxdepth 1 -name 'shard_*.pt' 2>/dev/null | wc -l | tr -d ' ')
    grow_bytes=$((new_total - have))
    log "  rsync ok: $(fmt_bytes "$new_total") local ($new_count shards, +$(fmt_bytes "$grow_bytes"))"
  else
    log "  rsync rc=$rc (will retry next cycle)"
  fi

  sleep "$POLL_SECONDS"
done