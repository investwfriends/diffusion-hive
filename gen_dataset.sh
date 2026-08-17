#!/usr/bin/env bash
# gen_dataset.sh — Generate a training dataset for ghive_diffusion_lite.
#
# Runs identically on the local Mac and on a Linux cloud VM: the Python
# interpreter is auto-detected, everything else is driven by env vars.
#
# Usage:
#   bash gen_dataset.sh                      # defaults (~100K samples)
#   GAMES=4000 bash gen_dataset.sh           # bigger run
#   HISTORY_WINDOW=0 bash gen_dataset.sh     # full transcript (old behaviour)
#   OUTPUT=data/foo.pt bash gen_dataset.sh
#
# Run it detached on a VM:
#   nohup bash gen_dataset.sh > gen.log 2>&1 &
#   tail -f gen.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# ── Tunables (override via env) ────────────────────────────────────
#
# GAMES: sample yield depends on game length, which depends on teacher
# strength. Expect roughly 60-90 samples/game with the defaults below,
# so ~1500 games lands near 100K samples. Watch the progress output and
# adjust; TARGET_BYTES gives a hard stop regardless.
GAMES="${GAMES:-1500}"
OUTPUT="${OUTPUT:-data/dataset.pt}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-}"              # empty = auto from CPU/RAM

# Teacher: native Mzinga C# engine over UHP.
TEACHER="${TEACHER:-uhp}"
UHP_DEPTH="${UHP_DEPTH:-4}"
UHP_EPSILON="${UHP_EPSILON:-0.05}"

# HISTORY_WINDOW caps the <history> section to the last N moves.
# This is the single biggest lever on training cost: the full transcript
# grows contexts to ~2100 tokens by the midgame and attention is O(T^2).
# Measured: 40 moves gives ~850-token contexts (mean), i.e. ~2.5x shorter
# and roughly 4-5x cheaper per step. The floor is set by the <features>
# and <legal> sections, and <legal> scales with the branching factor, so
# shrinking this below ~30 buys little.
# Contexts are stored PRE-TOKENIZED, so this is baked in at generation
# time and cannot be changed later without regenerating. Set to 0 for
# the full transcript.
HISTORY_WINDOW="${HISTORY_WINDOW:-40}"

# Decisiveness. Unfinished games get no terminal outcome backfill, so
# their samples carry only the teacher's weak root value — that is why
# the previous dataset had has_outcome=False on every sample.
MAX_PLIES="${MAX_PLIES:-300}"
DROP_UNFINISHED="${DROP_UNFINISHED:-1}"

# Opponent blend: fraction of games played vs a random opponent.
# vs-random games are short and decisive (good value signal); self-play
# games are stronger positionally. 0.5 keeps both.
RANDOM_FRACTION="${RANDOM_FRACTION:-0.5}"
SCRAMBLE_PLIES="${SCRAMBLE_PLIES:-6}"

REPORT_INTERVAL="${REPORT_INTERVAL:-60}"
TARGET_BYTES="${TARGET_BYTES:-0}"   # 0 = no byte cap

# ── Interpreter + PYTHONPATH ───────────────────────────────────────
if [ -n "${PYTHON:-}" ]; then
    :
elif [ -x "$REPO_ROOT/Mzinga/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/Mzinga/.venv/bin/python"   # local Mac (uv venv)
elif command -v python3.12 &>/dev/null; then
    PYTHON="python3.12"                            # cloud VM
else
    PYTHON="python3"
fi
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/Mzinga/src"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$(dirname "$OUTPUT")"

# ── Assemble args ──────────────────────────────────────────────────
ARGS=(
    --games "$GAMES"
    --output "$OUTPUT"
    --seed "$SEED"
    --teacher "$TEACHER"
    --uhp-depth "$UHP_DEPTH"
    --uhp-epsilon "$UHP_EPSILON"
    --random-fraction "$RANDOM_FRACTION"
    --scramble-plies "$SCRAMBLE_PLIES"
    --max-plies "$MAX_PLIES"
    --report-interval "$REPORT_INTERVAL"
)
[ -n "$WORKERS" ] && ARGS+=(--workers "$WORKERS")
[ "$HISTORY_WINDOW" != "0" ] && ARGS+=(--history-window "$HISTORY_WINDOW")
[ "$DROP_UNFINISHED" = "1" ] && ARGS+=(--drop-unfinished)
[ "$TARGET_BYTES" != "0" ] && ARGS+=(--target-bytes "$TARGET_BYTES")

echo "=== gen_dataset ==="
echo "  python          $PYTHON ($($PYTHON --version 2>&1))"
echo "  repo            $REPO_ROOT"
echo "  output          $OUTPUT"
echo "  games           $GAMES"
echo "  teacher         $TEACHER (depth $UHP_DEPTH)"
echo "  history window  ${HISTORY_WINDOW} $([ "$HISTORY_WINDOW" = "0" ] && echo '(full transcript)')"
echo "  max plies       $MAX_PLIES (drop unfinished: $DROP_UNFINISHED)"
echo "  random fraction $RANDOM_FRACTION"
echo ""

"$PYTHON" -u -m ghive_diffusion_lite.gen_data "${ARGS[@]}"

# ── Post-run report ────────────────────────────────────────────────
# Verifies the two properties that silently broke last time: contexts
# short enough to train on, and terminal outcomes actually present.
echo ""
echo "=== dataset report ==="
"$PYTHON" - "$OUTPUT" <<'PY'
import os, random, statistics, sys
import torch

path = sys.argv[1]
if not os.path.exists(path):
    print(f"  {path} not found (shards only?) — skipping report")
    raise SystemExit(0)

data = torch.load(path, weights_only=False)
n = len(data)
idx = random.sample(range(n), min(3000, n))
ctx = [len(data[i].context_ids) for i in idx]
nl = [len(data[i].legal_move_ids) for i in idx]
outc = sum(bool(getattr(data[i], "has_outcome", False)) for i in idx) / len(idx)
dec = sum(abs(float(data[i].value)) >= 0.9 for i in idx) / len(idx)

print(f"  samples          {n:,}")
print(f"  size on disk     {os.path.getsize(path)/1e9:.2f} GB "
      f"({os.path.getsize(path)/n:,.0f} B/sample)")
print(f"  context tokens   mean={statistics.mean(ctx):.0f} "
      f"p50={statistics.median(ctx):.0f} max={max(ctx)}")
print(f"  legal moves      mean={statistics.mean(nl):.0f} max={max(nl)}")
print(f"  has_outcome      {outc:.1%}")
print(f"  |value| >= 0.9   {dec:.1%}")
print()
if statistics.mean(ctx) > 900:
    print("  WARNING: contexts are long; training will be slow. Lower HISTORY_WINDOW.")
if outc < 0.5:
    print("  WARNING: few terminal outcomes — the value head will stay flat.")
PY

echo ""
echo "  Train with:"
echo "    PYTHONPATH=\"\$PWD:\$PWD/Mzinga/src\" $PYTHON \\"
echo "      -m ghive_diffusion_lite.pipeline \\"
echo "      --dataset $OUTPUT --device mps --out-dir runs/new_run"
