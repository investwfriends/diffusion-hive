#!/usr/bin/env bash
# setup_cloud.sh — Prepare a FRESH Linux cloud VM for DiffusionHive
#
# This script does everything from scratch on a clean Ubuntu 22.04 VM:
#   1. Installs Python 3.12 (deadsnakes PPA)
#   2. Installs pip + torch (CPU) + numpy
#   3. Installs Mzinga (editable, stdlib-only — no gymnasium)
#   4. Patches mcts.py to remove gymnasium import
#   5. Downloads the Linux MzingaEngine binary for the UHP teacher
#   6. Verifies all imports + UHP adapter + checkpoint loading
#
# Usage (on the VM, after extracting DiffusionHive.zip):
#   cd DiffusionHive && bash setup_cloud.sh
#
# Prerequisites: apt-get access, sudo/root

set -euo pipefail

PYTHON=python3.12
PIP="$PYTHON -m pip"
MZINGA_VERSION="v0.16.0"

echo "=== DiffusionHive Cloud Setup (fresh VM) ==="
echo "  OS: $(cat /etc/os-release 2>/dev/null | head -1)"
echo "  Date: $(date)"
echo ""

# ── 1. Python 3.12 via deadsnakes PPA ──────────────────────────────
if ! command -v $PYTHON &>/dev/null; then
    echo "--- Installing Python 3.12 (deadsnakes PPA) ---"
    apt-get update -qq
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
    echo "  ✓ Python 3.12 installed"
else
    echo "--- Python 3.12 already installed: $($PYTHON --version) ---"
fi

# Ensure pip is available
$PYTHON -m pip --version 2>/dev/null || {
    echo "--- Installing pip ---"
    $PYTHON -m ensurepip --default-pip 2>/dev/null || {
        apt-get install -y -qq python3.12-distutils
        curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON
    }
}

# ── 2. Install torch (CPU-only) + numpy ────────────────────────────
echo ""
echo "--- Installing torch (CPU) + numpy ---"
$PIP install --quiet --break-system-packages \
    --index-url https://download.pytorch.org/whl/cpu torch
$PIP install --quiet --break-system-packages numpy
echo "  ✓ torch + numpy installed"

# ── 3. Install Mzinga (editable, stdlib-only) ──────────────────────
echo ""
echo "--- Installing Mzinga (editable, no gymnasium) ---"
$PIP install --quiet --break-system-packages -e ./Mzinga
echo "  ✓ Mzinga installed"

# ── 4. Patch mcts.py — remove gymnasium import ─────────────────────
MCTS_FILE="./Mzinga/src/mzinga/rl/mcts.py"
if grep -q "from mzinga.gym.hive_env import" "$MCTS_FILE" 2>/dev/null; then
    echo ""
    echo "--- Patching mcts.py (remove gymnasium dependency) ---"
    $PYTHON -c "
path = '$MCTS_FILE'
with open(path, 'r') as f:
    content = f.read()
old = 'from mzinga.gym.hive_env import BOARD_NORM, MAX_MOVES, OBS_DIM'
new = '# Patched: inline constants (no gymnasium dependency)\nBOARD_NORM = 16.0\nMAX_MOVES = 2048\nOBS_DIM = 88'
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('  ✓ mcts.py patched')
"
else
    echo "  ✓ mcts.py already patched (or no gymnasium import found)"
fi

# ── 5. Download the Linux MzingaEngine binary for the UHP teacher ──
# The packed zip includes the macOS arm64 binary (useless on Linux).
# Download the matching Linux binary from GitHub releases.
UHP_DIR="./mzinga_uhp"
LINUX_DIR="$UHP_DIR/Mzinga.LinuxX64"
ENGINE_BIN="$LINUX_DIR/MzingaEngine"
if [ ! -x "$ENGINE_BIN" ]; then
    echo ""
    echo "--- Downloading MzingaEngine for Linux x64 (${MZINGA_VERSION}) ---"
    mkdir -p "$UHP_DIR"
    TAR="$UHP_DIR/Mzinga.LinuxX64.tar.gz"
    curl -sL -o "$TAR" "https://github.com/jonthysell/Mzinga/releases/download/${MZINGA_VERSION}/Mzinga.LinuxX64.tar.gz"
    tar xzf "$TAR" -C "$UHP_DIR"
    rm -f "$TAR"
    # Remove the macOS binary that came in the zip (wrong platform)
    rm -rf "$UHP_DIR/Mzinga.MacOSArm64"
    chmod +x "$ENGINE_BIN"
    echo "  ✓ MzingaEngine installed at $ENGINE_BIN"
else
    echo "  ✓ MzingaEngine already present at $ENGINE_BIN"
fi

# ── 6. Verify everything works ─────────────────────────────────────
echo ""
echo "--- Verifying imports ---"
export PYTHONPATH="$(pwd):$(pwd)/Mzinga/src"
$PYTHON -c "
import torch
print(f'  ✓ torch {torch.__version__}')

from mzinga.core.board import Board
b = Board()
print(f'  ✓ Mzinga Board (turn={b.current_turn})')

from ghive_diffusion_lite import build_lite_model
m = build_lite_model()
n = sum(p.numel() for p in m.parameters())
print(f'  ✓ HiveLiteModel ({n:,} params)')

# UHP teacher (default, strong, no torch needed for the teacher itself)
from ghive_diffusion_lite.mzinga_uhp_adapter import MzingaUHPAdapter
adapter = MzingaUHPAdapter(depth=3, sample=False)
label, play, val = adapter.evaluate(b)
print(f'  ✓ UHP teacher adapter (label={label}, value={val:+.3f})')
adapter.close()

print()
print('  All systems go!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "  CPU cores: $(nproc)"
echo "  RAM: $(free -h | awk '/^Mem:/{print \$2}')"
echo ""
echo "  To generate data (UHP teacher, short contexts, decisive games):"
echo "    cd /DiffusionHive"
echo "    nohup bash gen_dataset.sh > gen.log 2>&1 &"
echo ""
echo "  gen_dataset.sh is the single generation entrypoint (same script"
echo "  runs locally). Tune with env vars, e.g.:"
echo "    GAMES=4000 UHP_DEPTH=4 bash gen_dataset.sh"
echo "    HISTORY_WINDOW=0 bash gen_dataset.sh   # full transcript"
echo ""
echo "  It prints a dataset report at the end — check that 'has_outcome'"
echo "  is high and 'context tokens' is well under 900."
echo ""
echo "  To check progress:"
echo "    tail -15 /DiffusionHive/gen.log"
echo ""
echo "  To kill and merge early:"
echo "    pkill -f gen_data"
echo "    PYTHONPATH=\"\$PWD:\$PWD/Mzinga/src\" python3.12 -c \\"
echo "      'import torch; d=torch.load(\"/DiffusionHive/dataset.pt\",weights_only=False); print(len(d),\"samples\")'"
