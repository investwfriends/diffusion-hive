#!/usr/bin/env bash
# pack_for_cloud.sh — Package DiffusionHive for cloud VM upload
#
# Creates a zip with all code + Mzinga checkpoint + UHP engine binary
# (no datasets, no .venv, no runs). Run this on your Mac, then upload
# the zip to the cloud VM.
#
# Usage:
#   cd /path/to/DiffusionHive
#   bash pack_for_cloud.sh
#
# Then upload:
#   sftp -P <port> <user>@<host> << 'EOF'
#   put DiffusionHive_cloud.zip /DiffusionHive_cloud.zip
#   EOF
#
# On the VM:
#   cd / && python3 -c "import zipfile; zipfile.ZipFile('DiffusionHive_cloud.zip').extractall('.')"
#   cd /DiffusionHive && bash setup_cloud.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ZIP_NAME="DiffusionHive_cloud.zip"
ZIP_PATH="$REPO_ROOT/$ZIP_NAME"

echo "=== Packing DiffusionHive for cloud ==="
echo "  Source: $REPO_ROOT"
echo "  Output: $ZIP_PATH"
echo ""

# Remove old zip
rm -f "$ZIP_PATH"

# Create zip with only the files needed on cloud.
# We zip in two passes:
#   1. Code files + UHP engine binary (exclude .pt, .venv, __pycache__, etc.)
#   2. The Mzinga AlphaZero checkpoint specifically (it's a .pt but must
#      be included for the legacy --teacher alphazero path).
cd "$REPO_ROOT"

# Pass 1: everything except .pt files
zip -r "$ZIP_NAME" \
    Mzinga/src \
    Mzinga/tests \
    Mzinga/colab \
    Mzinga/pyproject.toml \
    Mzinga/AGENTS.md \
    Mzinga/README.md \
    ghive_diffusion \
    ghive_diffusion_lite \
    gemma_diffusion \
    mzinga_uhp \
    sanity_check.py \
    setup_cloud.sh \
    pack_for_cloud.sh \
    gen_dataset.sh \
    README.md \
    AGENTS.md \
    -x "*__pycache__*" "*.pyc" "*.pytest_cache*" "*.pt" "*.DS_Store" "*.tar.gz" \
    2>/dev/null

# Pass 2: add the AlphaZero checkpoint (append to existing zip)
# Only needed if you want to use --teacher alphazero on the cloud VM.
# The UHP teacher (default) only needs the binary in mzinga_uhp/.
if [ -f "Mzinga/colab/mzinga_alphazero_final.pt" ]; then
    zip -g "$ZIP_NAME" Mzinga/colab/mzinga_alphazero_final.pt
fi

# Clean up any .venv entries that snuck in
zip -d "$ZIP_NAME" "Mzinga/.venv/*" 2>/dev/null || true

SIZE=$(ls -lh "$ZIP_PATH" | awk '{print $5}')
echo ""
echo "=== Done ==="
echo "  File: $ZIP_PATH ($SIZE)"
echo ""
echo "  Upload to cloud:"
echo "    sftp -P <port> <user>@<host> << 'EOF'"
echo "    put $ZIP_PATH /DiffusionHive_cloud.zip"
echo "    EOF"
echo ""
echo "  On the VM:"
echo "    cd / && python3 -c \"import zipfile; zipfile.ZipFile('DiffusionHive_cloud.zip').extractall('.')\""
echo "    cd /DiffusionHive && bash setup_cloud.sh"
echo "    nohup bash gen_dataset.sh > gen.log 2>&1 &"
echo ""
echo "  gen_dataset.sh is the single generation entrypoint and runs the"
echo "  same way locally. Tune it with env vars (GAMES, HISTORY_WINDOW,"
echo "  UHP_DEPTH, ...); see the header of the script."
