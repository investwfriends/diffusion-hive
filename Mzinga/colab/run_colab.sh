#!/usr/bin/env bash
set -euo pipefail

# ── Mzinga AlphaZero — Colab CLI runner ────────────────────────────────
# Runs AlphaZero training on a Colab GPU from your local terminal.
#
# USAGE:
#   ./run_colab.sh setup     — Install VS Code Colab extension + deps
#   ./run_colab.sh open      — Open training notebook in VS Code
#   ./run_colab.sh monitor   — Watch training via W&B dashboard
#   ./run_colab.sh sync      — Two-way sync Drive folder ↔ colab/drive/
#   ./run_colab.sh push      — Upload local files to Drive
#   ./run_colab.sh pull      — Download Drive files locally
#   ./run_colab.sh stop      — Stop Colab runtime
#   ./run_colab.sh status    — Show runtime info
#
# PREREQUISITES:
#   - VS Code installed
#   - "Google Colab" VS Code extension (install via: code --install-extension google.colab)
#   - Google account with Colab access
#   - W&B account (free at wandb.ai)
# ────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK="$SCRIPT_DIR/mzinga_alphazero_colab.ipynb"
DRIVE_FOLDER_ID="1zeIsILCDptKzZhTCQfkX3G8ZJ2LRbyUt"
DRIVE_SYNC_DIR="$SCRIPT_DIR/drive"

# ── helpers ────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[info]${NC}  $*"; }
ok()      { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()     { echo -e "${RED}[err]${NC}   $*"; }

step()    { echo -e "\n${GREEN}==>${NC} ${YELLOW}$*${NC}"; }

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        err "$1 not found — install it first"
        return 1
    fi
    ok "$1 found: $($1 --version 2>/dev/null | head -1 || echo 'ok')"
    return 0
}

# ── commands ───────────────────────────────────────────────────────────

cmd_setup() {
    step "Setting up Colab CLI environment"

    check_cmd code

    if ! code --list-extensions 2>/dev/null | grep -q "google.colab"; then
        info "Installing Google Colab VS Code extension..."
        code --install-extension google.colab
        ok "Colab extension installed"
    else
        ok "Colab extension already installed"
    fi

    check_cmd uv

    if ! uv run python -c "import wandb" 2>/dev/null; then
        info "Installing wandb locally..."
        uv pip install wandb >/dev/null 2>&1
        ok "wandb installed"
    else
        ok "wandb already installed"
    fi

    if [[ ! -f "$NOTEBOOK" ]]; then
        err "Notebook not found at $NOTEBOOK"
        return 1
    fi

    ok "Setup complete!"
    echo ""
    info "Next: run './run_colab.sh open' to open the notebook in VS Code"
    info "      run './run_colab.sh monitor' to watch training via W&B"
}

cmd_open() {
    step "Opening Colab notebook in VS Code"

    if [[ ! -f "$NOTEBOOK" ]]; then
        err "Notebook not found: $NOTEBOOK"
        return 1
    fi

    info "Opening $NOTEBOOK"
    code "$NOTEBOOK"

    echo ""
    info "In VS Code:"
    info "  1. Click 'Select Kernel' (top-right)"
    info "  2. Choose 'Colab' and pick a GPU runtime (T4, V100, or A100)"
    info "  3. Run cells top-to-bottom:"
    info "     • 'Install' cell — installs wandb"
    info "     • 'Upload' cell — uploads mzinga_colab.zip"
    info "     • 'Drive' cell — mounts Google Drive"
    info "     • 'W&B Login' cell — paste your API key"
    info "     • 'Train' cell — starts AlphaZero training"
    echo ""
    info "Output streams to your VS Code terminal — no browser needed."
    info "Run './run_colab.sh monitor' from a separate terminal to watch W&B."
}

cmd_monitor() {
    step "Opening W&B dashboard"

    if ! uv run python -c "import wandb" 2>/dev/null; then
        err "wandb not installed locally — run './run_colab.sh setup' first"
        return 1
    fi

    info "Opening W&B project dashboard..."
    uv run wandb open

    info "Or visit: https://wandb.ai/<your-entity>/mzinga-alphazero"

    echo ""
    info "From CLI, you can tail metrics with:"
    info "  uv run wandb status"
    info "  uv run wandb sync <run-path>  # download run data"

    # If there's a running W&B run, show its status
    uv run python -c "
import wandb
try:
    api = wandb.Api()
    runs = list(api.runs('mzinga-alphazero'))
    active = [r for r in runs if r.state == 'running']
    if active:
        r = active[0]
        print(f'\nLive run: {r.name}')
        print(f'  Runtime: {(r.summary.get(\"elapsed_s\", 0) / 60):.0f}m')
        if 'win_rate' in r.summary:
            print(f'  Win rate: {r.summary[\"win_rate\"]:.1%}')
    else:
        print('\nNo active runs found.')
except Exception as e:
    print(f'\nCould not check runs: {e}')
" 2>/dev/null || true
}

cmd_sync() {
    step "Two-way syncing Drive folder ↔ colab/drive/"

    if ! command -v rclone &>/dev/null; then
        err "rclone not installed. Install with: brew install rclone"
        err "Then run: rclone config  (n → gdrive → drive → defaults → OAuth in browser)"
        return 1
    fi

    mkdir -p "$DRIVE_SYNC_DIR"

    info "Syncing $DRIVE_SYNC_DIR ↔ Drive:$DRIVE_FOLDER_ID"
    rclone bisync "gdrive:" "$DRIVE_SYNC_DIR" \
        --drive-root-folder-id "$DRIVE_FOLDER_ID" \
        --resync \
        --progress \
        --exclude "*.zip" \
        2>&1

    ok "Sync complete"
    ls -lh "$DRIVE_SYNC_DIR/"
}

cmd_push() {
    step "Uploading local files to Drive"

    if ! command -v rclone &>/dev/null; then
        err "rclone not installed. Install with: brew install rclone"
        return 1
    fi

    rclone copy "$DRIVE_SYNC_DIR" "gdrive:" \
        --drive-root-folder-id "$DRIVE_FOLDER_ID" \
        --progress \
        2>&1

    ok "Upload complete"
}

cmd_pull() {
    step "Downloading Drive files locally"

    if ! command -v rclone &>/dev/null; then
        err "rclone not installed."
        return 1
    fi

    mkdir -p "$DRIVE_SYNC_DIR"
    rclone copy "gdrive:" "$DRIVE_SYNC_DIR" \
        --drive-root-folder-id "$DRIVE_FOLDER_ID" \
        --progress \
        --exclude "*.zip" \
        2>&1

    ok "Download complete"
    ls -lh "$DRIVE_SYNC_DIR/"
}

cmd_stop() {
    step "Stopping Colab runtime"

    warn "Colab runtimes auto-stop when idle for ~90 min."
    warn "To force-stop, close the VS Code window running the notebook,"
    warn "or go to https://colab.research.google.com/ and click 'Stop' on the active runtime."
}

cmd_status() {
    step "Colab Runtime Status"

    info "GPU types available: T4 (free), V100/A100 (Pro/Pay-as-you-go)"
    info "Session limits: Free ~12h, Pro ~24h, Pay-as-you-go: none"
    echo ""

    info "Check active runtimes: https://colab.research.google.com/"
    echo ""

    # Check W&B for active runs
    uv run python -c "
import wandb, time
try:
    api = wandb.Api()
    runs = list(api.runs('mzinga-alphazero'))
    active = [r for r in runs if r.state == 'running']
    done = [r for r in runs if r.state in ('finished', 'crashed')]
    if active:
        print(f'Active runs: {len(active)}')
        for r in active[:3]:
            eta = r.summary.get('eta_s', 0)
            print(f'  {r.name}: {r.summary.get(\"iteration\",\"?\")}/{r.config.get(\"n_iterations\",\"?\")} iters, ETA {eta/60:.0f}m')
    if done:
        print(f'Finished runs: {len(done)} (latest: {done[0].name})')
    if not active and not done:
        print('No W&B runs found.')
except Exception as e:
    print(f'Cannot reach W&B: {e}')
" 2>/dev/null || true
}

# ── main ───────────────────────────────────────────────────────────────

case "${1:-help}" in
    setup)   cmd_setup ;;
    open)    cmd_open ;;
    monitor) cmd_monitor ;;
    sync)    cmd_sync ;;
    push)    cmd_push ;;
    pull)    cmd_pull ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    *)
        echo "Mzinga AlphaZero — Colab CLI"
        echo ""
        echo "Usage: ./run_colab.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup     Install VS Code extension + local deps (wandb, rclone)"
        echo "  open      Open training notebook in VS Code with Colab kernel"
        echo "  monitor   Open W&B dashboard to watch live training"
        echo "  sync      Two-way sync Drive folder ↔ colab/drive/"
        echo "  push      Upload local files to Drive"
        echo "  pull      Download Drive files to local"
        echo "  stop      Instructions to stop the runtime"
        echo "  status    Show runtime info + active W&B runs"
        ;;
esac
