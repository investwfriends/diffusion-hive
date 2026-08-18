#!/usr/bin/env bash
# Persist QUICKPOD_API_KEY (auto-injected in CI / Codespaces) to ~/.quickpod.json,
# the file the quickpod skill/plugin reads. Run once at setup time.
set -euo pipefail
KEY="${QUICKPOD_API_KEY:-}"
if [ -z "$KEY" ] && [ -f "$HOME/.quickpod.json" ]; then
  echo "~/.quickpod.json already exists — nothing to do."
  exit 0
fi
if [ -z "$KEY" ]; then
  echo "ERROR: QUICKPOD_API_KEY is not set and ~/.quickpod.json is missing." >&2
  echo "  CI/Codespaces: the repo secret is auto-injected as QUICKPOD_API_KEY." >&2
  echo "  Fresh machine:  export QUICKPOD_API_KEY=qpk_... first (or ask the user)." >&2
  exit 1
fi
umask 077
python3 - "$KEY" <<'PY'
import json, os, sys
p = os.path.expanduser("~/.quickpod.json")
try:
    with open(p) as f:
        d = json.load(f)
except Exception:
    d = {}
d["baseUrl"] = "https://api.quickpod.org"
d["token"] = sys.argv[1]
with open(p, "w") as f:
    json.dump(d, f, indent=2)
PY
chmod 600 "$HOME/.quickpod.json"
echo "OK: QUICKPOD_API_KEY persisted to $HOME/.quickpod.json (mode 600)"
