#!/usr/bin/env python3
"""DEPRECATED — use ``python -m ghive_diffusion_lite.pipeline`` instead.

This script only ran on-the-fly Mzinga self-play training without eval,
self-play follow-up, or mid-run best-model checkpoints.

Migration
---------
Old::

    python ghive_diffusion_lite/lite_train.py

New (full train → eval → self-play)::

    PYTHONPATH="/path/to/DiffusionHive" \\
      /path/to/Mzinga/.venv/bin/python -m ghive_diffusion_lite.pipeline \\
        --dataset ghive_diffusion_lite/dataset_merged.pt \\
        --device mps \\
        --out-dir runs/lite_run1

On-the-fly Mzinga training (no pre-generated dataset) is still available
via ``ghive_diffusion_lite.train_lite.train_lite`` for experiments, but
the supported product path is the pipeline.
"""

from __future__ import annotations

import os
import sys
import warnings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEPRECATION = (
    "ghive_diffusion_lite.lite_train is DEPRECATED. "
    "Use: python -m ghive_diffusion_lite.pipeline "
    "(train → eval → self-play with best-model checkpoints)."
)


if __name__ == "__main__":
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=1)
    print("=" * 72)
    print("  DEPRECATED ENTRY POINT")
    print("=" * 72)
    print()
    print("  lite_train.py is deprecated.")
    print("  Use the new pipeline instead:")
    print()
    print("    PYTHONPATH=\".../DiffusionHive\" \\")
    print("      .../Mzinga/.venv/bin/python -m ghive_diffusion_lite.pipeline \\")
    print("        --dataset ghive_diffusion_lite/dataset_merged.pt \\")
    print("        --device mps --out-dir runs/lite_run1")
    print()
    print("  Flags: --only train|eval|selfplay  --skip-train  --start-from eval")
    print("  Help:  python -m ghive_diffusion_lite.pipeline --help")
    print()
    # Still allow the old behaviour if the user insists via env var
    if os.environ.get("GHIVE_ALLOW_LEGACY_LITE_TRAIN") == "1":
        print("  GHIVE_ALLOW_LEGACY_LITE_TRAIN=1 set — running legacy train_lite()...")
        from ghive_diffusion_lite.train_lite import train_lite
        train_lite(
            total_steps=2000,
            lr=1e-4,
            warmup_steps=200,
            min_lr=1e-5,
            log_interval=200,
            save_path="lite_model.pt",
            use_mz_ai=True,
            mz_simulations=50,
            asymmetric=True,
            device_str="cpu",
        )
    else:
        print("  (exiting; set GHIVE_ALLOW_LEGACY_LITE_TRAIN=1 to force legacy run)")
        sys.exit(2)
