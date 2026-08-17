"""Smoke check for the ghive_diffusion package.

Builds a smoke model, runs a forward pass, and prints parameter count
plus output shape. Run from anywhere with::

    PYTHONPATH=/path/to/DiffusionHive \
    /path/to/.venv/bin/python sanity_check.py
"""

import torch

from ghive_diffusion import (
    HiveDiffusionModel,
    HiveSmokeConfig,
)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = HiveSmokeConfig()
    model = HiveDiffusionModel(cfg).eval()
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        h, kv = model.forward_encoder(ids, use_cache=True)
        canvas = torch.randint(0, cfg.vocab_size, (1, cfg.canvas_length))
        dec_h = model.forward_decoder(canvas, kv)
        logits = model._lm_head(dec_h)
    print("encoder out shape:", tuple(h.shape))
    print("decoder out shape:", tuple(dec_h.shape))
    print("logits shape:", tuple(logits.shape))
