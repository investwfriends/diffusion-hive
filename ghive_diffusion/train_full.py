#!/usr/bin/env python3
"""Train the full HiveDiffusionModel (MoE) on a pre-generated dataset.

Usage (cloud VM)::

    export PYTHONPATH=/DiffusionHive:/DiffusionHive/Mzinga/src
    python3.12 -m ghive_diffusion.train_full \\
        --dataset /DiffusionHive/dataset.pt \\
        --out-dir /DiffusionHive/runs/smoke_full \\
        --steps 20000

Resume from checkpoint::

    python3.12 -m ghive_diffusion.train_full \\
        --dataset /DiffusionHive/dataset.pt \\
        --resume /DiffusionHive/runs/smoke_full/best_model.pt \\
        --out-dir /DiffusionHive/runs/smoke_full \\
        --steps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch

# ── Pretty printing ─────────────────────────────────────────────────────


def _banner(title: str) -> None:
    W = 72
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)
    print(flush=True)


def _kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<24s} {value}", flush=True)


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


# ── Checkpoint save/load ────────────────────────────────────────────────


def save_ckpt(
    path: str,
    model: torch.nn.Module,
    step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if metrics:
        ckpt["metrics"] = metrics
    torch.save(ckpt, path)


def load_ckpt(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> tuple[int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0), ckpt.get("metrics", {})


# ── Main training ───────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Train full HiveDiffusionModel (MoE)")
    p.add_argument("--dataset", required=True, help="path to .pt dataset")
    p.add_argument("--out-dir", required=True, help="output directory for checkpoints + metrics")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    p.add_argument("--sc-ramp-steps", type=int, default=500)
    p.add_argument("--self-condition-prob", type=float, default=0.5)
    p.add_argument("--diffusion-weight", type=float, default=1.0)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--moe-lb-weight", type=float, default=0.01)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    p.add_argument("--benchmark-only", action="store_true",
                   help="run 100 steps and report step/sec, then exit")
    args = p.parse_args()

    # ── Imports (deferred so argparse is fast) ──
    from ghive_diffusion.hive_config import HiveSmokeConfig
    from ghive_diffusion.hive_model import HiveDiffusionModel
    from ghive_diffusion.training import HiveTrainer
    from ghive_diffusion.tokenizer import build_default_tokenizer
    from ghive_diffusion.train_loop import create_scheduler

    os.makedirs(args.out_dir, exist_ok=True)

    _banner("HiveDiffusionModel (MoE) Training")
    _kv("dataset", args.dataset)
    _kv("device", args.device)
    _kv("steps", args.steps)
    _kv("lr / warmup / min_lr", f"{args.lr} / {args.warmup} / {args.min_lr}")
    _kv("max_grad_norm", args.max_grad_norm)
    _kv("out_dir", args.out_dir)
    print()

    # ── Load dataset ──
    print("  Loading dataset ...", end=" ", flush=True)
    t0 = time.time()
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    print(f"{len(data):,} samples ({_fmt_time(time.time() - t0)})", flush=True)

    # ── Build model ──
    print("  Building model ...", end=" ", flush=True)
    cfg = HiveSmokeConfig()
    model = HiveDiffusionModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} params")

    tokenizer = build_default_tokenizer(cfg)
    device = torch.device(args.device)

    start_step = 0
    if args.resume:
        print(f"  Resuming from {args.resume} ...", end=" ", flush=True)
        start_step, prev_metrics = load_ckpt(args.resume, model)
        print(f"step {start_step}")

    model = model.to(device)

    # ── Trainer ──
    trainer = HiveTrainer(
        model, tokenizer,
        diffusion_weight=args.diffusion_weight,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        aux_weight=args.aux_weight,
        moe_lb_weight=args.moe_lb_weight,
        self_condition_prob=args.self_condition_prob,
        sc_ramp_steps=args.sc_ramp_steps,
        diffusion_schedule="cosine",
        max_grad_norm=args.max_grad_norm,
        device=device,
    )
    trainer.step_count = start_step

    # ── Optimizer ──
    # Separate weight-decay groups (match train_loop.py pattern)
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    opt_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(opt_groups, lr=args.lr)

    if args.resume:
        load_ckpt(args.resume, model, opt)

    scheduler = create_scheduler(
        opt,
        warmup_steps=args.warmup,
        total_steps=args.steps,
        min_lr=args.min_lr,
    )
    for _ in range(start_step):
        scheduler.step()

    remaining = max(0, args.steps - start_step)
    if remaining == 0:
        print("  Already at target steps; nothing to train.")
        return

    # ── Benchmark mode ──
    if args.benchmark_only:
        print(f"\n  Benchmarking {min(100, remaining)} steps ...")
        t_bench = time.time()
        for i in range(min(100, remaining)):
            sample = data[i % len(data)]
            trainer.step(sample, optimizer=opt)
            scheduler.step()
        elapsed = time.time() - t_bench
        n_bench = min(100, remaining)
        sps = elapsed / n_bench
        print(f"  Done: {n_bench} steps in {_fmt_time(elapsed)}  "
              f"({sps:.2f}s/step  →  {_fmt_time(sps * args.steps)} for full run)")
        return

    # ── Training loop ──
    best_path = os.path.join(args.out_dir, "best_model.pt")
    final_path = os.path.join(args.out_dir, "final_model.pt")
    metrics_path = os.path.join(args.out_dir, "train_metrics.jsonl")
    best_policy_loss = float("inf")

    # Track running averages
    running: Dict[str, float] = {}
    loss_history: List[Dict] = []
    t_start = time.time()
    epoch = 0
    n_data = len(data)

    print(f"\n  Training {remaining} steps on {n_data:,} samples ...\n")
    print(f"  {'step':>7s}  {'loss':>7s}  {'diff':>7s}  "
          f"{'pol':>7s}  {'val':>7s}  {'moe':>7s}  "
          f"{'aux':>7s}  {'v_pred':>7s}  {'lr':>9s}  {'eta':>8s}")
    print("  " + "─" * 85)

    while trainer.step_count < args.steps:
        epoch += 1
        indices = list(range(n_data))
        random.shuffle(indices)
        for i in indices:
            if trainer.step_count >= args.steps:
                break

            sample = data[i]
            metrics = trainer.step(sample, optimizer=opt)
            scheduler.step()
            step = trainer.step_count

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + float(v)

            if step % args.log_interval == 0 and step > 0:
                n = max(1, args.log_interval)
                loss = running.get("loss", 0.0) / n
                diff = running.get("diffusion_loss", 0.0) / n
                pol = running.get("policy_loss", 0.0) / n
                val = running.get("value_loss", 0.0) / n
                moe = running.get("moe_loss", 0.0) / n
                aux = running.get("aux_loss", 0.0) / n

                elapsed = time.time() - t_start
                done = step - start_step
                eta_s = (elapsed / max(1, done)) * (args.steps - step) if done > 0 else 0
                lr_now = scheduler.get_last_lr()[0]
                sps = elapsed / max(1, done)

                star = "  *" if pol < best_policy_loss else ""
                print(
                    f"  {step:>7d}  {loss:>7.3f}  {diff:>7.3f}  "
                    f"{pol:>7.3f}  {val:>7.3f}  {moe:>7.3f}  "
                    f"{aux:>7.3f}  {metrics.get('value_pred', 0.0):>+7.3f}  "
                    f"{lr_now:>7.1e}  {_fmt_time(eta_s):>8s}"
                    f"{'  *' if pol < best_policy_loss else ''}",
                    flush=True,
                )

                row = {
                    "step": step,
                    "loss": loss, "diffusion_loss": diff,
                    "policy_loss": pol, "value_loss": val,
                    "moe_loss": moe, "aux_loss": aux,
                    "value_pred": metrics.get("value_pred", 0.0),
                    "lr": lr_now, "step_per_sec": 1.0 / max(sps, 1e-9),
                }
                loss_history.append(row)

                with open(metrics_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                running = {}

                if pol < best_policy_loss:
                    best_policy_loss = pol
                    save_ckpt(best_path, model, step, opt,
                              metrics={"policy_loss": pol, "diffusion_loss": diff,
                                       "value_loss": val, "step": step})
                    print(f"           ↳ saved best → {best_path}  (pol={pol:.4f})",
                          flush=True)

            if step % args.ckpt_interval == 0 and step > 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step:06d}.pt")
                save_ckpt(ckpt_path, model, step, opt)
                print(f"           ↳ checkpoint → {ckpt_path}", flush=True)

    # ── Save final ──
    save_ckpt(final_path, model, trainer.step_count, opt)
    print(f"\n  ✓ Final model saved → {final_path}")

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n  Total: {trainer.step_count} steps in {_fmt_time(elapsed)}")
    if loss_history:
        final = loss_history[-1]
        print(f"  Final  loss={final['loss']:.4f}  "
              f"diff={final['diffusion_loss']:.4f}  "
              f"pol={final['policy_loss']:.4f}  "
              f"val={final['value_loss']:.4f}  "
              f"moe={final['moe_loss']:.4f}")
    print(f"  Best policy loss: {best_policy_loss:.4f}")


if __name__ == "__main__":
    main()
