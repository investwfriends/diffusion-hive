#!/usr/bin/env python3
"""End-to-end HiveLite pipeline: train → eval → self-play.

This is the **preferred** entry point for ``ghive_diffusion_lite``.
Older scripts (``lite_train.py``, bare ``train_from_dataset``) are
deprecated; use this CLI instead.

Pipeline steps
--------------
1. **train**     — offline training on a pre-generated dataset.
                   Saves periodic checkpoints + best-by-policy-loss model.
2. **eval**      — opening-move diagnostics + games vs random.
3. **selfplay**  — mixed self-play (model vs random / optional teacher),
                   then optional fine-tune on the new samples.

Each step prints clear progress, approximate success goals, and a pass/fail
summary. Steps can be skipped or started mid-pipeline via CLI flags.

Examples
--------
Full pipeline on an M4 Mac::

    PYTHONPATH="/path/to/DiffusionHive" \\
      /path/to/Mzinga/.venv/bin/python -m ghive_diffusion_lite.pipeline \\
        --dataset ghive_diffusion_lite/dataset_merged.pt \\
        --device mps \\
        --out-dir runs/lite_run1

Skip training (use an existing checkpoint), eval + self-play only::

    python -m ghive_diffusion_lite.pipeline \\
        --skip-train --checkpoint runs/lite_run1/best_model.pt \\
        --device mps --out-dir runs/lite_run1

Eval only::

    python -m ghive_diffusion_lite.pipeline --only eval \\
        --checkpoint runs/lite_run1/best_model.pt --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pretty CLI helpers
# ---------------------------------------------------------------------------

_W = 72


def _banner(title: str) -> None:
    print()
    print("=" * _W)
    print(f"  {title}")
    print("=" * _W)
    print(flush=True)


def _section(title: str) -> None:
    print()
    print("─" * _W)
    print(f"  {title}")
    print("─" * _W, flush=True)


def _kv(key: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{key:<22s} {value}", flush=True)


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "?"
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _fmt_sec(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s // 60:.0f}m {s % 60:.0f}s"
    return f"{s // 3600:.0f}h {(s % 3600) // 60:.0f}m"


def _goal_line(name: str, ok: Optional[bool], detail: str) -> None:
    if ok is True:
        mark = "PASS"
    elif ok is False:
        mark = "FAIL"
    else:
        mark = "----"
    print(f"  [{mark}] {name}: {detail}", flush=True)


def _resolve_device(device_str: str) -> torch.device:
    ds = device_str.lower()
    if ds == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if ds == "mps" and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    if ds == "cpu":
        return torch.device("cpu")
    # Auto / requested but unavailable
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Goals (approximate success criteria printed each step)
# ---------------------------------------------------------------------------

@dataclass
class TrainGoals:
    """Approximate goals for the offline training step."""
    policy_loss: float = 3.0      # uniform over ~52 legal moves is ln(52)≈3.95
    diffusion_loss: float = 1.0
    value_separation: float = 0.5   # mean v_pred(+1) - mean v_pred(-1)
    value_ranking_acc: float = 0.75  # pairwise ranking accuracy on outcomes


@dataclass
class EvalGoals:
    """Approximate goals for the evaluation step."""
    win_rate_vs_random: float = 0.65
    min_opening_spread: float = 0.3   # max_score - min_score among top legal


@dataclass
class SelfPlayGoals:
    """Approximate goals for the self-play generation + fine-tune step."""
    min_games: int = 20
    min_samples: int = 200
    fine_tune_policy_loss: float = 0.5


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(path: str, model, step: int, metrics: Optional[dict] = None,
              extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": step,
        "config": model.cfg,
        "metrics": metrics or {},
        "saved_at": _utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_from_ckpt(path: str, device: torch.device):
    """Load a HiveLiteModel from a pipeline / train_from_dataset checkpoint."""
    from ghive_diffusion_lite import build_lite_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_lite_model()
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        step = int(ckpt.get("step", 0))
        metrics = ckpt.get("metrics") or {}
    elif isinstance(ckpt, dict) and any(
            k.startswith("text.") or k.startswith("value_head") for k in ckpt):
        state = ckpt
        step = 0
        metrics = {}
    else:
        raise ValueError(
            f"Unrecognised checkpoint format at {path!r}. "
            "Expected dict with 'model' key or a raw state_dict."
        )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, step, metrics, ckpt if isinstance(ckpt, dict) else {}


def _make_eval_adapter(model, tokenizer, player_type: str, fast_eval: bool = True,
                       lookahead_k: Optional[int] = None,
                       lookahead_weight: Optional[float] = None):
    """Build the player adapter used for win-rate evaluation."""
    from ghive_diffusion.inference import FastPlayer, ValuePlayer
    from ghive_diffusion.eval.runner import FastPlayerAdapter, ValuePlayerAdapter
    if player_type == "value":
        return ValuePlayerAdapter(ValuePlayer(model, tokenizer, deterministic=True))
    if fast_eval:
        # Use 1-ply value-head lookahead on the top-3 moves. The value head is
        # the offensive signal (it predicts win/loss from terminal-outcome
        # backfill); lookahead_k=0 ignores it entirely and was the reason eval
        # produced 80% draws (model defends but never converts to wins).
        player = FastPlayer(
            model, tokenizer, deterministic=True,
            lookahead_k=3, lookahead_weight=0.5, diffusion_candidates=False,
        )
    else:
        kwargs: Dict[str, Any] = {}
        if lookahead_k is not None:
            kwargs["lookahead_k"] = lookahead_k
        if lookahead_weight is not None:
            kwargs["lookahead_weight"] = lookahead_weight
        player = FastPlayer(model, tokenizer, deterministic=True, **kwargs)
    return FastPlayerAdapter(player)


def _eval_win_rate(
    checkpoint: str,
    device: torch.device,
    n_games: int = 20,
    num_workers: int = 1,
    fast_eval: bool = True,
    player_type: str = "fast",
    max_plies: int = 400,
    seed: int = 42,
) -> Dict[str, Any]:
    """Play *n_games* vs RandomPlayer and return win-rate stats.

    Lightweight version of ``step_eval`` used to rank training checkpoints.
    """
    from mzinga.core.enums import GameType
    from ghive_diffusion.tokenizer import build_default_tokenizer
    from ghive_diffusion.eval import RandomPlayer
    from ghive_diffusion.eval.runner import _play_one_game, EvalResults
    import concurrent.futures

    model, step, _metrics, _ = load_model_from_ckpt(checkpoint, device)
    tk = build_default_tokenizer(model.cfg)
    adapter = _make_eval_adapter(model, tk, player_type, fast_eval)

    results = EvalResults(player_name=adapter.name, opponent_name="random")

    if num_workers > 1:
        tasks_args = [(i, max_plies) for i in range(n_games)]
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_eval_worker,
            initargs=(checkpoint, str(device), player_type, fast_eval, seed),
        ) as executor:
            futures = [executor.submit(_worker_play_eval_game, arg) for arg in tasks_args]
            for future in concurrent.futures.as_completed(futures):
                _game_idx, outcome, plies = future.result()
                if outcome == "win":
                    results.wins += 1
                elif outcome == "loss":
                    results.losses += 1
                else:
                    results.draws += 1
                results.n_games += 1
                results.game_lengths.append(plies)
                completed += 1
                if completed % max(1, n_games // 5) == 0 or completed == n_games:
                    wr = results.wins / results.n_games
                    print(
                        f"    [best-eval] {completed:>3d}/{n_games}  "
                        f"W{results.wins}-L{results.losses}-D{results.draws}  wr={wr:.1%}",
                        flush=True,
                    )
    else:
        opponent = RandomPlayer()
        for i in range(n_games):
            random.seed(seed + i)
            torch.manual_seed(seed + i)
            player_first = (i % 2 == 0)
            outcome, plies = _play_one_game(
                adapter, opponent, GameType.Base,
                max_plies=max_plies, player_first=player_first,
            )
            if outcome == "win":
                results.wins += 1
            elif outcome == "loss":
                results.losses += 1
            else:
                results.draws += 1
            results.n_games += 1
            results.game_lengths.append(plies)
            if (i + 1) % max(1, n_games // 5) == 0 or (i + 1) == n_games:
                wr = results.wins / results.n_games
                print(
                    f"    [best-eval] {i + 1:>3d}/{n_games}  "
                    f"W{results.wins}-L{results.losses}-D{results.draws}  wr={wr:.1%}",
                    flush=True,
                )

    wr = results.win_rate
    return {
        "win_rate": wr,
        "wins": results.wins,
        "losses": results.losses,
        "draws": results.draws,
        "mean_plies": results.mean_game_length,
        "step": step,
        "checkpoint": checkpoint,
    }


# ---------------------------------------------------------------------------
# Metrics log
# ---------------------------------------------------------------------------

class MetricsLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def write(self, event: str, **fields) -> None:
        row = {
            "ts": _utc_now_iso(),
            "event": event,
            **fields,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


# ---------------------------------------------------------------------------
# STEP 1 — Train
# ---------------------------------------------------------------------------

def step_train(
    dataset_path: str,
    out_dir: str,
    device: torch.device,
    log: MetricsLog,
    total_steps: int = 20_000,
    lr: float = 1e-4,
    warmup_steps: int = 200,
    min_lr: float = 1e-5,
    log_interval: int = 500,
    ckpt_interval: int = 2000,
    max_grad_norm: float = 5.0,
    resume: Optional[str] = None,
    goals: Optional[TrainGoals] = None,
    best_by_eval: bool = True,
    best_eval_games: int = 20,
    best_eval_fast: bool = True,
    best_eval_player: str = "fast",
    best_eval_workers: int = 1,
    best_eval_seed: int = 42,
    best_eval_race_top_k: int = 3,
    best_eval_race_games: int = 100,
    max_plies: int = 400,
    outcome_oversample_target: float = 0.5,
    batch_size: int = 16,
    diffusion_weight: float = 0.1,
    value_weight: float = 1.0,
    self_condition_prob: float = 0.0,
) -> Dict[str, Any]:
    """Offline training with best-model selected by eval win rate vs random.

    With ``batch_size > 1`` the trainer takes one optimizer step per batch
    (``LiteHiveTrainer.step_batch``); ``total_steps`` then counts optimizer
    steps, not samples.  Steps/epoch = n_samples / batch_size.
    """
    from ghive_diffusion.tokenizer import build_default_tokenizer
    from ghive_diffusion.train_loop import create_scheduler
    from ghive_diffusion_lite import build_lite_model
    from ghive_diffusion_lite.lite_trainer import LiteHiveTrainer

    goals = goals or TrainGoals()
    _banner("STEP 1/3  TRAIN  — offline distillation on dataset")
    _kv("dataset", dataset_path)
    _kv("device", str(device))
    _kv("steps", f"{total_steps} (optimizer steps)")
    _kv("batch size", batch_size)
    _kv("loss weights", f"diff={diffusion_weight} pol=1.0 val={value_weight} (dynamic ramp)")
    _kv("lr / warmup / min_lr", f"{lr} / {warmup_steps} / {min_lr}")
    _kv("ckpt every", f"{ckpt_interval} steps")
    _kv("best metric", "eval win rate vs random" if best_by_eval else "policy_loss (lower is better)")
    if best_by_eval:
        _kv("best-eval games", best_eval_games)
        _kv("best-eval fast", best_eval_fast)
        _kv("best-eval player", best_eval_player)
        _kv("best-eval workers", best_eval_workers)
        _kv("best-eval seed", best_eval_seed)
        _kv("best-eval race top-k", best_eval_race_top_k)
        _kv("best-eval race games", best_eval_race_games)
    _kv("outcome oversample target", f"{outcome_oversample_target:.0%}")
    _kv("out_dir", out_dir)
    print()
    print("  Goals (approximate):")
    _kv("policy_loss  ≤", goals.policy_loss, indent=4)
    _kv("diffusion_loss ≤", goals.diffusion_loss, indent=4)
    _kv("value separation", f"≥ {goals.value_separation:.2f}", indent=4)
    _kv("value ranking acc", f"≥ {goals.value_ranking_acc:.0%}", indent=4)
    print(flush=True)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"  Loading dataset...", flush=True)
    t0 = time.time()
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    n_data = len(data)
    print(f"  ✓ {n_data:,} samples loaded in {_fmt_sec(time.time() - t0)}", flush=True)

    # Oversample decisive outcome samples so the value head sees enough ±1 signal.
    outcome_indices = [i for i in range(n_data) if abs(data[i].value) >= 0.9]
    n_out = len(outcome_indices)
    extra_copies = 0
    if n_out > 0 and 0 < outcome_oversample_target < 1:
        extra_copies = max(0, int(round(
            (outcome_oversample_target * n_data - n_out)
            / ((1 - outcome_oversample_target) * n_out)
        )))
    print(f"  outcome samples: {n_out:,} ({n_out / n_data:.1%}); "
          f"oversampling to ~{outcome_oversample_target:.0%} of each epoch "
          f"({extra_copies} extra copies)", flush=True)

    model = build_lite_model()
    tokenizer = build_default_tokenizer(model.cfg)
    start_step = 0

    if resume:
        print(f"  Resuming from {resume}", flush=True)
        model, start_step, _, _ = load_model_from_ckpt(resume, torch.device("cpu"))
        print(f"  ✓ resumed at step {start_step}", flush=True)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _kv("params", f"{n_params:,}")

    trainer = LiteHiveTrainer(
        model, tokenizer,
        diffusion_weight=diffusion_weight,
        value_weight=value_weight,
        diffusion_schedule="cosine",
        sc_ramp_steps=warmup_steps,
        self_condition_prob=self_condition_prob,
        max_grad_norm=max_grad_norm,
        total_steps=total_steps,
        dynamic_weights=True,
        device=device,
    )
    # Align step counter if resuming (scheduler still from 0 — acceptable)
    trainer.step_count = start_step

    remaining = max(0, total_steps - start_step)
    if remaining == 0:
        print("  Already at/ past total_steps; skipping training loop.", flush=True)
        final_path = os.path.join(out_dir, "final_model.pt")
        save_ckpt(final_path, model, trainer.step_count)
        return {
            "best_path": os.path.join(out_dir, "best_model.pt"),
            "final_path": final_path,
            "best_policy_loss": None,
            "final_metrics": {},
            "goals_met": {},
        }

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = create_scheduler(
        opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr,
    )
    # Fast-forward scheduler if resuming
    for _ in range(start_step):
        scheduler.step()

    best_policy = float("inf")
    best_path = os.path.join(out_dir, "best_model.pt")
    final_path = os.path.join(out_dir, "final_model.pt")
    last_path = os.path.join(out_dir, "last_model.pt")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    running: Dict[str, float] = {}
    loss_history: List[dict] = []
    t_start = time.time()
    epoch = 0

    log.write("train_start", dataset=dataset_path, n_samples=n_data,
              total_steps=total_steps, start_step=start_step, device=str(device))

    print(flush=True)
    while trainer.step_count < total_steps:
        epoch += 1
        indices = list(range(n_data)) + outcome_indices * extra_copies
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            if trainer.step_count >= total_steps:
                break

            batch = [data[i] for i in indices[start:start + batch_size]]
            metrics = trainer.step_batch(batch, optimizer=opt)
            scheduler.step()
            step = trainer.step_count

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + float(v)

            # ── log interval ──────────────────────────────────────
            if step % log_interval == 0 and step > 0:
                avg = {k: v / log_interval for k, v in running.items()}
                elapsed = time.time() - t_start
                done = step - start_step
                eta = (elapsed / max(1, done)) * (total_steps - step)
                lr_now = scheduler.get_last_lr()[0]
                pol = avg.get("policy_loss", float("nan"))
                diff = avg.get("diffusion_loss", float("nan"))
                val = avg.get("value_loss", float("nan"))
                loss = avg.get("loss", float("nan"))
                v_pred = metrics.get("value_pred", 0.0)
                v_sep = avg.get("value_separation", float("nan"))
                v_rank = avg.get("value_ranking_acc", float("nan"))
                top1 = avg.get("policy_top1", float("nan"))

                star = ""
                if pol < best_policy:
                    star = "  ★ best policy"

                print(
                    f"  [train] step {step:>6d}/{total_steps}  "
                    f"loss={loss:>6.3f}  diff={diff:>6.3f}  "
                    f"pol={pol:>6.3f}  val={val:>6.3f}  "
                    f"| v_pred={v_pred:+.3f}  top1={top1:.1%}  "
                    f"lr={lr_now:.1e}  eta {_fmt_eta(eta)}{star}",
                    flush=True,
                )
                if math.isfinite(v_sep) or math.isfinite(v_rank):
                    print(
                        f"           ↳ value: sep={v_sep:+.3f}  rank={v_rank:.1%}",
                        flush=True,
                    )
                loss_history.append({"step": step, **avg, "v_pred": v_pred})
                log.write(
                    "train_step", step=step, loss=loss, policy_loss=pol,
                    diffusion_loss=diff, value_loss=val, v_pred=v_pred,
                    policy_top1=top1,
                    value_separation=v_sep, value_ranking_acc=v_rank,
                    lr=lr_now, eta_s=eta,
                )
                running = {}

                # track best policy loss for logging; actual best_model.pt is chosen
                # by eval win rate at the end of training when best_by_eval=True.
                if pol < best_policy:
                    best_policy = pol
                    if not best_by_eval:
                        save_ckpt(
                            best_path, model, step,
                            metrics={
                                "policy_loss": pol,
                                "diffusion_loss": diff,
                                "loss": loss,
                                "value_loss": val,
                                "v_pred": v_pred,
                            },
                        )
                        print(f"           ↳ saved best → {best_path}  "
                              f"(pol={pol:.4f})", flush=True)
                    else:
                        print(f"           ↳ best policy so far (pol={pol:.4f}); "
                              f"will pick final best by eval win rate", flush=True)

            # ── periodic checkpoint ───────────────────────────────
            if ckpt_interval > 0 and step % ckpt_interval == 0 and step > 0:
                path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
                save_ckpt(path, model, step)
                save_ckpt(last_path, model, step)
                print(f"           ↳ checkpoint → {path}", flush=True)

    elapsed = time.time() - t_start
    # Always save final + last
    final_metrics = loss_history[-1] if loss_history else {}
    save_ckpt(final_path, model, trainer.step_count, metrics=final_metrics)
    save_ckpt(last_path, model, trainer.step_count, metrics=final_metrics)

    # ── select best checkpoint by eval win rate vs random ─────────
    best_eval: Dict[str, Any] = {}
    if best_by_eval:
        _section("STEP 1b  BEST CHECKPOINT SELECTION  — eval win rate vs random")
        candidates = [final_path, last_path]
        if ckpt_interval > 0 and os.path.isdir(ckpt_dir):
            for name in sorted(os.listdir(ckpt_dir)):
                if name.startswith("step_") and name.endswith(".pt"):
                    candidates.append(os.path.join(ckpt_dir, name))

        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if os.path.isfile(c) and c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        print(f"  Evaluating {len(unique_candidates)} checkpoints "
              f"({best_eval_games} games each, player={best_eval_player}, "
              f"fast={best_eval_fast}, workers={best_eval_workers})...", flush=True)

        sweep_results: List[Dict[str, Any]] = []
        for c in unique_candidates:
            print(f"\n  → {c}", flush=True)
            stats = _eval_win_rate(
                c, device,
                n_games=best_eval_games,
                num_workers=best_eval_workers,
                fast_eval=best_eval_fast,
                player_type=best_eval_player,
                max_plies=max_plies,
                seed=best_eval_seed,
            )
            wr = stats["win_rate"]
            print(f"    step {stats['step']:>6d}  wr={wr:.1%}  "
                  f"W{stats['wins']}-L{stats['losses']}-D{stats['draws']}  "
                  f"mean {stats['mean_plies']:.0f} plies", flush=True)
            sweep_results.append(stats)

        # Log the full sweep table
        sweep_path = os.path.join(out_dir, "best_eval_sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        log.write("best_eval_sweep", results=sweep_results, out_path=sweep_path)

        # Two-stage race: re-evaluate the top-k checkpoints with the full player
        # and more games, then promote the stable winner.
        race_top_k = max(1, best_eval_race_top_k)
        race_games = max(0, best_eval_race_games)
        if race_games > 0 and len(sweep_results) >= 2:
            _section("STEP 1c  BEST CHECKPOINT RACE  — full player, more games")
            ranked = sorted(sweep_results, key=lambda s: s["win_rate"], reverse=True)
            top_k = ranked[:race_top_k]
            print(f"  Top-{race_top_k} from sweep will race at {race_games} games each "
                  f"(fast=False, workers={best_eval_workers})...", flush=True)

            race_results: List[Dict[str, Any]] = []
            best_wr = -1.0
            best_ckpt = final_path
            best_stats: Dict[str, Any] = {}
            for stats in top_k:
                c = stats["checkpoint"]
                print(f"\n  → {c} (sweep wr={stats['win_rate']:.1%})", flush=True)
                race_stats = _eval_win_rate(
                    c, device,
                    n_games=race_games,
                    num_workers=best_eval_workers,
                    fast_eval=False,
                    player_type=best_eval_player,
                    max_plies=max_plies,
                    seed=best_eval_seed,
                )
                wr = race_stats["win_rate"]
                print(f"    step {race_stats['step']:>6d}  wr={wr:.1%}  "
                      f"W{race_stats['wins']}-L{race_stats['losses']}-D{race_stats['draws']}  "
                      f"mean {race_stats['mean_plies']:.0f} plies", flush=True)
                race_results.append(race_stats)
                if wr > best_wr:
                    best_wr = wr
                    best_ckpt = c
                    best_stats = race_stats

            race_path = os.path.join(out_dir, "best_eval_race.json")
            with open(race_path, "w") as f:
                json.dump(race_results, f, indent=2, default=str)
            log.write("best_eval_race", results=race_results, out_path=race_path)
        else:
            # No race: promote the sweep winner directly
            best_stats = max(sweep_results, key=lambda s: s["win_rate"])
            best_wr = best_stats["win_rate"]
            best_ckpt = best_stats["checkpoint"]

        # Load best and save as best_model.pt with eval metrics
        best_model, best_step, _best_metrics, _ = load_model_from_ckpt(best_ckpt, torch.device("cpu"))
        save_ckpt(
            best_path, best_model, best_step,
            metrics=_best_metrics,
            extra={
                "eval_win_rate": best_wr,
                "eval_wins": best_stats.get("wins", 0),
                "eval_losses": best_stats.get("losses", 0),
                "eval_draws": best_stats.get("draws", 0),
                "source_checkpoint": best_ckpt,
            },
        )
        best_eval = {
            "win_rate": best_wr,
            "wins": best_stats.get("wins", 0),
            "losses": best_stats.get("losses", 0),
            "draws": best_stats.get("draws", 0),
            "source_checkpoint": best_ckpt,
            "source_step": best_step,
        }
        print(f"\n  ✓ best_model.pt saved from {best_ckpt} "
              f"(step {best_step}, wr={best_wr:.1%})", flush=True)
    else:
        if best_policy == float("inf"):
            save_ckpt(best_path, model, trainer.step_count, metrics=final_metrics)
            best_policy = float(final_metrics.get("policy_loss", float("nan")))

    _section("STEP 1  TRAIN — summary")
    _kv("steps", trainer.step_count)
    _kv("epochs", epoch)
    _kv("time", _fmt_sec(elapsed))
    _kv("best policy_loss", f"{best_policy:.4f}")
    if best_by_eval and best_eval:
        _kv("best eval win rate", f"{best_eval['win_rate']:.1%}")
        _kv("best eval source", best_eval.get("source_checkpoint", best_path))
    _kv("best_model", best_path)
    _kv("final_model", final_path)

    pol_f = float(final_metrics.get("policy_loss", best_policy))
    diff_f = float(final_metrics.get("diffusion_loss", float("nan")))
    v_pred_f = float(final_metrics.get("v_pred", 0.0))
    v_sep_f = float(final_metrics.get("value_separation", float("nan")))
    v_rank_f = float(final_metrics.get("value_ranking_acc", float("nan")))

    goals_met = {
        "policy_loss": pol_f <= goals.policy_loss if math.isfinite(pol_f) else False,
        "diffusion_loss": diff_f <= goals.diffusion_loss if math.isfinite(diff_f) else False,
        "value_signal": (
            v_sep_f >= goals.value_separation and v_rank_f >= goals.value_ranking_acc
        ) if math.isfinite(v_sep_f) and math.isfinite(v_rank_f) else False,
    }
    print()
    print("  Goal check (final / best window):")
    _goal_line(
        f"policy_loss ≤ {goals.policy_loss}",
        goals_met["policy_loss"],
        f"best={best_policy:.4f}  final≈{pol_f:.4f}",
    )
    _goal_line(
        f"diffusion_loss ≤ {goals.diffusion_loss}",
        goals_met["diffusion_loss"],
        f"final≈{diff_f:.4f}",
    )
    _goal_line(
        f"value separation ≥ {goals.value_separation:.2f} & rank ≥ {goals.value_ranking_acc:.0%}",
        goals_met["value_signal"],
        f"sep={v_sep_f:+.3f}  rank={v_rank_f:.1%}  v_pred={v_pred_f:+.4f}",
    )

    log.write(
        "train_end", step=trainer.step_count, best_policy_loss=best_policy,
        final_metrics=final_metrics, goals_met=goals_met,
        best_path=best_path, final_path=final_path, elapsed_s=elapsed,
        best_eval=best_eval if best_by_eval else {},
    )

    return {
        "best_path": best_path,
        "final_path": final_path,
        "best_policy_loss": best_policy,
        "best_eval": best_eval if best_by_eval else {},
        "final_metrics": final_metrics,
        "goals_met": goals_met,
    }


_worker_adapter = None
_worker_opponent = None
_worker_eval_seed: int = 42


def _init_eval_worker(checkpoint: str, device_str: str, player_type: str,
                      fast_eval: bool, seed: int = 42,
                      lookahead_k: Optional[int] = None,
                      lookahead_weight: Optional[float] = None) -> None:
    global _worker_adapter, _worker_opponent, _worker_eval_seed
    import random
    import torch
    from mzinga.core.enums import GameType
    from ghive_diffusion.tokenizer import build_default_tokenizer
    from ghive_diffusion.eval import RandomPlayer

    _worker_eval_seed = seed

    if device_str == "cpu" or not torch.cuda.is_available():
        torch.set_num_threads(1)

    device = torch.device(device_str)
    model, _, _, _ = load_model_from_ckpt(checkpoint, device)
    tk = build_default_tokenizer(model.cfg)
    _worker_adapter = _make_eval_adapter(model, tk, player_type, fast_eval,
                                         lookahead_k, lookahead_weight)
    _worker_opponent = RandomPlayer()


def _worker_play_eval_game(args: tuple) -> tuple[int, str, int]:
    game_idx, max_plies = args
    import random
    import torch
    from mzinga.core.enums import GameType
    from ghive_diffusion.eval.runner import _play_one_game

    random.seed(_worker_eval_seed + game_idx)
    torch.manual_seed(_worker_eval_seed + game_idx)
    player_first = (game_idx % 2 == 0)

    outcome, plies = _play_one_game(
        _worker_adapter, _worker_opponent, GameType.Base,
        max_plies=max_plies, player_first=player_first
    )
    return game_idx, outcome, plies


def step_eval(
    checkpoint: str,
    out_dir: str,
    device: torch.device,
    log: MetricsLog,
    n_games: int = 40,
    max_plies: int = 400,
    num_workers: int = 1,
    fast_eval: bool = False,
    player_type: str = "fast",
    seed: int = 42,
    goals: Optional[EvalGoals] = None,
    lookahead_k: Optional[int] = None,
    lookahead_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """Opening diagnostics (fast player only) + win rate vs random."""
    from mzinga.core.board import Board
    from mzinga.core.enums import GameType
    from ghive_diffusion.tokenizer import build_default_tokenizer
    from ghive_diffusion.eval import RandomPlayer
    from ghive_diffusion.eval.runner import _play_one_game, EvalResults
    import concurrent.futures

    goals = goals or EvalGoals()
    _banner("STEP 2/3  EVAL  — openings + vs-random strength")
    _kv("checkpoint", checkpoint)
    _kv("device", str(device))
    _kv("games vs random", n_games)
    _kv("parallel workers", num_workers)
    _kv("player type", player_type)
    _kv("fast eval mode", fast_eval)
    print()
    print("  Goals (approximate):")
    _kv("win_rate vs random ≥", f"{goals.win_rate_vs_random:.0%}", indent=4)
    _kv("opening score spread ≥", goals.min_opening_spread, indent=4)
    print(flush=True)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model, step, metrics, _ = load_model_from_ckpt(checkpoint, device)
    _kv("loaded step", step)
    if metrics:
        _kv("ckpt metrics", {k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in metrics.items()})

    tk = build_default_tokenizer(model.cfg)
    adapter = _make_eval_adapter(model, tk, player_type, fast_eval,
                                 lookahead_k, lookahead_weight)

    # ── opening scores (only meaningful for the policy head) ────────
    opening_spread = 0.0
    top_move = None
    if player_type == "fast":
        _section("Opening preferences (empty Base board)")
        board = Board(GameType.Base)
        scored = adapter.fast.score(board)
        if not scored:
            print("  (no legal moves — unexpected)", flush=True)
        else:
            scores_t = torch.tensor([s.score for s in scored], dtype=torch.float32)
            probs = F.softmax(scores_t, dim=-1).tolist()
            order = sorted(range(len(scored)), key=lambda i: scored[i].score, reverse=True)
            print(f"  {'rank':<5} {'score':>8} {'prob':>7}  move")
            for rank, i in enumerate(order[:12], 1):
                s = scored[i]
                print(f"  {rank:<5} {s.score:>+8.3f} {probs[i]:>7.3f}  {s.move_str}")
            opening_spread = float(scores_t.max() - scores_t.min())
            top_move = scored[order[0]].move_str
            print(f"\n  top move: {top_move}   score spread: {opening_spread:.3f}",
                  flush=True)
    else:
        _section("Opening preferences (skipped for value-only player)")

    # ── vs random ─────────────────────────────────────────────────
    _section(f"Playing {n_games} games vs random ({num_workers} workers)")
    t0 = time.time()
    results = EvalResults(player_name=adapter.name, opponent_name="random")

    if num_workers > 1:
        tasks_args = [(i, max_plies) for i in range(n_games)]
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_eval_worker,
            initargs=(checkpoint, str(device), player_type, fast_eval, seed,
                      lookahead_k, lookahead_weight)
        ) as executor:
            futures = [executor.submit(_worker_play_eval_game, arg) for arg in tasks_args]
            for future in concurrent.futures.as_completed(futures):
                game_idx, outcome, plies = future.result()
                if outcome == "win":
                    results.wins += 1
                elif outcome == "loss":
                    results.losses += 1
                else:
                    results.draws += 1
                results.n_games += 1
                results.game_lengths.append(plies)
                completed += 1

                if completed % max(1, n_games // 10) == 0 or completed == n_games:
                    wr = results.wins / results.n_games
                    print(
                        f"  [eval] game {completed:>3d}/{n_games}  "
                        f"W{results.wins}-L{results.losses}-D{results.draws}  "
                        f"wr={wr:.1%}  last={outcome} ({plies} plies)",
                        flush=True,
                    )

    else:
        opponent = RandomPlayer()

        for i in range(n_games):
            random.seed(seed + i)
            torch.manual_seed(seed + i)
            player_first = (i % 2 == 0)
            outcome, plies = _play_one_game(
                adapter, opponent, GameType.Base,
                max_plies=max_plies, player_first=player_first,
            )
            if outcome == "win":
                results.wins += 1
            elif outcome == "loss":
                results.losses += 1
            else:
                results.draws += 1
            results.n_games += 1
            results.game_lengths.append(plies)

            if (i + 1) % max(1, n_games // 10) == 0 or (i + 1) == n_games:
                wr = results.wins / results.n_games
                print(
                    f"  [eval] game {i + 1:>3d}/{n_games}  "
                    f"W{results.wins}-L{results.losses}-D{results.draws}  "
                    f"wr={wr:.1%}  last={outcome} ({plies} plies)",
                    flush=True,
                )

    elapsed = time.time() - t0

    ci_lo, ci_hi = results.win_rate_ci

    _section("STEP 2  EVAL — summary")
    _kv("record", f"W{results.wins}-L{results.losses}-D{results.draws}")
    _kv("win rate", f"{results.win_rate:.1%}  95% CI [{ci_lo:.1%}, {ci_hi:.1%}]")
    _kv("mean plies", f"{results.mean_game_length:.1f}")
    _kv("time", _fmt_sec(elapsed))
    _kv("top opening", top_move)

    goals_met = {
        "win_rate": results.win_rate >= goals.win_rate_vs_random,
        "opening_spread": opening_spread >= goals.min_opening_spread,
    }
    print()
    print("  Goal check:")
    _goal_line(
        f"win_rate ≥ {goals.win_rate_vs_random:.0%}",
        goals_met["win_rate"],
        f"{results.win_rate:.1%}  CI[{ci_lo:.1%}, {ci_hi:.1%}]",
    )
    _goal_line(
        f"opening spread ≥ {goals.min_opening_spread}",
        goals_met["opening_spread"],
        f"{opening_spread:.3f}",
    )

    if not goals_met["win_rate"]:
        print()
        print("  ⚠  Win rate below goal. Self-play will still run if requested,")
        print("     but pure model self-play may reinforce weak play.")
        print("     Prefer more teacher data or longer train before relying on it.",
              flush=True)

    report_path = os.path.join(out_dir, "eval_report.json")
    report = {
        "checkpoint": checkpoint,
        "step": step,
        "opening_top": top_move,
        "opening_spread": opening_spread,
        "results": results.to_dict(),
        "goals_met": goals_met,
        "elapsed_s": elapsed,
        "seed": seed,
        "max_plies": max_plies,
        "fast_eval": fast_eval,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    _kv("report", report_path)

    log.write("eval_end", **report)
    return {
        "win_rate": results.win_rate,
        "goals_met": goals_met,
        "report_path": report_path,
        "results": results.to_dict(),
        "opening_top": top_move,
        "checkpoint": checkpoint,
    }


# ---------------------------------------------------------------------------
# STEP 3 — Self-play + optional fine-tune
# ---------------------------------------------------------------------------

def _to_lite_sample(s) -> "LiteTrainingSample":
    from ghive_diffusion_lite.lite_trainer import LiteTrainingSample
    has_outcome = getattr(s, "has_outcome", False) or abs(float(s.value)) >= 0.9
    return LiteTrainingSample(
        context_ids=s.context_ids.clone(),
        target_move_ids=s.target_move_ids.clone(),
        legal_move_ids=[list(ids) for ids in s.legal_move_ids],
        target_legal_idx=s.target_legal_idx,
        value=s.value,
        aux_targets=getattr(s, "aux_targets", None),
        has_outcome=has_outcome,
    )


def step_selfplay(
    checkpoint: str,
    out_dir: str,
    device: torch.device,
    log: MetricsLog,
    n_games: int = 50,
    use_teacher: bool = True,
    teacher_simulations: int = 50,
    fine_tune_steps: int = 5000,
    fine_tune_lr: float = 5e-5,
    log_interval: int = 250,
    ckpt_interval: int = 1000,
    max_plies: int = 400,
    goals: Optional[SelfPlayGoals] = None,
) -> Dict[str, Any]:
    """Generate mixed self-play data, then fine-tune the checkpoint."""
    from mzinga.core.enums import GameType, PlayerColor
    from ghive_diffusion.context_builder import HiveContextBuilder
    from ghive_diffusion.dataset import (
        SelfPlayGenerator, make_model_policy, make_random_policy,
    )
    from ghive_diffusion.tokenizer import build_default_tokenizer
    from ghive_diffusion.train_loop import create_scheduler
    from ghive_diffusion_lite.lite_trainer import LiteHiveTrainer, LiteTrainingSample

    goals = goals or SelfPlayGoals()
    _banner("STEP 3/3  SELF-PLAY  — mixed games + fine-tune")
    _kv("checkpoint", checkpoint)
    _kv("device", str(device))
    _kv("games", n_games)
    _kv("use_teacher", use_teacher)
    _kv("fine_tune_steps", fine_tune_steps)
    _kv("fine_tune_lr", fine_tune_lr)
    print()
    print("  Goals (approximate):")
    _kv("games ≥", goals.min_games, indent=4)
    _kv("samples ≥", goals.min_samples, indent=4)
    if fine_tune_steps > 0:
        _kv("fine-tune pol ≤", goals.fine_tune_policy_loss, indent=4)
    print()
    print("  Policy mix:")
    print("    • model  — current checkpoint (FastPlayer argmax)")
    print("    • random — weak opponent for value signal (asymmetric games)")
    if use_teacher:
        print("    • teacher — Mzinga AlphaZero MCTS (strong labels)")
        print("    Games alternate: model-vs-random sides + teacher-guided")
    else:
        print("    • teacher — OFF (model vs random only)")
    print(flush=True)

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model, step0, _, _ = load_model_from_ckpt(checkpoint, device)
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)

    model_policy = make_model_policy(model, tk, builder, deterministic=True)
    rand_policy = make_random_policy()

    teacher = None
    if use_teacher:
        print("  Loading Mzinga AlphaZero teacher...", flush=True)
        try:
            from ghive_diffusion_lite.mzinga_adapter import MzingaMCTSAdapter
            teacher = MzingaMCTSAdapter(
                device=str(device), num_simulations=teacher_simulations,
            )
            print(f"  ✓ teacher ready ({teacher_simulations} sims)", flush=True)
        except Exception as e:
            print(f"  ⚠ teacher load failed ({e}); continuing without teacher",
                  flush=True)
            use_teacher = False

    # Build alternating asymmetric policies:
    #   even games: White=model, Black=random  (or teacher mix on labels)
    #   odd games:  White=random, Black=model
    # For move *targets* we use the policy that actually played.
    # Additionally, with teacher: 1/3 of games use pure teacher for both sides.

    samples: List[LiteTrainingSample] = []
    n_mvr = 0
    n_teacher = 0
    n_ok = 0
    plies_list: List[int] = []
    t0 = time.time()

    def _play_with(policy) -> List[LiteTrainingSample]:
        gen = SelfPlayGenerator(
            tk, builder, game_type=GameType.Base,
            move_policy=policy, max_plies=max_plies,
        )
        game_samples, _ = gen._play_one()
        return [_to_lite_sample(s) for s in game_samples]

    print(flush=True)
    for gi in range(n_games):
        if use_teacher and teacher is not None:
            if gi % 3 == 2:
                mode = "teacher_vs_teacher"
                def policy(board):
                    ans = teacher(board)
                    policy.last_value = getattr(teacher, "last_value", None)
                    return ans
            else:
                mode = "teacher_vs_random"
                teacher_is_white = (gi % 2 == 0)
                def policy(board, _tw=teacher_is_white):
                    is_teacher_turn = (board.current_color == PlayerColor.White) == _tw
                    p = teacher if is_teacher_turn else rand_policy
                    ans = p(board)
                    policy.last_value = getattr(p, "last_value", None)
                    return ans
        else:
            mode = "model_vs_random"
            model_is_white = (gi % 2 == 0)
            def policy(board, _mw=model_is_white):
                is_model_turn = (board.current_color == PlayerColor.White) == _mw
                p = model_policy if is_model_turn else rand_policy
                return p(board)

        try:
            raw_samples = _play_with(policy)
            
            # Filter samples to only keep the moves from the non-random side
            game_samples = []
            from ghive_diffusion.dataset import _iter_ply_sides
            
            if mode == "teacher_vs_random":
                for sample, ply_side in zip(raw_samples, _iter_ply_sides(GameType.Base, len(raw_samples))):
                    teacher_side = PlayerColor.White if teacher_is_white else PlayerColor.Black
                    if ply_side == teacher_side:
                        game_samples.append(sample)
            elif mode == "model_vs_random":
                for sample, ply_side in zip(raw_samples, _iter_ply_sides(GameType.Base, len(raw_samples))):
                    model_side = PlayerColor.White if model_is_white else PlayerColor.Black
                    if ply_side == model_side:
                        game_samples.append(sample)
            else:
                game_samples = raw_samples

        except Exception as e:
            print(f"  [selfplay] game {gi + 1} failed: {e}", flush=True)
            continue

        samples.extend(game_samples)
        n_ok += 1
        if mode == "teacher_vs_teacher":
            n_teacher += 1
        else:
            n_mvr += 1
        plies_list.append(len(game_samples))

        if (gi + 1) % max(1, n_games // 10) == 0 or (gi + 1) == n_games:
            elapsed = time.time() - t0
            rate = (gi + 1) / max(elapsed, 1e-6)
            eta = (n_games - gi - 1) / max(rate, 1e-6)
            print(
                f"  [selfplay] game {gi + 1:>4d}/{n_games}  "
                f"samples={len(samples):,}  "
                f"mode={mode:<16s}  plies={len(game_samples):3d}  "
                f"eta {_fmt_eta(eta)}",
                flush=True,
            )

    gen_elapsed = time.time() - t0
    sp_path = os.path.join(out_dir, "selfplay_dataset.pt")
    random.shuffle(samples)
    torch.save(samples, sp_path)

    _section("Self-play generation — summary")
    _kv("games completed", f"{n_ok}/{n_games}")
    _kv("model_vs_random games", n_mvr)
    _kv("teacher games", n_teacher)
    _kv("samples", f"{len(samples):,}")
    mean_plies = sum(plies_list) / max(1, len(plies_list))
    _kv("mean plies", f"{mean_plies:.1f}")
    _kv("time", _fmt_sec(gen_elapsed))
    _kv("dataset", sp_path)

    gen_goals = {
        "min_games": n_ok >= goals.min_games,
        "min_samples": len(samples) >= goals.min_samples,
    }
    print()
    print("  Goal check (generation):")
    _goal_line(f"games ≥ {goals.min_games}", gen_goals["min_games"], str(n_games))
    _goal_line(
        f"samples ≥ {goals.min_samples}", gen_goals["min_samples"],
        f"{len(samples):,}",
    )

    log.write(
        "selfplay_gen_end", n_games=n_games, n_ok=n_ok, n_samples=len(samples),
        path=sp_path, n_mvr=n_mvr, n_teacher=n_teacher,
        elapsed_s=gen_elapsed, goals_met=gen_goals,
    )

    result: Dict[str, Any] = {
        "selfplay_dataset": sp_path,
        "n_samples": len(samples),
        "n_games": n_ok,
        "gen_goals_met": gen_goals,
        "finetune": None,
    }

    if fine_tune_steps <= 0 or len(samples) == 0:
        print("\n  Skipping fine-tune (steps=0 or no samples).", flush=True)
        return result

    # ── fine-tune ─────────────────────────────────────────────────
    _section(f"Fine-tune for {fine_tune_steps} steps on self-play data")
    model.train()
    trainer = LiteHiveTrainer(
        model, tk,
        diffusion_schedule="cosine",
        sc_ramp_steps=min(200, fine_tune_steps // 5),
        self_condition_prob=0.5,
        max_grad_norm=5.0,
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=fine_tune_lr)
    scheduler = create_scheduler(
        opt, warmup_steps=min(100, fine_tune_steps // 10),
        total_steps=fine_tune_steps, min_lr=fine_tune_lr * 0.1,
    )

    best_pol = float("inf")
    ft_best = os.path.join(out_dir, "selfplay_best_model.pt")
    ft_final = os.path.join(out_dir, "selfplay_final_model.pt")
    running: Dict[str, float] = {}
    t_ft = time.time()
    n_data = len(samples)

    while trainer.step_count < fine_tune_steps:
        indices = list(range(n_data))
        random.shuffle(indices)
        for i in indices:
            if trainer.step_count >= fine_tune_steps:
                break
            metrics = trainer.step(samples[i], optimizer=opt)
            scheduler.step()
            step = trainer.step_count
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + float(v)

            if step % log_interval == 0 and step > 0:
                avg = {k: v / log_interval for k, v in running.items()}
                elapsed = time.time() - t_ft
                eta = (elapsed / step) * (fine_tune_steps - step)
                pol = avg.get("policy_loss", float("nan"))
                diff = avg.get("diffusion_loss", float("nan"))
                loss = avg.get("loss", float("nan"))
                star = "  ★ best" if pol < best_pol else ""
                print(
                    f"  [finetune] step {step:>5d}/{fine_tune_steps}  "
                    f"loss={loss:>6.3f}  diff={diff:>6.3f}  pol={pol:>6.3f}  "
                    f"lr={scheduler.get_last_lr()[0]:.1e}  "
                    f"eta {_fmt_eta(eta)}{star}",
                    flush=True,
                )
                log.write("finetune_step", step=step, **avg)
                running = {}
                if pol < best_pol:
                    best_pol = pol
                    save_ckpt(ft_best, model, step0 + step,
                              metrics={"policy_loss": pol, "diffusion_loss": diff,
                                       "loss": loss, "phase": "selfplay_finetune"})
                    print(f"             ↳ saved best → {ft_best}", flush=True)

            if ckpt_interval > 0 and step % ckpt_interval == 0 and step > 0:
                ckpt_dir = os.path.join(out_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                path = os.path.join(ckpt_dir, f"finetune_{step:06d}.pt")
                save_ckpt(path, model, step0 + step)

    save_ckpt(ft_final, model, step0 + trainer.step_count)
    if best_pol == float("inf"):
        save_ckpt(ft_best, model, step0 + trainer.step_count)
        best_pol = float("nan")

    ft_goals = {
        "policy_loss": (best_pol <= goals.fine_tune_policy_loss
                        if math.isfinite(best_pol) else False),
    }
    _section("STEP 3  SELF-PLAY — summary")
    _kv("fine-tune steps", trainer.step_count)
    _kv("best policy_loss", f"{best_pol:.4f}" if math.isfinite(best_pol) else "n/a")
    _kv("best model", ft_best)
    _kv("final model", ft_final)
    _kv("time", _fmt_sec(time.time() - t_ft))
    print()
    print("  Goal check (fine-tune):")
    _goal_line(
        f"policy_loss ≤ {goals.fine_tune_policy_loss}",
        ft_goals["policy_loss"],
        f"best={best_pol:.4f}" if math.isfinite(best_pol) else "n/a",
    )

    log.write(
        "selfplay_finetune_end", steps=trainer.step_count,
        best_policy_loss=best_pol, best_path=ft_best, final_path=ft_final,
        goals_met=ft_goals,
    )
    result["finetune"] = {
        "best_path": ft_best,
        "final_path": ft_final,
        "best_policy_loss": best_pol,
        "goals_met": ft_goals,
    }
    result["goals_met"] = {**gen_goals, **{f"ft_{k}": v for k, v in ft_goals.items()}}
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = _resolve_device(args.device)
    log = MetricsLog(os.path.join(out_dir, "metrics.jsonl"))

    # Resolve which steps to run
    order = ["train", "eval", "selfplay"]
    if args.only:
        steps = [args.only]
    else:
        steps = list(order)
        if args.start_from:
            if args.start_from not in order:
                print(f"Unknown --start-from {args.start_from!r}", file=sys.stderr)
                return 2
            steps = order[order.index(args.start_from):]
        if args.skip_train:
            steps = [s for s in steps if s != "train"]
        if args.skip_eval:
            steps = [s for s in steps if s != "eval"]
        if args.skip_selfplay:
            steps = [s for s in steps if s != "selfplay"]

    _banner("HiveLite pipeline")
    _kv("out_dir", out_dir)
    _kv("device (requested)", args.device)
    _kv("device (resolved)", str(device))
    _kv("steps", " → ".join(steps) if steps else "(none)")
    _kv("dataset", args.dataset)
    _kv("checkpoint", args.checkpoint or "(from train best)")
    print(flush=True)

    log.write("pipeline_start", steps=steps, device=str(device),
              out_dir=out_dir, args=vars(args))

    summary: Dict[str, Any] = {"steps": steps, "results": {}}
    active_ckpt = args.checkpoint

    # ── TRAIN ─────────────────────────────────────────────────────
    if "train" in steps:
        try:
            tr = step_train(
                dataset_path=args.dataset,
                out_dir=out_dir,
                device=device,
                log=log,
                total_steps=args.steps,
                lr=args.lr,
                warmup_steps=args.warmup_steps,
                min_lr=args.min_lr,
                log_interval=args.log_interval,
                ckpt_interval=args.ckpt_interval,
                max_grad_norm=args.max_grad_norm,
                resume=args.resume,
                best_by_eval=args.best_by_eval,
                best_eval_games=args.best_eval_games,
                best_eval_fast=args.best_eval_fast,
                best_eval_player=args.eval_player,
                best_eval_workers=args.num_workers,
                best_eval_seed=args.best_eval_seed,
                best_eval_race_top_k=args.best_eval_race_top_k,
                best_eval_race_games=args.best_eval_race_games,
                max_plies=args.max_plies,
                outcome_oversample_target=args.outcome_oversample_target,
                batch_size=args.batch_size,
                diffusion_weight=args.diffusion_weight,
                value_weight=args.value_weight,
                self_condition_prob=args.self_condition_prob,
            )
            summary["results"]["train"] = tr
            active_ckpt = tr["best_path"]
        except Exception as e:
            print(f"\n  ✗ TRAIN failed: {e}", flush=True)
            log.write("pipeline_error", step="train", error=str(e))
            if not args.continue_on_fail:
                raise
    else:
        print("\n  (skipping STEP 1 train)", flush=True)

    # Default checkpoint for later steps
    if active_ckpt is None:
        candidates = [
            os.path.join(out_dir, "best_model.pt"),
            os.path.join(out_dir, "final_model.pt"),
            os.path.join(out_dir, "last_model.pt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                active_ckpt = c
                break
    if active_ckpt is None and ("eval" in steps or "selfplay" in steps):
        print(
            "\n  ✗ No checkpoint available for eval/selfplay.\n"
            "    Pass --checkpoint PATH or run train first.",
            file=sys.stderr,
        )
        return 1

    # ── EVAL ──────────────────────────────────────────────────────
    if "eval" in steps:
        try:
            ev = step_eval(
                checkpoint=active_ckpt,
                out_dir=out_dir,
                device=device,
                log=log,
                n_games=args.eval_games,
                max_plies=args.max_plies,
                num_workers=args.num_workers,
                fast_eval=args.fast_eval,
                player_type=args.eval_player,
                seed=args.best_eval_seed,
                lookahead_k=args.eval_lookahead_k,
                lookahead_weight=args.eval_lookahead_weight,
            )
            summary["results"]["eval"] = ev
            # Gate self-play recommendation
            if not ev["goals_met"].get("win_rate", False) and "selfplay" in steps:
                if args.force_selfplay:
                    print("\n  ⚠ Eval goals not met, but --force-selfplay set; continuing.",
                          flush=True)
                elif args.skip_selfplay_on_eval_fail:
                    print("\n  ⚠ Eval win-rate goal not met — skipping self-play "
                          "(override with --force-selfplay).", flush=True)
                    steps = [s for s in steps if s != "selfplay"]
        except Exception as e:
            print(f"\n  ✗ EVAL failed: {e}", flush=True)
            log.write("pipeline_error", step="eval", error=str(e))
            if not args.continue_on_fail:
                raise
    else:
        print("\n  (skipping STEP 2 eval)", flush=True)

    # ── SELFPLAY ──────────────────────────────────────────────────
    if "selfplay" in steps:
        try:
            sp = step_selfplay(
                checkpoint=active_ckpt,
                out_dir=out_dir,
                device=device,
                log=log,
                n_games=args.selfplay_games,
                use_teacher=not args.no_teacher,
                teacher_simulations=args.teacher_sims,
                fine_tune_steps=args.finetune_steps,
                fine_tune_lr=args.finetune_lr,
                log_interval=max(50, args.log_interval // 2),
                ckpt_interval=args.ckpt_interval,
                max_plies=args.max_plies,
            )
            summary["results"]["selfplay"] = sp
            if sp.get("finetune") and sp["finetune"].get("best_path"):
                active_ckpt = sp["finetune"]["best_path"]
        except Exception as e:
            print(f"\n  ✗ SELFPLAY failed: {e}", flush=True)
            log.write("pipeline_error", step="selfplay", error=str(e))
            if not args.continue_on_fail:
                raise
    else:
        print("\n  (skipping STEP 3 selfplay)", flush=True)

    # ── final rollup ──────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    _kv("out_dir", out_dir)
    _kv("active checkpoint", active_ckpt)
    if "train" in summary["results"]:
        tr = summary["results"]["train"]
        _kv("train best pol", tr.get("best_policy_loss"))
        _kv("train best path", tr.get("best_path"))
    if "eval" in summary["results"]:
        ev = summary["results"]["eval"]
        _kv("eval win rate", f"{ev.get('win_rate', 0):.1%}")
        _kv("eval goals", ev.get("goals_met"))
    if "selfplay" in summary["results"]:
        sp = summary["results"]["selfplay"]
        _kv("selfplay samples", sp.get("n_samples"))
        _kv("selfplay dataset", sp.get("selfplay_dataset"))
        if sp.get("finetune"):
            _kv("finetune best", sp["finetune"].get("best_path"))
    _kv("metrics log", os.path.join(out_dir, "metrics.jsonl"))
    print(flush=True)

    summary_path = os.path.join(out_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary written to {summary_path}", flush=True)
    log.write("pipeline_end", summary=summary, active_ckpt=active_ckpt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ghive_diffusion_lite.pipeline",
        description=(
            "HiveLite end-to-end pipeline: train → eval → self-play.\n"
            "Saves best-model checkpoints, prints goals, and can skip steps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Full pipeline (M4 Mac)
  python -m ghive_diffusion_lite.pipeline \\
      --dataset ghive_diffusion_lite/dataset_merged.pt \\
      --device mps --out-dir runs/lite_run1

  # Eval + self-play only
  python -m ghive_diffusion_lite.pipeline --skip-train \\
      --checkpoint runs/lite_run1/best_model.pt --device mps \\
      --out-dir runs/lite_run1

  # Start from eval
  python -m ghive_diffusion_lite.pipeline --start-from eval \\
      --checkpoint runs/lite_run1/best_model.pt --device mps

  # Fast parallel eval only
  python -m ghive_diffusion_lite.pipeline --only eval \\
      --checkpoint runs/lite_run_dagger/best_model.pt \\
      --num-workers 4 --fast-eval --device cpu
""",
    )

    # Paths
    p.add_argument("--dataset", default="data/dataset_merged.pt",
                   help="pre-generated .pt dataset for step 1")
    p.add_argument("--out-dir", default="runs/lite_pipeline",
                   help="directory for checkpoints, logs, reports")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint for eval/selfplay (default: out-dir/best_model.pt)")
    p.add_argument("--resume", default=None,
                   help="optional checkpoint to resume training from")
    p.add_argument("--device", default="cpu",
                   help="cpu | mps | cuda (falls back if unavailable)")

    # Step selection
    g = p.add_argument_group("step selection")
    g.add_argument("--only", choices=["train", "eval", "selfplay"],
                   help="run exactly one step")
    g.add_argument("--start-from", choices=["train", "eval", "selfplay"],
                   help="run this step and all following")
    g.add_argument("--skip-train", action="store_true")
    g.add_argument("--skip-eval", action="store_true")
    g.add_argument("--skip-selfplay", action="store_true")
    g.add_argument("--force-selfplay", action="store_true",
                   help="run self-play even if eval win-rate goal fails")
    g.add_argument("--skip-selfplay-on-eval-fail", action="store_true",
                   default=True,
                   help="(default) skip self-play when eval win-rate goal fails")
    g.add_argument("--no-skip-selfplay-on-eval-fail",
                   action="store_false", dest="skip_selfplay_on_eval_fail",
                   help="always run self-play after eval (unless --skip-selfplay)")
    g.add_argument("--continue-on-fail", action="store_true",
                   help="continue pipeline if a step raises")

    # Train hyperparams
    t = p.add_argument_group("train")
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup-steps", type=int, default=200)
    t.add_argument("--min-lr", type=float, default=1e-5)
    t.add_argument("--log-interval", type=int, default=500)
    t.add_argument("--ckpt-interval", type=int, default=2000,
                   help="periodic checkpoint interval (0=disable periodic)")
    t.add_argument("--max-grad-norm", type=float, default=5.0)
    t.add_argument("--best-by-eval", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="select best_model.pt by eval win rate vs random (default: True)")
    t.add_argument("--best-eval-games", type=int, default=20,
                   help="games per checkpoint when selecting best by eval win rate")
    t.add_argument("--best-eval-fast", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="use fast eval (no lookahead) for best-checkpoint sweep")
    t.add_argument("--best-eval-seed", type=int, default=42,
                   help="base seed for eval games (deterministic per game index)")
    t.add_argument("--best-eval-race-top-k", type=int, default=3,
                   help="top-k checkpoints from the sweep to race with the full player")
    t.add_argument("--best-eval-race-games", type=int, default=100,
                   help="games per checkpoint in the full-player race (0=skip race)")
    t.add_argument("--outcome-oversample-target", type=float, default=0.5,
                   help="target fraction of decisive outcome samples per training epoch")
    t.add_argument("--batch-size", type=int, default=16,
                   help="samples per optimizer step (default 16; --steps counts "
                        "optimizer steps, so steps/epoch = n_samples/batch_size)")
    t.add_argument("--diffusion-weight", type=float, default=0.1,
                   help="base weight of the diffusion denoising loss (default 0.1; "
                        "the policy pathway used at play time is what matters)")
    t.add_argument("--value-weight", type=float, default=1.0,
                   help="base weight of the value loss (default 1.0; outcome "
                        "samples are the strongest signal in the dataset)")
    t.add_argument("--self-condition-prob", type=float, default=0.0,
                   help="probability of a self-conditioning pass per sample "
                        "(default 0.0 — it only helps text generation, not play, "
                        "and costs an extra encoder+decoder forward)")

    # Eval
    e = p.add_argument_group("eval")
    e.add_argument("--eval-games", type=int, default=40)
    e.add_argument("--max-plies", type=int, default=400)
    e.add_argument("--num-workers", type=int, default=1,
                   help="number of parallel processes for eval games (default: 1)")
    e.add_argument("--fast-eval", action="store_true",
                   help="speed up eval by disabling lookahead and candidate generation")
    e.add_argument("--eval-player", choices=["fast", "value"], default="fast",
                   help="player for eval: fast (policy) or value (pure value-greedy)")
    e.add_argument("--eval-lookahead-k", type=int, default=None,
                   help="override FastPlayer lookahead_k for eval (default: "
                        "player default, 3 when not --fast-eval)")
    e.add_argument("--eval-lookahead-weight", type=float, default=None,
                   help="override FastPlayer lookahead_weight for eval "
                        "(default: player default, 0.2; try 1.0 to make the "
                        "value head the dominant move-selection signal)")


    # Self-play
    s = p.add_argument_group("selfplay")
    s.add_argument("--selfplay-games", type=int, default=50)
    s.add_argument("--no-teacher", action="store_true",
                   help="disable Mzinga teacher during self-play")
    s.add_argument("--teacher-sims", type=int, default=50)
    s.add_argument("--finetune-steps", type=int, default=5000)
    s.add_argument("--finetune-lr", type=float, default=5e-5)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
