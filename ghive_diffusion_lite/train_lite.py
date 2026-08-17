"""On-the-fly / dataset training loops for HiveLiteModel.

.. deprecated::
    Prefer ``python -m ghive_diffusion_lite.pipeline`` for the supported
    train → eval → self-play flow with best-model checkpoints and goals.

These functions remain for library use and experiments:

- :func:`train_lite` — on-the-fly Mzinga (or random) self-play training
- :func:`train_from_dataset` — offline training (no mid-run best save)

Pass ``use_mz_ai=False`` to fall back to random self-play in ``train_lite``.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mzinga.core.board import Board
from mzinga.core.enums import GameType

from ghive_diffusion.context_builder import HiveContextBuilder
from ghive_diffusion.dataset import SelfPlayGenerator
from ghive_diffusion.tokenizer import build_default_tokenizer
from ghive_diffusion.training import HiveTrainer, TrainingSample
from ghive_diffusion.train_loop import create_scheduler

from .hive_lite_config import HiveLiteConfig
from .hive_lite_model import HiveLiteModel, build_lite_model
from .lite_trainer import LiteHiveTrainer, LiteTrainingSample
from .mzinga_adapter import MzingaMCTSAdapter


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def _compute_grad_stats(model: torch.nn.Module) -> dict:
    """Compute per-parameter gradient norms (post-optimizer step, pre-zerograd)."""
    param_norms = []
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2).item()
            param_norms.append(n)
    if not param_norms:
        return {"grad_mean": 0.0, "grad_max": 0.0, "grad_std": 0.0, "grad_l2": 0.0}
    return {
        "grad_mean": float(np.mean(param_norms)),
        "grad_max": float(np.max(param_norms)),
        "grad_std": float(np.std(param_norms)),
        "grad_l2": float(np.sqrt(sum(n ** 2 for n in param_norms))),
    }


def _compute_param_stats(model: torch.nn.Module) -> dict:
    """Compute aggregate parameter statistics."""
    norms = []
    for p in model.parameters():
        norms.append(p.data.norm(2).item())
    return {
        "param_l2_total": float(np.sqrt(sum(n ** 2 for n in norms))),
        "param_mean": float(np.mean(norms)),
        "param_max": float(np.max(norms)),
    }


# ---------------------------------------------------------------------------
# Training report
# ---------------------------------------------------------------------------

def _generate_report(history: dict) -> str:
    """Produce a human-readable training report from collected history.

    Parameters
    ----------
    history : dict
        Keys: losses (list of dicts), grads (list of dicts),
        params (list of dicts), total_steps, elapsed_s, device,
        lr, nan_steps (set of step #s where NaN appeared).
    """
    losses = history["losses"]
    grads = history["grads"]
    total = history["total_steps"]

    if not losses:
        return "No training steps completed."

    _SKIP_KEYS = {"step", "step_count", "effective_sc_prob", "used_self_conditioning", "value_pred"}

    def _loss_keys(record: dict) -> list:
        return [k for k in record if k not in _SKIP_KEYS
                and isinstance(record[k], (int, float))]

    loss_keys = sorted(_loss_keys(losses[-1]))

    # ── value head diagnostics ────────────────────────────────────
    final_all = losses[-1]
    value_preds = [r.get("value_pred", 0.0) for r in losses[5:] if "value_pred" in r]
    val_losses = [r.get("value_loss", 0.0) for r in losses if "value_loss" in r]

    # ── gradient diagnostics ──────────────────────────────────────
    grad_l2s = [g["grad_l2"] for g in grads]
    clip_frac = history.get("clip_frac", 0.0)

    # ── loss trends ───────────────────────────────────────────────
    def _slope(key, window=50):
        if len(losses) < window:
            return 0.0
        recent = [r.get(key, 0.0) for r in losses[-window:]]
        return (recent[-1] - recent[0]) / max(1, len(recent))

    # ── issues ────────────────────────────────────────────────────
    issues: list[str] = []
    nan_steps = history.get("nan_steps", set())

    if nan_steps:
        issues.append(
            f"NaN losses at steps: {sorted(nan_steps)[:10]}"
            + (" ..." if len(nan_steps) > 10 else ""))

    if clip_frac > 0.20:
        issues.append(
            f"Gradient clipping in {clip_frac*100:.0f}% of steps (>20%) — "
            "gradients may be exploding; reduce LR or add warmup")

    if clip_frac > 0.05:
        issues.append(
            f"Gradient clipping in {clip_frac*100:.0f}% of steps (>5%) — "
            "moderate clipping, consider reducing LR")

    if value_preds and abs(np.mean(value_preds)) < 0.01 and len(losses) > 200:
        issues.append(
            f"Value prediction mean is {np.mean(value_preds):.4f} (near zero) — "
            "value head may not be learning. Game outcomes are dominated by "
            "random-play draws/early wins; expected with random self-play.")

    diff_slope = _slope("diffusion_loss")
    if diff_slope > 0 and len(losses) > 100:
        issues.append(
            f"Diffusion loss INCREASING ({diff_slope:+.4f}/step) — model may be "
            "destabilising. Check LR and gradient clipping.")

    total_loss_start = losses[0].get("loss", 0)
    total_loss_end = losses[-1].get("loss", 0)
    if total_loss_end > total_loss_start * 0.95 and len(losses) > 200:
        issues.append(
            "Total loss barely changed — model may be undertrained or the "
            "config is too small for the task.")

    pol_slope = _slope("policy_loss")
    if pol_slope < 0:
        issues.append("Policy loss decreasing — model IS learning move preferences.")

    if not issues:
        issues.append("No training issues detected — training appears stable.")

    # ── format ─────────────────────────────────────────────────────
    lines = []
    w = 64
    lines.append("=" * w)
    lines.append("TRAINING REPORT".center(w))
    lines.append("=" * w)
    lines.append("")
    lines.append(f"Steps:             {total}")
    lines.append(f"Time:              {history['elapsed_s']:.1f}s "
                  f"({history['elapsed_s'] / total:.3f}s/step)")
    lines.append(f"Device:            {history['device']}")
    lines.append(f"Learning rate:     {history['lr']} (peak)")
    lines.append(f"  warmup:          {history.get('warmup_steps', 0)} steps")
    lines.append(f"  min_lr:          {history.get('min_lr', 0)}")
    lines.append(f"Gradient clip:     {history.get('max_grad_norm', 1.0)}")
    lines.append(f"Asymmetric play:   {history.get('asymmetric', False)}")
    lines.append(f"Data policy:       {history.get('policy', 'random')}")
    lines.append("")

    last_param = history.get("params", [{}])[-1] if history.get("params") else {}
    if last_param:
        lines.append("── Parameters ──")
        lines.append(f"  Total L2:       {last_param.get('param_l2_total', 0):.2f}")
        lines.append(f"  Per-layer mean: {last_param.get('param_mean', 0):.4f}")
        lines.append(f"  Per-layer max:  {last_param.get('param_max', 0):.4f}")
        lines.append("")

    lines.append("── Final Losses ──")
    for key in loss_keys:
        lines.append(f"  {key:>22s}: {final_all[key]:.4f}")

    lines.append("")
    lines.append("── Loss Trends (last 10%) ──")
    window = max(10, total // 10)
    for key in sorted(loss_keys):
        recent = [r.get(key, 0.0) for r in losses[-window:]]
        if recent:
            sl = _slope(key, window)
            dir_sym = "↓" if sl < 0 else "↑" if sl > 0 else "→"
            lines.append(
                f"  {key:>22s}: {recent[-1]:.4f} {dir_sym} "
                f"(was {losses[0].get(key, 0):.4f} at step 0, "
                f"slope {sl:+.4f}/step)")

    lines.append("")
    lines.append("── Gradients ──")
    if grad_l2s:
        lines.append(f"  Mean L2:        {np.mean(grad_l2s):.4f}")
        lines.append(f"  Max L2:         {np.max(grad_l2s):.4f}")
        lines.append(f"  Std L2:         {np.std(grad_l2s):.4f}")
        lines.append(f"  Final L2:       {grad_l2s[-1]:.4f}")
        lines.append(f"  Clipped steps:  {clip_frac*100:.1f}% "
                      f"({int(clip_frac * total)}/{total})")

    lines.append("")
    lines.append("── Diagnostics ──")
    if value_preds:
        lines.append(f"  value_pred mean: {np.mean(value_preds):+.4f}")
        lines.append(f"  value_pred min:  {np.min(value_preds):+.4f}")
        lines.append(f"  value_pred max:  {np.max(value_preds):+.4f}")
    if final_all.get("effective_sc_prob", 0) > 0:
        lines.append(f"  sc_prob (final): {final_all['effective_sc_prob']:.2f}")

    lines.append("")
    lines.append("── Issues ──")
    for i, issue in enumerate(issues, 1):
        lines.append(f"  {i}. {issue}")

    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lite(total_steps: int = 2000,
               lr: float = 1e-4,
               warmup_steps: int = 200,
               min_lr: float = 1e-5,
               log_interval: int = 200,
               save_path: Optional[str] = "lite_model.pt",
               device_str: str = "cpu",
               game_type: GameType = GameType.Base,
               use_mz_ai: bool = True,
               mz_simulations: int = 50,
               asymmetric: bool = True,
               progress_fn: Optional[Callable[[dict], None]] = None,
               ) -> HiveLiteModel:
    """Train a ``HiveLiteModel`` with on-the-fly self-play.

    .. deprecated::
        Use ``python -m ghive_diffusion_lite.pipeline`` for production
        runs (best checkpoints, eval, self-play).

    By default uses Mzinga's pretrained AlphaZero via MCTS to generate
    strong training moves with alternating asymmetric play (MCTS vs
    random, sides swapped each game).  Pass ``use_mz_ai=False`` for
    random self-play, ``asymmetric=False`` for symmetric MCTS-vs-MCTS.

    Uses a linear warmup → cosine decay LR schedule with configurable
    gradient clipping.

    Parameters
    ----------
    total_steps : int
        Number of optimizer updates.
    lr : float
        Peak AdamW learning rate (reached after warmup).
    warmup_steps : int
        Linear warmup from 0 → lr over this many steps.
    min_lr : float
        Cosine decay floor — LR decays to this value by total_steps.
    log_interval : int
        Print progress every N steps.
    save_path : str or None
        Path for final checkpoint (None = no save).
    device_str : str
        ``"cpu"`` or ``"mps"``.
    game_type : GameType
        Mzinga game type for self-play.
    use_mz_ai : bool
        Use Mzinga's pretrained AlphaZero as move policy (default True).
    mz_simulations : int
        MCTS rollouts per move when ``use_mz_ai=True``.
    asymmetric : bool
        Alternate MCTS-vs-random sides per game to train value head.
    progress_fn : callable or None
        Called with per-metric averages at every log_interval.

    Returns
    -------
    HiveLiteModel
    """
    device = torch.device(device_str if torch.backends.mps.is_available() else "cpu")

    policy_label = "random"
    move_policy = None
    asymmetric_policies = None  # list of two alternating policies

    if use_mz_ai:
        print(f"Loading Mzinga AlphaZero (MCTS, {mz_simulations} sims)...")
        mz = MzingaMCTSAdapter(device=device_str, num_simulations=mz_simulations)
        print(f"  ✓ Mzinga AI loaded")
        if asymmetric:
            from mzinga.core.enums import PlayerColor
            from ghive_diffusion.dataset import make_random_policy

            rand = make_random_policy()

            def _white_strong(board):
                return (mz(board) if board.current_color == PlayerColor.White
                        else rand(board))

            def _black_strong(board):
                return (mz(board) if board.current_color == PlayerColor.Black
                        else rand(board))

            asymmetric_policies = [_white_strong, _black_strong]
            policy_label = "mz_white_strong (alternating)"
            print(f"  ✓ asymmetric play — White=MCTS/Black=random, alternating")
        else:
            move_policy = mz
            policy_label = f"mz_alphazero_{mz_simulations}sim"

    print(f"train_lite  device={device}  steps={total_steps}  lr={lr}  policy={policy_label}")
    print(f"  warmup={warmup_steps}  min_lr={min_lr}  "
          f"Estimated ~{total_steps // 40} games of self-play needed\n")

    model = build_lite_model()
    tokenizer = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tokenizer)
    model = model.to(device)

    trainer = HiveTrainer(
        model, tokenizer, builder,
        diffusion_schedule="cosine",
        sc_ramp_steps=500,
        self_condition_prob=0.5,
        max_grad_norm=5.0,
        device=device,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = create_scheduler(opt, warmup_steps=warmup_steps,
                                 total_steps=total_steps, min_lr=min_lr)

    # ── monitoring state ──────────────────────────────────────────
    loss_history: list[dict] = []
    grad_history: list[dict] = []
    param_history: list[dict] = []
    nan_steps: set[int] = set()
    running_losses: dict[str, float] = {}
    t_start = time.time()

    step = 0
    game_count = 0
    clip_count = 0
    clip_threshold = trainer.max_grad_norm * 0.95

    while step < total_steps:
        if asymmetric_policies is not None:
            cur_policy = asymmetric_policies[game_count % 2]
        else:
            cur_policy = move_policy
        gen = SelfPlayGenerator(tokenizer, builder, game_type=game_type,
                                move_policy=cur_policy)
        samples, _ = gen._play_one()
        game_count += 1

        for sample in samples:
            if step >= total_steps:
                break

            # ── training step ──────────────────────────────────
            metrics = trainer.step(sample, optimizer=opt)
            scheduler.step()
            step += 1

            # ── detect NaN ─────────────────────────────────────
            if math.isnan(metrics.get("loss", 0)):
                nan_steps.add(step)
                print(f"  ⚠ NaN loss at step {step} — skipping")
                continue

            # ── gradient stats (post-clip, pre-next-zerograd) ──
            gstats = _compute_grad_stats(model)
            clip_count += int(gstats["grad_l2"] >= clip_threshold)
            grad_history.append({"step": step, **gstats})

            # ── loss history ───────────────────────────────────
            loss_history.append({"step": step, **{k: v for k, v in metrics.items()
                                                  if isinstance(v, (int, float))}})

            # ── running averages ───────────────────────────────
            for k, v in metrics.items():
                running_losses[k] = running_losses.get(k, 0.0) + v

            # ── log interval ───────────────────────────────────
            if step % log_interval == 0:
                avg = {k: v / log_interval for k, v in running_losses.items()}
                avg["step"] = step
                avg["grad_l2"] = gstats["grad_l2"]
                avg["elapsed_s"] = time.time() - t_start
                avg["clip_frac"] = clip_count / step

                elapsed = avg["elapsed_s"]
                eta = (elapsed / step) * (total_steps - step) if step > 0 else 0
                eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta/60:.1f}m"

                print(
                    f"Step {step:>6d}/{total_steps}  "
                    f"loss={avg['loss']:>6.3f}  "
                    f"diff={avg['diffusion_loss']:>6.3f}  "
                    f"pol={avg['policy_loss']:>6.3f}  "
                    f"val={avg['value_loss']:>6.3f}  "
                    f"grad={gstats['grad_l2']:.3f}  "
                    f"|  v_pred={metrics.get('value_pred', 0):+.3f}  "
                    f"sc={metrics.get('effective_sc_prob', 0):.2f}  "
                    f"lr={scheduler.get_last_lr()[0]:.1e}  "
                    f"eta {eta_str}"
                )

                if progress_fn is not None:
                    progress_fn(avg)
                running_losses = {}

            # ── param stats (periodic) ─────────────────────────
            if step % 100 == 0:
                pstats = _compute_param_stats(model)
                param_history.append({"step": step, **pstats})

    elapsed = time.time() - t_start
    print(f"\nDone.  {step} steps, {game_count} games, {elapsed:.0f}s "
          f"({elapsed / step:.3f}s/step)")

    # ── save ──────────────────────────────────────────────────────
    if save_path:
        torch.save({
            "model": model.state_dict(),
            "step": step,
            "config": model.cfg,
        }, save_path)
        print(f"Saved to {save_path}")

    # ── report ────────────────────────────────────────────────────
    report = _generate_report({
        "losses": loss_history,
        "grads": grad_history,
        "params": param_history,
        "total_steps": step,
        "elapsed_s": elapsed,
        "device": str(device),
        "lr": lr,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "max_grad_norm": trainer.max_grad_norm,
        "asymmetric": asymmetric,
        "policy": policy_label,
        "clip_frac": clip_count / max(1, step),
        "nan_steps": nan_steps,
    })
    print("\n" + report)

    return model


# ---------------------------------------------------------------------------
# Dataset-based training (no Mzinga needed)
# ---------------------------------------------------------------------------


def train_from_dataset(dataset_path: str = "dataset.pt",
                       total_steps: int = 50000,
                       lr: float = 1e-4,
                       warmup_steps: int = 200,
                       min_lr: float = 1e-5,
                       log_interval: int = 500,
                       save_path: Optional[str] = "lite_model.pt",
                       device_str: str = "cuda",
                       ) -> HiveLiteModel:
    """Train a ``HiveLiteModel`` from a pre-generated dataset.

    .. deprecated::
        Prefer ``python -m ghive_diffusion_lite.pipeline`` which saves
        best-by-policy checkpoints mid-run and continues to eval/self-play.
        This function only writes ``save_path`` once at the end.

    No Mzinga dependency — loads ``LiteTrainingSample`` objects from
    *dataset_path* and trains on them.  Supports GPU training.

    Parameters
    ----------
    dataset_path : str
        Path to the saved ``.pt`` file (list of ``LiteTrainingSample``).
    total_steps : int
        Number of optimizer updates.  Cycles through the dataset.
    lr : float
        Peak learning rate.
    warmup_steps : int
        Linear warmup duration.
    min_lr : float
        Cosine decay floor.
    log_interval : int
        Print progress every N steps.
    save_path : str or None
        Path for final checkpoint.
    device_str : str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    HiveLiteModel
    """
    device = torch.device(device_str if torch.cuda.is_available()
                          or torch.backends.mps.is_available() else "cpu")

    print(f"train_from_dataset  device={device}  steps={total_steps}  lr={lr}")
    print(f"  warmup={warmup_steps}  min_lr={min_lr}")

    # ── load dataset ────────────────────────────────────────────
    print(f"  Loading {dataset_path}...")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    print(f"  ✓ {len(data)} samples loaded")

    # ── build model ─────────────────────────────────────────────
    model = build_lite_model()
    tokenizer = build_default_tokenizer(model.cfg)
    model = model.to(device)

    from ghive_diffusion.train_loop import create_scheduler

    trainer = LiteHiveTrainer(
        model, tokenizer,
        diffusion_schedule="cosine",
        sc_ramp_steps=warmup_steps,
        self_condition_prob=0.5,
        max_grad_norm=5.0,
        device=device,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = create_scheduler(opt, warmup_steps=warmup_steps,
                                 total_steps=total_steps, min_lr=min_lr)

    # ── training loop ───────────────────────────────────────────
    import random

    loss_history: list[dict] = []
    running_losses: dict[str, float] = {}
    t_start = time.time()
    epoch = 0
    idx = 0
    n_data = len(data)

    while trainer.step_count < total_steps:
        epoch += 1
        indices = list(range(n_data))
        random.shuffle(indices)
        for idx_inner, i in enumerate(indices):
            if trainer.step_count >= total_steps:
                break

            sample = data[i]
            metrics = trainer.step(sample, optimizer=opt)
            scheduler.step()

            for k, v in metrics.items():
                running_losses[k] = running_losses.get(k, 0.0) + v

            if trainer.step_count % log_interval == 0:
                avg = {k: v / log_interval
                       for k, v in running_losses.items()}
                elapsed = time.time() - t_start
                eta = (elapsed / trainer.step_count) * \
                    (total_steps - trainer.step_count) if trainer.step_count > 0 else 0
                eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta/60:.1f}m"

                print(
                    f"Step {trainer.step_count:>6d}/{total_steps}  "
                    f"loss={avg['loss']:>6.3f}  "
                    f"diff={avg['diffusion_loss']:>6.3f}  "
                    f"pol={avg['policy_loss']:>6.3f}  "
                    f"val={avg['value_loss']:>6.3f}  "
                    f"|  v_pred={metrics.get('value_pred', 0):+.3f}  "
                    f"lr={scheduler.get_last_lr()[0]:.1e}  "
                    f"eta {eta_str}"
                )
                loss_history.append({
                    "step": trainer.step_count,
                    **{k: v for k, v in metrics.items()
                       if isinstance(v, (int, float))},
                })
                running_losses = {}

    elapsed = time.time() - t_start
    print(f"\nDone.  {trainer.step_count} steps, {epoch} epochs, "
          f"{elapsed:.0f}s ({elapsed / trainer.step_count:.4f}s/step)")

    # ── save ────────────────────────────────────────────────────
    if save_path:
        torch.save({
            "model": model.state_dict(),
            "step": trainer.step_count,
            "config": model.cfg,
        }, save_path)
        print(f"Saved to {save_path}")

    # ── summary ─────────────────────────────────────────────────
    if loss_history:
        final = loss_history[-1]
        print(f"\nFinal (avg over last {log_interval} steps):")
        for k in sorted(final):
            if k not in ("step", "step_count"):
                print(f"  {k:>22s}: {final[k]:.4f}")

    return model
