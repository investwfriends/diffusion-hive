"""Multi-objective training step for ``HiveDiffusionModel`` (Phase 6/7).

The training step combines:

- diffusion denoising loss on a noisy canvas (the "next move" canvas)
- legal-policy loss over Mzinga legal moves
- value loss on the game outcome
- MoE load-balancing loss (Phase 6.5)
- self-conditioning (Phase 7)

The training step is designed to be the only place where gradients
flow. It returns a dictionary of named losses so callers can log them
individually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_builder import HiveContextBuilder
from .hive_model import HiveDiffusionModel
from .moe import RouterInfo
from .tokenizer import HiveTokenizer, build_default_tokenizer


@dataclass
class TrainingSample:
    """One supervised training sample."""
    context_ids: torch.Tensor       # (T_ctx,) long
    target_move_ids: torch.Tensor   # (T_canvas,) long — the next move, with <mask> tokens replaced during forward
    legal_move_ids: List[List[int]]  # list of tokenized legal moves
    target_legal_idx: int            # index of target_move in legal_moves
    value: float                     # from side-to-move perspective in [-1, 1]
    timestep: Optional[torch.Tensor] = None  # scalar (or vector) for diffusion noise level
    # Auxiliary targets (NEXT_STEPS 1.2) — list of int class labels, one per head.
    # None means "not computed"; the trainer skips aux loss in that case.
    aux_targets: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Auxiliary target computation (NEXT_STEPS 1.2)
# ---------------------------------------------------------------------------

# Head index -> (name, n_classes)
AUX_HEAD_SPECS = [
    (0, "game_phase_bucket",      3),   # 0=open, 1=midgame, 2=endgame
    (1, "legal_move_count_bucket", 5),  # 0:0-5, 1:6-10, 2:11-20, 3:21-40, 4:41+
    (2, "queen_in_play",          2),   # 0=no, 1=yes
    (3, "queen_placement_required", 2), # 0=no, 1=yes (turn 4 imminent)
    (4, "noisy_move_available",   2),   # 0=no, 1=yes
    (5, "pass_legal",             2),   # 0=no, 1=yes
    (6, "queen_surround_count",   4),   # 0,1,2,3+
    (7, "pinned_piece_count",     3),   # 0,1,2+
    (8, "mobility_advantage",     3),   # 0=negative, 1=neutral, 2=positive
]


def compute_aux_targets(board) -> List[int]:
    """Compute the 9 auxiliary target class labels from a Mzinga ``Board``.

    This function is cheap and does not modify the board.  It uses
    ``board.get_board_metrics()`` which internally may temporarily
    increment ``current_turn`` but restores it.
    """
    from mzinga.core.enums import PieceName, BoardState

    targets: List[int] = []

    # 0: game_phase_bucket — open (turns 1-10), midgame (11-30), endgame (31+)
    turn = board.current_turn
    if turn <= 10:
        targets.append(0)
    elif turn <= 30:
        targets.append(1)
    else:
        targets.append(2)

    # 1: legal_move_count_bucket
    n_legal = len(list(board.get_valid_moves()))
    if n_legal <= 5:
        targets.append(0)
    elif n_legal <= 10:
        targets.append(1)
    elif n_legal <= 20:
        targets.append(2)
    elif n_legal <= 40:
        targets.append(3)
    else:
        targets.append(4)

    # 2: queen_in_play (current side's queen)
    targets.append(1 if board.current_turn_queen_in_play else 0)

    # 3: queen_placement_required — turn 4 imminent (turns 3-4, queen not yet placed)
    if 3 <= turn <= 4 and not board.current_turn_queen_in_play:
        targets.append(1)
    else:
        targets.append(0)

    # 4: noisy_move_available — any move that is "noisy" (moves a beetle or pillbug)
    has_noisy = False
    for mv in board.get_valid_moves():
        if board.is_noisy_move(mv):
            has_noisy = True
            break
    targets.append(1 if has_noisy else 0)

    # 5: pass_legal
    legal_strs = []
    for mv in board.get_valid_moves():
        try:
            legal_strs.append(board.get_move_string(mv))
        except Exception:
            continue
    targets.append(1 if "pass" in legal_strs else 0)

    # 6: queen_surround_count — how many enemy pieces surround the current side's queen
    from mzinga.core.enums import PlayerColor
    queen = PieceName.wQ if board.current_color == PlayerColor.White else PieceName.bQ
    if board.piece_in_play(queen):
        _, _, enemy = board._count_neighbors(queen)
        targets.append(min(3, enemy))
    else:
        targets.append(0)

    # 7: pinned_piece_count — count pieces that are pinned (no moves available)
    metrics = board.get_board_metrics()
    pinned_count = 0
    start_piece = int(PieceName.wQ if board.current_color == PlayerColor.White else PieceName.bQ)
    end_piece = int(PieceName.bQ if board.current_color == PlayerColor.White else PieceName.NumPieceNames)
    for pn in range(start_piece, end_piece):
        piece_name = PieceName(pn)
        if metrics[piece_name].is_pinned:
            pinned_count += 1
    targets.append(min(2, pinned_count))

    # 8: mobility_advantage — compare current side's legal move count to opponent's
    # (approximated: if we have more moves than a threshold, positive; fewer, negative)
    if n_legal > 15:
        targets.append(2)
    elif n_legal < 6:
        targets.append(0)
    else:
        targets.append(1)

    return targets


class HiveTrainer:
    """Lightweight training step wrapper around :class:`HiveDiffusionModel`."""

    def __init__(self, model: HiveDiffusionModel,
                 tokenizer: Optional[HiveTokenizer] = None,
                 builder: Optional[HiveContextBuilder] = None,
                 diffusion_weight: float = 1.0,
                 policy_weight: float = 1.0,
                 value_weight: float = 0.5,
                 moe_lb_weight: float = 0.01,
                 self_condition_prob: float = 0.5,
                 sc_ramp_steps: int = 1000,
                 diffusion_schedule: str = "linear",
                 aux_weight: float = 0.1,
                 max_grad_norm: float = 1.0,
                 device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer or build_default_tokenizer(model.cfg)
        self.builder = builder or HiveContextBuilder(self.tokenizer)
        self.diffusion_weight = diffusion_weight
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.moe_lb_weight = moe_lb_weight
        self.aux_weight = aux_weight
        self.max_grad_norm = max_grad_norm
        self.self_condition_prob = self_condition_prob
        self.device = device or next(model.parameters()).device
        # Self-conditioning ramp-up (NEXT_STEPS 2.3).
        # effective_prob = base_prob * min(1, step / sc_ramp_steps)
        self.step_count: int = 0
        self.sc_ramp_steps: int = sc_ramp_steps
        # Diffusion noise schedule (NEXT_STEPS 2.2).
        # "linear" (default) or "cosine" (Nichol & Dhariwal).
        self.diffusion_schedule: str = diffusion_schedule

    # ----- helpers ---------------------------------------------------------

    def _diffusion_loss(self, sample: TrainingSample,
                        self_cond: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        """Compute denoising cross-entropy on the canvas tokens."""
        device = self.device
        ctx = sample.context_ids.unsqueeze(0).to(device)            # (1, T_ctx)
        target = sample.target_move_ids.to(device)                  # (T,)
        T_canvas = target.size(0)
        if T_canvas <= 0:
            return torch.tensor(0.0, device=device)

        # Sample a noise level (uniform over [0, 1]).
        if sample.timestep is None:
            t_scalar = torch.rand((), device=device)
        else:
            t_scalar = sample.timestep.to(device).reshape(())

        # Corrupt the target canvas with mask + random replacement.
        noisy = self.model.add_diffusion_noise(
            target.unsqueeze(0), self.tokenizer.mask_id,
            timesteps=t_scalar.unsqueeze(0),
            mask_prob=t_scalar.unsqueeze(0),
            vocab_size=self.tokenizer.vocab_size,
            schedule=self.diffusion_schedule,
        ).squeeze(0)

        # Forward: encode context, decode noisy canvas.
        with torch.no_grad():
            _, encoder_kv = self.model.forward_encoder(ctx, use_cache=True)
        h = self.model.forward_decoder(
            noisy.unsqueeze(0), encoder_kv,
            self_conditioning_embeds=self_cond,
            timestep=t_scalar,
        )
        logits = self.model._lm_head(h).squeeze(0)                # (T, V)
        # Mask out padding (target is right-padded with PAD_ID in caller).
        non_pad = (target != self.tokenizer.pad_id).float()
        ce = F.cross_entropy(logits, target, reduction="none")
        denom = non_pad.sum().clamp(min=1.0)
        return (ce * non_pad).sum() / denom

    def _policy_loss(self, sample: TrainingSample) -> torch.Tensor:
        device = self.device
        ctx = sample.context_ids.unsqueeze(0).to(device)
        legal_ids_list = [list(ids) for ids in sample.legal_move_ids]
        scores, value = self.model.score_legal_moves(
            ctx, legal_ids_list, use_value_head=True,
        )
        # Cross-entropy over the legal-move distribution.
        target_idx = torch.tensor([sample.target_legal_idx], dtype=torch.long, device=device)
        ce = F.cross_entropy(scores, target_idx)
        return ce

    def _value_loss(self, sample: TrainingSample,
                    value_pred: torch.Tensor) -> torch.Tensor:
        target = torch.tensor([sample.value], dtype=torch.float32, device=self.device)
        return F.mse_loss(value_pred.view(()), target.view(()))

    def _moe_load_balance_loss(self, records: List[RouterInfo]) -> torch.Tensor:
        """Switch-style load-balance loss from captured router statistics.

        For each layer with gradient-bearing records:

        - ``P_e`` = mean over tokens of ``softmax(all_scores)_e``  (differentiable router probability)
        - ``f_e`` = mean over tokens of scattered ``top_weights`` for expert ``e``  (differentiable expert load)
        - ``loss = E * sum_e(f_e * P_e)``

        The loss is minimised (= 1) when both ``f`` and ``P`` are uniform
        across experts, and grows toward ``E`` when a single expert
        dominates.  Records from no-grad passes (``requires_grad=False``)
        are skipped since they cannot contribute router gradients.
        """
        device = self.device
        valid = [r for r in records if r.all_scores.requires_grad]
        if not valid:
            return torch.tensor(0.0, device=device)
        losses = []
        for rec in valid:
            E = rec.all_scores.size(-1)
            probs = F.softmax(rec.all_scores, dim=-1)       # (B, T, E)
            P = probs.mean(dim=(0, 1))                        # (E,)
            f_token = torch.zeros_like(probs)
            f_token.scatter_add_(-1, rec.top_indices.long(), rec.top_weights)
            f = f_token.mean(dim=(0, 1))                      # (E,)
            losses.append(E * (f * P).sum())
        return torch.stack(losses).mean()

    def _aux_loss(self, sample: TrainingSample) -> torch.Tensor:
        """Compute the auxiliary-heads classification loss.

        Returns 0 if ``sample.aux_targets`` is None.
        """
        if sample.aux_targets is None:
            return torch.tensor(0.0, device=self.device)
        device = self.device
        ctx = sample.context_ids.unsqueeze(0).to(device)
        aux_logits = self.model.forward_aux_heads(ctx)
        total = torch.tensor(0.0, device=device)
        for i, (logits, target_cls) in enumerate(zip(aux_logits, sample.aux_targets)):
            target = torch.tensor([target_cls], dtype=torch.long, device=device)
            total = total + F.cross_entropy(logits, target)
        return total / len(aux_logits)

    # ----- training step ---------------------------------------------------

    def step(self, sample: TrainingSample, optimizer: Optional[torch.optim.Optimizer] = None
             ) -> Dict[str, float]:
        """Run one training step and return per-loss metrics.

        If ``optimizer`` is provided, performs backward + step.
        """
        device = self.device
        self.model.train()

        # Decide whether to use self-conditioning for this sample.
        # Ramp up from 0 to base_prob over sc_ramp_steps (NEXT_STEPS 2.3).
        ramp = min(1.0, self.step_count / max(1, self.sc_ramp_steps))
        effective_sc_prob = self.self_condition_prob * ramp
        use_self_cond = (torch.rand(()) < effective_sc_prob)

        # First pass: get self-conditioning logits (no grad).
        sc_logits = None
        if use_self_cond:
            with torch.no_grad():
                ctx = sample.context_ids.unsqueeze(0).to(device)
                _, encoder_kv = self.model.forward_encoder(ctx, use_cache=True)
                noisy = self.model.add_diffusion_noise(
                    sample.target_move_ids.unsqueeze(0).to(device),
                    self.tokenizer.mask_id,
                    timesteps=torch.tensor([0.5], device=device),
                    mask_prob=torch.tensor([0.5], device=device),
                    vocab_size=self.tokenizer.vocab_size,
                    schedule=self.diffusion_schedule,
                )
                h = self.model.forward_decoder(noisy, encoder_kv, timestep=torch.tensor(0.5, device=device))
                sc_logits = self.model._lm_head(h).detach()

        # Diffusion loss (with self-conditioning if used)
        # Begin MoE capture AFTER the no-grad self-conditioning pass so
        # only gradient-bearing forward passes contribute router stats.
        self.model.begin_moe_capture()
        diff_loss = self._diffusion_loss(sample, self_cond=sc_logits)

        # Policy + value loss
        ctx = sample.context_ids.unsqueeze(0).to(device)
        legal_ids_list = [list(ids) for ids in sample.legal_move_ids]
        scores, value = self.model.score_legal_moves(
            ctx, legal_ids_list, use_value_head=True,
        )
        target_idx = torch.tensor([sample.target_legal_idx], dtype=torch.long, device=device)
        pol_loss = F.cross_entropy(scores, target_idx)
        val_loss = self._value_loss(sample, value)

        # MoE load-balance loss from captured router statistics
        records = self.model.end_moe_capture()
        moe_loss = self._moe_load_balance_loss(records)

        # Auxiliary heads loss (NEXT_STEPS 1.2)
        aux_loss = self._aux_loss(sample)

        total = (self.diffusion_weight * diff_loss
                 + self.policy_weight * pol_loss
                 + self.value_weight * val_loss
                 + self.moe_lb_weight * moe_loss
                 + self.aux_weight * aux_loss)

        if optimizer is not None:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            optimizer.step()

        self.step_count += 1

        return {
            "loss": float(total.item()),
            "diffusion_loss": float(diff_loss.item()),
            "policy_loss": float(pol_loss.item()),
            "value_loss": float(val_loss.item()),
            "moe_loss": float(moe_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "value_pred": float(value.item()),
            "used_self_conditioning": float(use_self_cond),
            "effective_sc_prob": float(effective_sc_prob),
            "step_count": int(self.step_count),
        }