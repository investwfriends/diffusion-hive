"""Mzinga-free training step for pre-generated datasets.

``LiteTrainingSample`` — same fields as ``TrainingSample``, but defined
here so loading a saved dataset doesn't require importing
``ghive_diffusion.training`` (which pulls in Mzinga at module level).

``LiteHiveTrainer`` — replicates the five-loss training step from
``HiveTrainer.step()`` but imports nothing from Mzinga.

Both run anywhere (Colab, MPS, CPU) without the Mzinga package.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ghive_diffusion.tokenizer import HiveTokenizer, build_default_tokenizer

from .hive_lite_config import HiveLiteConfig
from .hive_lite_model import HiveLiteModel, build_lite_model


@dataclass
class LiteTrainingSample:
    """One supervised training sample — same fields as TrainingSample."""
    context_ids: Tensor          # (T_ctx,) long
    target_move_ids: Tensor      # (T_canvas,) long
    legal_move_ids: List[List[int]]
    target_legal_idx: int
    value: float
    timestep: Optional[Tensor] = None
    aux_targets: Optional[List[int]] = None
    # True when ``value`` is a terminal game outcome (±1) rather than a noisy
    # MCTS root value. Used to up-weight decisive value samples during training.
    has_outcome: bool = False


class LiteHiveTrainer:
    """Mzinga-free training step for ``HiveLiteModel``.

    Same five-loss computation as ``HiveTrainer.step()`` but depends
    only on torch + tokenizer + model.  No context_builder, no
    MoERouter, no Mzinga.
    """

    def __init__(self, model: HiveLiteModel,
                 tokenizer: Optional[HiveTokenizer] = None,
                 diffusion_weight: float = 1.0,
                 policy_weight: float = 1.0,
                 value_weight: float = 0.5,
                 aux_weight: float = 0.1,
                 self_condition_prob: float = 0.5,
                 sc_ramp_steps: int = 500,
                 diffusion_schedule: str = "cosine",
                 max_grad_norm: float = 5.0,
                 total_steps: int = 20000,
                 dynamic_weights: bool = True,
                 teacher_value_weight: float = 0.05,
                 outcome_value_weight: float = 1.0,
                 value_rank_buffer_size: int = 256,
                 device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer or build_default_tokenizer(model.cfg)
        self.diffusion_weight = diffusion_weight
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.aux_weight = aux_weight
        self.self_condition_prob = self_condition_prob
        self.sc_ramp_steps = sc_ramp_steps
        self.diffusion_schedule = diffusion_schedule
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.dynamic_weights = dynamic_weights
        self.teacher_value_weight = teacher_value_weight
        self.outcome_value_weight = outcome_value_weight
        self.device = device or next(model.parameters()).device
        self.step_count: int = 0
        self._value_buffer: List[Tuple[float, float]] = []  # (target, pred) for ranking
        self._value_buffer_size = value_rank_buffer_size

    # ── loss helpers ─────────────────────────────────────────────

    def _diffusion_loss(self, sample: LiteTrainingSample,
                        self_cond: Optional[Tensor] = None,
                        encoder_kv: Optional[list] = None) -> Tensor:
        device = self.device
        ctx = sample.context_ids.unsqueeze(0).to(device)
        target = sample.target_move_ids.to(device)
        T_canvas = target.size(0)
        if T_canvas <= 0:
            return torch.tensor(0.0, device=device)

        if sample.timestep is None:
            t_scalar = torch.rand((), device=device)
        else:
            t_scalar = sample.timestep.to(device).reshape(())

        noisy = self.model.add_diffusion_noise(
            target.unsqueeze(0), self.tokenizer.mask_id,
            timesteps=t_scalar.unsqueeze(0),
            mask_prob=t_scalar.unsqueeze(0),
            vocab_size=self.tokenizer.vocab_size,
            schedule=self.diffusion_schedule,
        ).squeeze(0)

        if encoder_kv is None:
            with torch.no_grad():
                _, encoder_kv = self.model.forward_encoder(ctx, use_cache=True)
        h = self.model.forward_decoder(
            noisy.unsqueeze(0), encoder_kv,
            self_conditioning_embeds=self_cond,
            timestep=t_scalar,
        )
        logits = self.model._lm_head(h).squeeze(0)
        non_pad = (target != self.tokenizer.pad_id).float()
        ce = F.cross_entropy(logits, target, reduction="none")
        denom = non_pad.sum().clamp(min=1.0)
        return (ce * non_pad).sum() / denom

    def _policy_loss(self, sample: LiteTrainingSample) -> Tensor:
        device = self.device
        ctx = sample.context_ids.unsqueeze(0).to(device)
        legal_ids_list = [list(ids) for ids in sample.legal_move_ids]
        scores, _value = self.model.score_legal_moves(
            ctx, legal_ids_list, use_value_head=True)
        target_idx = torch.tensor([sample.target_legal_idx],
                                   dtype=torch.long, device=device)
        return F.cross_entropy(scores, target_idx)

    def _value_loss(self, sample: LiteTrainingSample,
                    value_pred: Tensor) -> Tensor:
        target = torch.tensor([sample.value], dtype=torch.float32,
                              device=self.device)
        mse = F.mse_loss(value_pred.view(()), target.view(()))
        # Decisive terminal outcomes drive value learning; tiny teacher root
        # values are kept only as a mild regularizer.  Infer outcomes from the
        # flag or from the magnitude of the target so old datasets and self-play
        # backfills are treated consistently.
        is_outcome = getattr(sample, "has_outcome", False) or abs(float(sample.value)) >= 0.9
        weight = self.outcome_value_weight if is_outcome else self.teacher_value_weight
        return weight * mse

    def _update_value_buffer(self, target: float, pred: float):
        """Keep a rolling buffer of decisive outcome samples for value metrics."""
        if abs(target) < 0.9:
            return
        self._value_buffer.append((target, pred))
        if len(self._value_buffer) > self._value_buffer_size:
            self._value_buffer.pop(0)

    def _value_metrics(self) -> Dict[str, float]:
        """Compute separation and pairwise-ranking accuracy on outcome buffer.

        Returns an empty dict when the buffer isn't yet decisive, so callers
        can skip averaging NaNs across a training log interval.
        """
        metrics: Dict[str, float] = {}
        if len(self._value_buffer) < 2:
            return metrics
        pos_preds = [p for t, p in self._value_buffer if t > 0.5]
        neg_preds = [p for t, p in self._value_buffer if t < -0.5]
        if pos_preds and neg_preds:
            metrics["value_separation"] = (sum(pos_preds) / len(pos_preds)) - (
                sum(neg_preds) / len(neg_preds)
            )

        # Sample 64 random pairs from the buffer for a cheap, stable ranking
        # estimate. Accuracy = P(pred_i > pred_j | target_i > target_j + margin).
        buf = self._value_buffer
        n = len(buf)
        n_pairs = min(64, n * (n - 1) // 2)
        correct = total = 0
        margin = 0.2
        for _ in range(n_pairs):
            i, j = random.sample(range(n), 2)
            ti, pi = buf[i]
            tj, pj = buf[j]
            if abs(ti - tj) < margin:
                continue
            total += 1
            if (ti > tj and pi > pj) or (ti < tj and pi < pj):
                correct += 1
        if total > 0:
            metrics["value_ranking_acc"] = correct / total

        return metrics

    def _aux_loss(self, sample: LiteTrainingSample) -> Tensor:
        if sample.aux_targets is None:
            return torch.tensor(0.0, device=self.device)
        device = self.device
        ctx = sample.context_ids.unsqueeze(0).to(device)
        aux_logits = self.model.forward_aux_heads(ctx)
        total = torch.tensor(0.0, device=device)
        for logits, target_cls in zip(aux_logits, sample.aux_targets):
            t = torch.tensor([target_cls], dtype=torch.long, device=device)
            total = total + F.cross_entropy(logits, t)
        return total / len(aux_logits)

    # ── training step ────────────────────────────────────────────

    def _weights(self) -> Tuple[float, float, float, float]:
        if self.dynamic_weights:
            r = min(1.0, float(self.step_count) / max(1.0, float(self.total_steps)))
            w_diff = max(0.1, self.diffusion_weight * (1.0 - 0.85 * r))
            w_pol  = self.policy_weight * (1.0 + 1.0 * r)
            w_val  = self.value_weight * (1.0 + 1.0 * r)
            w_aux  = self.aux_weight
        else:
            w_diff, w_pol, w_val, w_aux = self.diffusion_weight, self.policy_weight, self.value_weight, self.aux_weight
        return w_diff, w_pol, w_val, w_aux

    def _sample_losses(self, sample: LiteTrainingSample):
        """Compute the four loss tensors + extras for one sample.

        Returns (diff_loss, pol_loss, val_loss, aux_loss, value_pred,
        policy_top1, used_sc, effective_sc).
        """
        device = self.device

        ramp = min(1.0, self.step_count / max(1, self.sc_ramp_steps))
        effective_sc = self.self_condition_prob * ramp
        use_sc = (torch.rand(()) < effective_sc)

        ctx = sample.context_ids.unsqueeze(0).to(device)

        sc_logits = None
        if use_sc:
            with torch.no_grad():
                _, sc_kv = self.model.forward_encoder(ctx, use_cache=True)
                noisy = self.model.add_diffusion_noise(
                    sample.target_move_ids.unsqueeze(0).to(device),
                    self.tokenizer.mask_id,
                    timesteps=torch.tensor([0.5], device=device),
                    mask_prob=torch.tensor([0.5], device=device),
                    vocab_size=self.tokenizer.vocab_size,
                    schedule=self.diffusion_schedule,
                )
                h_sc = self.model.forward_decoder(
                    noisy, sc_kv, timestep=torch.tensor(0.5, device=device))
                sc_logits = self.model._lm_head(h_sc).detach()

        # One shared encoder pass (grad) feeds policy, value and aux heads.
        # The diffusion loss gets a detached copy (it must not train the
        # encoder, matching the original no_grad behaviour).
        h_enc, encoder_kv = self.model.forward_encoder(ctx, use_cache=True)

        diff_loss = self._diffusion_loss(
            sample, self_cond=sc_logits,
            encoder_kv=[(k.detach(), v.detach()) for k, v in encoder_kv])

        legal_ids_list = [list(ids) for ids in sample.legal_move_ids]
        scores, _ = self.model.score_legal_moves(
            ctx, legal_ids_list, use_value_head=False, encoder_kv=encoder_kv)
        target_idx = torch.tensor([sample.target_legal_idx],
                                   dtype=torch.long, device=device)
        pol_loss = F.cross_entropy(scores, target_idx)
        policy_top1 = float(
            (scores.argmax(dim=-1) == target_idx).float().item())

        last = h_enc[:, -1:, :]
        value = self.model.value_head(last).squeeze(-1).squeeze(-1)
        val_loss = self._value_loss(sample, value)

        if sample.aux_targets is not None:
            aux_logits = [head(last).squeeze(1) for head in self.model.aux_heads]
            aux_loss = torch.tensor(0.0, device=device)
            for logits, target_cls in zip(aux_logits, sample.aux_targets):
                t = torch.tensor([target_cls], dtype=torch.long, device=device)
                aux_loss = aux_loss + F.cross_entropy(logits, t)
            aux_loss = aux_loss / len(aux_logits)
        else:
            aux_loss = torch.tensor(0.0, device=device)

        return (diff_loss, pol_loss, val_loss, aux_loss,
                float(value.item()), policy_top1, float(use_sc), effective_sc)

    def step(self, sample: LiteTrainingSample,
             optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        self.model.train()
        self.model.begin_moe_capture()
        (diff_loss, pol_loss, val_loss, aux_loss,
         v_pred, top1, used_sc, effective_sc) = self._sample_losses(sample)
        self.model.end_moe_capture()

        self._update_value_buffer(float(sample.value), v_pred)
        val_metrics = self._value_metrics()

        w_diff, w_pol, w_val, w_aux = self._weights()
        total = (w_diff * diff_loss
                 + w_pol * pol_loss
                 + w_val * val_loss
                 + w_aux * aux_loss)

        if optimizer is not None:
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.max_grad_norm)
            optimizer.step()

        self.step_count += 1

        return {
            "loss": float(total.item()),
            "diffusion_loss": float(diff_loss.item()),
            "policy_loss": float(pol_loss.item()),
            "value_loss": float(val_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "value_pred": v_pred,
            "policy_top1": top1,
            "used_self_conditioning": used_sc,
            "effective_sc_prob": float(effective_sc),
            "step_count": int(self.step_count),
            **val_metrics,
        }

    def step_batch(self, samples: List[LiteTrainingSample],
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """Batched training step: one backward/optimizer step per batch.

        Each sample is scored independently (legal-move counts differ), the
        per-sample weighted losses are averaged, and a single backward pass
        is taken.  ``step_count`` counts optimizer steps, not samples.
        """
        self.model.train()
        device = self.device
        n = max(1, len(samples))

        sums: Dict[str, float] = {
            "diffusion_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
            "aux_loss": 0.0, "value_pred": 0.0, "policy_top1": 0.0,
            "used_self_conditioning": 0.0,
        }
        total = torch.zeros((), device=device)
        w_diff, w_pol, w_val, w_aux = self._weights()

        self.model.begin_moe_capture()
        for sample in samples:
            (diff_loss, pol_loss, val_loss, aux_loss,
             v_pred, top1, used_sc, _sc) = self._sample_losses(sample)
            total = total + (w_diff * diff_loss
                             + w_pol * pol_loss
                             + w_val * val_loss
                             + w_aux * aux_loss) / n
            sums["diffusion_loss"] += float(diff_loss.item())
            sums["policy_loss"] += float(pol_loss.item())
            sums["value_loss"] += float(val_loss.item())
            sums["aux_loss"] += float(aux_loss.item())
            sums["value_pred"] += v_pred
            sums["policy_top1"] += top1
            sums["used_self_conditioning"] += used_sc
            self._update_value_buffer(float(sample.value), v_pred)
        self.model.end_moe_capture()

        val_metrics = self._value_metrics()

        if optimizer is not None:
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.max_grad_norm)
            optimizer.step()

        self.step_count += 1

        return {
            "loss": float(total.item()),
            **{k: v / n for k, v in sums.items()},
            "effective_sc_prob": float(self.self_condition_prob),
            "batch_size": n,
            "step_count": int(self.step_count),
            **val_metrics,
        }
