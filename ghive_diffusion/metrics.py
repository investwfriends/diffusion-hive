"""Evaluation metrics for ``HiveDiffusionModel`` (Phase 11).

This module provides the metric tracker described in the adaptation
plan. Categories:

- Legality — parse failure rate, illegal move rate (should be 0 after
  Mzinga projection), canonical roundtrip, pass misuse, expansion-piece
  misuse, legal top-k accuracy.
- Strength — win/draw/loss vs random and prior checkpoints.
- Hive-specific — queen placement timing, mobility, etc.
- Diffusion — loss by timestep, entropy by step, candidate diversity.
- MoE — expert usage histogram, dead experts, router entropy.

The primary entry point is :class:`MetricsTracker`, which records
samples and aggregates them into summary statistics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from mzinga.core.board import Board
from mzinga.core.enums import GameType


@dataclass
class LegalityMetrics:
    total: int = 0
    parse_failures: int = 0
    illegal_pre_projection: int = 0
    illegal_post_projection: int = 0
    canonical_roundtrip_failures: int = 0
    pass_when_illegal: int = 0
    pass_when_legal: int = 0
    expansion_piece_misuse: int = 0
    legal_top1_correct: int = 0
    legal_top3_correct: int = 0
    legal_top5_correct: int = 0

    def update_topk(self, target_idx: int, ranked_indices: Sequence[int]) -> None:
        if target_idx == ranked_indices[0]:
            self.legal_top1_correct += 1
        if target_idx in ranked_indices[:3]:
            self.legal_top3_correct += 1
        if target_idx in ranked_indices[:5]:
            self.legal_top5_correct += 1

    def topk_acc(self, k: int) -> float:
        total = max(1, self.total)
        if k == 1:
            return self.legal_top1_correct / total
        if k == 3:
            return self.legal_top3_correct / total
        if k == 5:
            return self.legal_top5_correct / total
        return 0.0

    def summary(self) -> Dict[str, float]:
        total = max(1, self.total)
        return {
            "parse_failure_rate": self.parse_failures / total,
            "illegal_pre_projection_rate": self.illegal_pre_projection / total,
            "illegal_post_projection_rate": self.illegal_post_projection / total,
            "canonical_roundtrip_failure_rate": self.canonical_roundtrip_failures / total,
            "pass_misuse_rate": self.pass_when_illegal / total,
            "expansion_piece_misuse_rate": self.expansion_piece_misuse / total,
            "legal_top1_acc": self.legal_top1_correct / total,
            "legal_top3_acc": self.legal_top3_correct / total,
            "legal_top5_acc": self.legal_top5_correct / total,
        }


@dataclass
class StrengthMetrics:
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def record(self, outcome: str) -> None:
        self.games_played += 1
        if outcome == "win":
            self.wins += 1
        elif outcome == "draw":
            self.draws += 1
        elif outcome == "loss":
            self.losses += 1

    def summary(self) -> Dict[str, float]:
        g = max(1, self.games_played)
        return {
            "games": self.games_played,
            "win_rate": self.wins / g,
            "draw_rate": self.draws / g,
            "loss_rate": self.losses / g,
        }


@dataclass
class DiffusionMetrics:
    loss_by_timestep: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    entropy_by_step: List[float] = field(default_factory=list)
    candidate_diversity: List[int] = field(default_factory=list)
    accepted_per_step: List[int] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for bucket, losses in self.loss_by_timestep.items():
            if losses:
                out[f"diff_loss_{bucket}"] = float(np.mean(losses))
        if self.entropy_by_step:
            out["mean_entropy"] = float(np.mean(self.entropy_by_step))
        if self.candidate_diversity:
            out["mean_candidate_diversity"] = float(np.mean(self.candidate_diversity))
        if self.accepted_per_step:
            out["mean_accepted_per_step"] = float(np.mean(self.accepted_per_step))
        return out


@dataclass
class MoEMetrics:
    num_experts_configured: int = 0
    expert_token_counts: Counter = field(default_factory=Counter)
    expert_usage_by_layer: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    router_entropy_by_layer: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def num_experts(self) -> int:
        return self.num_experts_configured or len(self.expert_token_counts)

    @property
    def dead_experts(self) -> int:
        if self.num_experts_configured > 0:
            return self.num_experts_configured - len(self.expert_token_counts)
        return sum(1 for c in self.expert_token_counts.values() if c == 0)

    def record_layer(self, layer_idx: int, top_indices: np.ndarray,
                     router_probs: Optional[np.ndarray] = None) -> None:
        for idx in top_indices.flatten().tolist():
            self.expert_token_counts[int(idx)] += 1
            self.expert_usage_by_layer[layer_idx][int(idx)] += 1
        if router_probs is not None and router_probs.size:
            # entropy over the avg expert distribution for this layer
            avg = router_probs.mean(axis=tuple(range(router_probs.ndim - 1)))
            avg = np.clip(avg, 1e-12, 1.0)
            avg = avg / avg.sum()
            ent = -float(np.sum(avg * np.log(avg)))
            self.router_entropy_by_layer[layer_idx].append(ent)

    def record_router_info(self, info) -> None:
        """Record router statistics from a :class:`RouterInfo` instance.

        Accepts ``ghive_diffusion.moe.RouterInfo`` and handles the
        torch-to-numpy conversion internally.
        """
        E = info.all_scores.size(-1)
        if E > self.num_experts_configured:
            self.num_experts_configured = E

        top_indices_np = info.top_indices.detach().cpu().numpy()
        all_scores_np = info.all_scores.detach().cpu().numpy()

        for idx in top_indices_np.flatten().tolist():
            self.expert_token_counts[int(idx)] += 1
            self.expert_usage_by_layer[info.layer_idx][int(idx)] += 1

        if all_scores_np.size:
            avg = all_scores_np.mean(axis=tuple(range(all_scores_np.ndim - 1)))
            avg = np.clip(avg, 1e-12, 1.0)
            avg = avg / avg.sum()
            ent = -float(np.sum(avg * np.log(avg)))
            self.router_entropy_by_layer[info.layer_idx].append(ent)

    def summary(self) -> Dict[str, float]:
        if self.num_experts_configured > 0:
            counts = np.zeros(self.num_experts_configured, dtype=np.float64)
            for k, v in self.expert_token_counts.items():
                if k < self.num_experts_configured:
                    counts[k] = v
        else:
            counts = np.array(list(self.expert_token_counts.values()), dtype=np.float64)
        if counts.sum() == 0:
            return {
                "dead_experts": 0,
                "router_entropy": 0.0,
                "top_expert_share": 0.0,
            }
        usage = counts / counts.sum()
        usage_safe = np.clip(usage, 1e-12, 1.0)
        ent = -float(np.sum(usage_safe * np.log(usage_safe)))
        out = {
            "dead_experts": int(self.dead_experts),
            "router_entropy": ent,
            "top_expert_share": float(usage.max()),
        }
        avg_layer_ent = []
        for layer_idx, ents in self.router_entropy_by_layer.items():
            if ents:
                avg_layer_ent.append(np.mean(ents))
        if avg_layer_ent:
            out["mean_layer_router_entropy"] = float(np.mean(avg_layer_ent))
        return out


@dataclass
class MetricsTracker:
    """Aggregate per-category metrics for a single evaluation run."""

    legality: LegalityMetrics = field(default_factory=LegalityMetrics)
    strength: StrengthMetrics = field(default_factory=StrengthMetrics)
    diffusion: DiffusionMetrics = field(default_factory=DiffusionMetrics)
    moe: MoEMetrics = field(default_factory=MoEMetrics)

    def record_legality_sample(self, *, parse_ok: bool, illegal_pre: bool,
                               illegal_post: bool, roundtrip_ok: bool,
                               pass_legal: bool, pass_chosen: bool,
                               expansion_legal: bool, expansion_chosen: bool,
                               target_idx: int, ranked_indices: Sequence[int]) -> None:
        self.legality.total += 1
        if not parse_ok:
            self.legality.parse_failures += 1
        if illegal_pre:
            self.legality.illegal_pre_projection += 1
        if illegal_post:
            self.legality.illegal_post_projection += 1
        if not roundtrip_ok:
            self.legality.canonical_roundtrip_failures += 1
        if pass_chosen and not pass_legal:
            self.legality.pass_when_illegal += 1
        if pass_chosen and pass_legal:
            self.legality.pass_when_legal += 1
        if expansion_chosen and not expansion_legal:
            self.legality.expansion_piece_misuse += 1
        self.legality.update_topk(target_idx, ranked_indices)

    def record_game_outcome(self, outcome: str) -> None:
        self.strength.record(outcome)

    def record_diffusion_step(self, timestep: float, loss: float,
                              entropy: Optional[float] = None,
                              accepted: Optional[int] = None) -> None:
        bucket = f"{int(timestep * 5) / 5:.1f}"
        self.diffusion.loss_by_timestep[bucket].append(float(loss))
        if entropy is not None:
            self.diffusion.entropy_by_step.append(float(entropy))
        if accepted is not None:
            self.diffusion.accepted_per_step.append(int(accepted))

    def record_candidate_set(self, n_unique: int) -> None:
        self.diffusion.candidate_diversity.append(int(n_unique))

    def record_moe_layer(self, layer_idx: int, top_indices: np.ndarray,
                         router_probs: Optional[np.ndarray] = None) -> None:
        self.moe.record_layer(layer_idx, top_indices, router_probs)

    def record_moe_router_info(self, info) -> None:
        """Record MoE router statistics from a :class:`RouterInfo`."""
        self.moe.record_router_info(info)

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update({f"legality/{k}": v for k, v in self.legality.summary().items()})
        out.update({f"strength/{k}": v for k, v in self.strength.summary().items()})
        out.update({f"diffusion/{k}": v for k, v in self.diffusion.summary().items()})
        out.update({f"moe/{k}": v for k, v in self.moe.summary().items()})
        return out