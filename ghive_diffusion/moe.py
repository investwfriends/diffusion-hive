import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
from typing import Optional


@dataclass
class MoERouterStats:
    """Raw router statistics from a single MoE layer forward (no metadata)."""
    top_indices: Tensor   # (B, T, k) long
    top_weights: Tensor   # (B, T, k) float — normalised gate weights over selected k
    all_scores: Tensor    # (B, T, E) float — pre-top-k sigmoid scores


@dataclass
class RouterInfo:
    """Router statistics tagged with layer and stack metadata.

    Used by the trainer to build the load-balance loss and by the
    metrics tracker to report expert usage / entropy.
    """
    top_indices: Tensor   # (B, T, k) long
    top_weights: Tensor   # (B, T, k) float
    all_scores: Tensor    # (B, T, E) float
    layer_idx: int
    stack: str            # "encoder" or "decoder"


class Expert(nn.Module):
    def __init__(self, hidden_size: int, moe_intermediate: int, act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.gate = nn.Linear(hidden_size, moe_intermediate, bias=False)
        self.up   = nn.Linear(hidden_size, moe_intermediate, bias=False)
        self.down = nn.Linear(moe_intermediate, hidden_size, bias=False)
        self.act_fn = F.gelu  # tanh-approximate; F.gelu has approximate='tanh' option

    def forward(self, x: Tensor) -> Tensor:
        # gelu_pytorch_tanh: 0.5 * x * (1 + tanh(√(2/π)(x + 0.044715 x³)))
        return self.down(self.act_fn(self.gate(x), approximate="tanh") * self.up(x))


class MoELayer(nn.Module):
    """
    Gemma 4 MoE:
      router = sigmoid(W·x)  (no softmax, no renormalisation)
      pick top-K experts per token, weight them by router score (normalised over the K)
      plus 1 dense "shared" expert applied to every token.
    """
    def __init__(self, hidden_size: int, moe_intermediate: int,
                 num_experts: int = 128, top_k: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(hidden_size, moe_intermediate) for _ in range(num_experts))
        self.shared_expert = Expert(hidden_size, moe_intermediate)
        self.last_router_info: Optional[MoERouterStats] = None

    def forward(self, x: Tensor) -> Tensor:
        b, t, h = x.shape
        scores = torch.sigmoid(self.gate(x))           # (b, t, E)
        # top-k
        topv, topi = scores.topk(self.top_k, dim=-1)  # (b, t, k)
        topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
        # Store router statistics for load-balancing loss / metrics
        self.last_router_info = MoERouterStats(
            top_indices=topi,
            top_weights=topv,
            all_scores=scores,
        )
        # dispatch
        flat_x = x.reshape(b * t, h)
        out = torch.zeros_like(flat_x)
        idx_flat = topi.reshape(-1, self.top_k)         # (b*t, k)
        w_flat   = topv.reshape(-1, self.top_k)
        for e in range(self.num_experts):
            mask = (idx_flat == e)                      # (b*t, k)
            if not mask.any():
                continue
            tok_idx, slot_idx = mask.nonzero(as_tuple=True)
            y = self.experts[e](flat_x[tok_idx])
            w = w_flat[tok_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, tok_idx, w * y)
        out = out.view(b, t, h)
        return out + self.shared_expert(x)
