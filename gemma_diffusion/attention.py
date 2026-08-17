import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from .utils import _softcap

def _rope_freqs(head_dim: int, theta: float, max_pos: int = 131072,
                proportional: bool = False) -> Tensor:
    """Return cos/sin tables of shape (max_pos, head_dim/2)."""
    half = head_dim // 2
    # standard: 1/theta^(2i/d); proportional: 1/theta^(2i/(pi*d))
    if proportional:
        exponents = torch.arange(0, half, dtype=torch.float32) / half
        inv_freq = 1.0 / (theta ** exponents)
    else:
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))
    pos = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)              # (max_pos, half)
    return freqs.cos().to(torch.float32), freqs.sin().to(torch.float32)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor,
                partial_rotary_factor: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Apply RoPE; rotates only the first `partial*head_dim` dims of each head.

    Accepts cos/sin either as 2-D `(max_pos, head_dim/2)` raw buffers
    or 4-D `(B|1, 1, T, head_dim/2)` already gathered by the caller.
    """
    b, h, t, d = q.shape
    pr = max(2, int(d * partial_rotary_factor))      # ≥ 2 so chunk(2) is well-defined
    pr -= pr % 2                                     # round down to even
    if cos.dim() == 2:                               # raw buffer path
        cos = cos[:t].unsqueeze(0).unsqueeze(0)
        sin = sin[:t].unsqueeze(0).unsqueeze(0)
    # *** Partial RoPE: use only the first pr/2 (lowest-frequency) rotary embeddings. ***
    if cos.size(-1) > pr // 2:
        cos = cos[..., : pr // 2]
        sin = sin[..., : pr // 2]

    def rot(x: Tensor) -> Tensor:
        x_rot, x_pass = x[..., :pr], x[..., pr:]
        x1, x2 = x_rot.chunk(2, dim=-1)
        out_rot = torch.cat([x1 * cos - x2 * sin,
                             x2 * cos + x1 * sin], dim=-1)
        return torch.cat([out_rot, x_pass], dim=-1)
    return rot(q), rot(k)


class GemmaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, sliding_window: int, layer_type: str,
                 rope_theta: float, rope_proportional: bool,
                 partial_rotary_factor: float, max_pos: int,
                 softcap: float = 30.0, dropout: float = 0.0):
        super().__init__()
        assert layer_type in ("sliding_attention", "full_attention")
        self.layer_type = layer_type
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.softcap = softcap
        self.partial_rotary_factor = partial_rotary_factor
        self.dropout = dropout
        # If this is a "global" layer, the config has a *different* head_dim.
        # We support both by accepting the appropriate per-layer head_dim.
        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_out, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.o_proj = nn.Linear(q_out, hidden_size, bias=False)
        # RoPE tables
        self.register_buffer("rope_cos", _rope_freqs(head_dim, rope_theta, max_pos, rope_proportional)[0],
                             persistent=False)
        self.register_buffer("rope_sin", _rope_freqs(head_dim, rope_theta, max_pos, rope_proportional)[1],
                             persistent=False)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_kv: Optional[Tuple[Tensor, Tensor]] = None,
            use_cache: bool = False,
            is_bidir: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
      b, t, _ = x.shape
      q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
      k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
      v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

      if position_ids is None:
          position_ids = torch.arange(t, device=x.device).unsqueeze(0)
      cos = self.rope_cos.to(x.dtype)
      sin = self.rope_sin.to(x.dtype)
      cos = cos[position_ids].unsqueeze(1)        # (b, 1, t, head_dim/2)
      sin = sin[position_ids].unsqueeze(1)
      q, k = _apply_rope(q, k, cos, sin, self.partial_rotary_factor)

      if past_kv is not None:
          pk, pv = past_kv
          k = torch.cat([pk, k], dim=2)
          v = torch.cat([pv, v], dim=2)
      new_kv = (k, v) if use_cache else None
      T = k.size(2)

      rep = self.num_heads // self.num_kv_heads
      if rep > 1:
          k = k.repeat_interleave(rep, dim=1)
          v = v.repeat_interleave(rep, dim=1)

      # Build additive attention bias (1, 1, t, T) with -inf on disallowed positions
      attn_bias = torch.zeros(t, T, device=x.device, dtype=x.dtype)
      if not is_bidir:
          causal = torch.ones(t, T, dtype=torch.bool, device=x.device).tril(T - t)
          attn_bias = attn_bias.masked_fill(~causal, float("-inf"))
      if self.layer_type == "sliding_attention" and self.sliding_window is not None:
          qpos = torch.arange(t, device=x.device) + (T - t)
          kpos = torch.arange(T, device=x.device)
          win = (qpos.unsqueeze(1) - kpos.unsqueeze(0)) <= self.sliding_window
          win &= (qpos.unsqueeze(1) - kpos.unsqueeze(0)) >= 0
          attn_bias = attn_bias.masked_fill(~win, float("-inf"))
      if attn_mask is not None:
          if attn_mask.dim() == 2:
              attn_bias = attn_bias + attn_mask[:, -t:]
          else:
              attn_bias = attn_bias + attn_mask[..., -t:, :]
      attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

      # Manual attention so we can apply Gemma 4's logit soft-capping pre-softmax
      scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
      scores = scores + attn_bias
      if self.softcap and self.softcap > 0:
          scores = _softcap(scores, self.softcap)
      probs = scores.softmax(dim=-1)
      if self.dropout and self.training:
          probs = F.dropout(probs, p=self.dropout)
      out = probs @ v

      out = out.transpose(1, 2).contiguous().view(b, t, -1)
      out = self.o_proj(out)
      return out, new_kv
