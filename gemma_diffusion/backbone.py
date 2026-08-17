import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from .attention import GemmaAttention
from .moe import MoELayer
from .utils import GemmaRMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int, head_dim: int, num_kv_heads: int,
                 is_cross: bool = False):
        super().__init__()
        self.cfg = cfg
        self.is_cross = is_cross
        self.attn = GemmaAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=cfg.sliding_window,
            layer_type=cfg.layer_types[layer_idx],
            rope_theta=cfg.rope_theta_for(cfg.layer_types[layer_idx]),
            rope_proportional=(cfg.layer_types[layer_idx] == "full_attention"),
            partial_rotary_factor=cfg.partial_rotary_factor_for(cfg.layer_types[layer_idx]),
            max_pos=cfg.max_position_embeddings,
            softcap=cfg.final_logit_softcapping,
        )
        self.attn_norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp_norm  = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MoELayer(
            hidden_size=cfg.hidden_size,
            moe_intermediate=cfg.moe_intermediate_size,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k_experts,
        )

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                past_kv: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False,
                is_bidir: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        h = self.attn_norm(x)
        a, new_kv = self.attn(h, attn_mask=attn_mask, position_ids=position_ids,
                              past_kv=past_kv, use_cache=use_cache, is_bidir=is_bidir)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


class TextBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(cfg.num_hidden_layers):
            is_full = (cfg.layer_types[i] == "full_attention")
            head_dim = cfg.global_head_dim if is_full else cfg.head_dim
            num_kv   = cfg.num_global_key_value_heads if is_full else cfg.num_key_value_heads
            self.layers.append(TransformerBlock(cfg, i, head_dim, num_kv))
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids: Tensor, attn_mask=None, position_ids=None,
                inputs_embeds=None, is_bidir: bool = False) -> Tensor:
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for blk in self.layers:
            x, _ = blk(x, attn_mask=attn_mask, position_ids=position_ids, is_bidir=is_bidir)
        return self.norm(x)
