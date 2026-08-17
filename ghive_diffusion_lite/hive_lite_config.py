"""Lite model config — no MoE, dense FFN, minimal layer count.

Target: ~200K params, trainable on Apple Silicon in <10 min.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class HiveLiteConfig:
    """Minimal block-diffusion config for MacBook training.

    Drops MoE entirely in favour of a dense gated-FFN per layer.
    Every field required by ``LiteTransformerBlock`` / ``LiteBackbone``
    is present; MoE-specific fields (``num_experts``, ``top_k_experts``,
    ``moe_intermediate_size``) are not defined.
    """

    vocab_size: int = 256
    hidden_size: int = 128
    dense_intermediate_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    num_global_key_value_heads: int = 2
    global_head_dim: int = 32
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 8192
    sliding_window: int = 128
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    canvas_length: int = 32
    use_bidirectional_attention: str = "always"

    layer_types: List[str] = field(
        default_factory=lambda: ["sliding_attention"] * 4 + ["full_attention"] * 2
    )

    share_encoder_decoder: bool = False
    self_cond_dim: int = 0

    def rope_theta_for(self, layer_type: str) -> float:
        return 1_000_000.0 if layer_type == "full_attention" else 10_000.0

    def partial_rotary_factor_for(self, layer_type: str) -> float:
        return 0.25 if layer_type == "full_attention" else 1.0

    @property
    def effective_self_cond_dim(self) -> int:
        return self.self_cond_dim or self.hidden_size
