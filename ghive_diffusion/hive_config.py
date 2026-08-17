"""Hive-compatible model configurations.

These are pure-text configurations tailored for Hive. They preserve every
field required by ``TextBackbone`` (``global_head_dim``, ``num_global_key_value_heads``,
``layer_types``, ``rope_theta_for``, ``partial_rotary_factor_for``) so the
same backbone can be reused without refactoring.

Three tiers are provided:

- :class:`HiveSmokeConfig` — for unit tests and CPU sanity checks.
- :class:`HiveTrainableConfig` — for single-GPU local training.
- :class:`HiveStrongConfig` — for serious self-play / search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class HiveDiffusionConfig:
    """Base Hive-compatible text configuration.

    The defaults match :class:`HiveSmokeConfig` so the dataclass can be
    instantiated directly for tests.
    """

    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 128          # dense MLP intermediate (unused in MoE layers)
    moe_intermediate_size: int = 64
    num_experts: int = 8
    top_k_experts: int = 2
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    num_global_key_value_heads: int = 2
    global_head_dim: int = 32
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 2048
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
    canvas_length: int = 64
    use_bidirectional_attention: str = "always"

    # Layer schedule: 3 sliding + 1 full, repeating. Full/global layers are
    # critical for late-game Hive positions because early moves, queen timing,
    # and expansion-piece constraints can remain strategically relevant.
    layer_types: List[str] = field(
        default_factory=lambda: ["sliding_attention"] * 3 + ["full_attention"]
    )

    # Encoder / decoder sharing. The encoder is used causally (prefill) and
    # the decoder is used bidirectionally over the canvas with cross-attention
    # to the encoded prefix. Sharing the weights keeps memory low; set
    # ``share_encoder_decoder=False`` to instantiate a separate decoder stack.
    share_encoder_decoder: bool = True

    # Self-conditioning embedding dimension. Mirrors ``hidden_size``.
    self_cond_dim: int = 0  # 0 -> same as hidden_size

    def rope_theta_for(self, layer_type: str) -> float:
        return 1_000_000.0 if layer_type == "full_attention" else 10_000.0

    def partial_rotary_factor_for(self, layer_type: str) -> float:
        return 0.25 if layer_type == "full_attention" else 1.0

    @property
    def effective_self_cond_dim(self) -> int:
        return self.self_cond_dim or self.hidden_size


@dataclass
class HiveSmokeConfig(HiveDiffusionConfig):
    """Tiny config for unit tests and CPU smoke checks."""

    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    num_global_key_value_heads: int = 2
    head_dim: int = 32
    global_head_dim: int = 32
    moe_intermediate_size: int = 64
    num_experts: int = 8
    top_k_experts: int = 2
    max_position_embeddings: int = 2048
    sliding_window: int = 128
    canvas_length: int = 64
    layer_types: List[str] = field(
        default_factory=lambda: ["sliding_attention"] * 3 + ["full_attention"]
    )


@dataclass
class HiveTrainableConfig(HiveDiffusionConfig):
    """Single-GPU local training config (~M-head range, tractable on 12GB)."""

    hidden_size: int = 384
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    num_global_key_value_heads: int = 2
    head_dim: int = 48
    global_head_dim: int = 64
    moe_intermediate_size: int = 192
    num_experts: int = 16
    top_k_experts: int = 2
    max_position_embeddings: int = 4096
    sliding_window: int = 256
    canvas_length: int = 128
    layer_types: List[str] = field(
        default_factory=lambda: ["sliding_attention"] * 3 + ["full_attention"] * 1
    )

    def __post_init__(self):
        # The repeating pattern (3 sliding + 1 full) must produce the right
        # number of layer_types entries. If the caller overrides ``num_hidden_layers``
        # we re-expand automatically.
        if len(self.layer_types) != self.num_hidden_layers:
            n_blocks = self.num_hidden_layers // 4
            self.layer_types = (
                ["sliding_attention"] * 3 + ["full_attention"]
            ) * n_blocks
            if len(self.layer_types) < self.num_hidden_layers:
                self.layer_types += ["sliding_attention"] * (
                    self.num_hidden_layers - len(self.layer_types)
                )


@dataclass
class HiveStrongConfig(HiveDiffusionConfig):
    """Stronger config for serious self-play / search."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    num_global_key_value_heads: int = 4
    head_dim: int = 64
    global_head_dim: int = 64
    moe_intermediate_size: int = 384
    num_experts: int = 32
    top_k_experts: int = 4
    max_position_embeddings: int = 8192
    sliding_window: int = 512
    canvas_length: int = 256
    layer_types: List[str] = field(
        default_factory=lambda: ["sliding_attention"] * 3 + ["full_attention"] * 1
    )

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            n_blocks = self.num_hidden_layers // 4
            self.layer_types = (
                ["sliding_attention"] * 3 + ["full_attention"]
            ) * n_blocks
            if len(self.layer_types) < self.num_hidden_layers:
                self.layer_types += ["sliding_attention"] * (
                    self.num_hidden_layers - len(self.layer_types)
                )


def make_smoke_config(**overrides) -> HiveDiffusionConfig:
    """Return a fresh :class:`HiveSmokeConfig` with optional overrides."""

    cfg = HiveSmokeConfig()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg