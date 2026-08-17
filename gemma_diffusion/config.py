from dataclasses import dataclass, field
from typing import List

@dataclass
class DiffusionGemmaTextConfig:
    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112      # dense MLP intermediate (mostly unused in MoE layers)
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    num_global_key_value_heads: int = 2
    global_head_dim: int = 512
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 262144
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    use_bidirectional_attention: str = "vision"
    canvas_length: int = 256

    # layer schedule: 5 sliding, 1 full, repeating (matches the published config)
    layer_types: List[str] = field(default_factory=lambda: (
        ["sliding_attention"] * 5 + ["full_attention"]) * 5
    )

    def rope_theta_for(self, layer_type: str) -> float:
        return 1_000_000.0 if layer_type == "full_attention" else 10_000.0

    def partial_rotary_factor_for(self, layer_type: str) -> float:
        return 0.25 if layer_type == "full_attention" else 1.0


@dataclass
class Gemma4VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    patch_size: int = 16
    pooling_kernel_size: int = 3
    max_position_embeddings: int = 131072
    rope_theta: float = 100.0
    rms_norm_eps: float = 1e-6


@dataclass
class DiffusionGemmaConfig:
    text: DiffusionGemmaTextConfig = field(default_factory=DiffusionGemmaTextConfig)
    vision: Gemma4VisionConfig = field(default_factory=Gemma4VisionConfig)
    canvas_length: int = 256
    image_token_id: int = 258880
    boi_token_id: int = 255999
    eoi_token_id: int = 258882
    tie_word_embeddings: bool = True

    # backward-compat shims for the dataclass-style access used above
    def __getattr__(self, name):
        if name in ("vocab_size", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "num_key_value_heads", "head_dim",
                    "num_global_key_value_heads", "global_head_dim",
                    "moe_intermediate_size", "num_experts", "top_k_experts",
                    "sliding_window", "rms_norm_eps", "final_logit_softcapping",
                    "max_position_embeddings", "layer_types", "pad_token_id",
                    "bos_token_id", "eos_token_id", "intermediate_size",
                    "use_bidirectional_attention", "canvas_length"):
            return getattr(self.text, name)
        raise AttributeError(name)


# A tiny config for smoke-testing the code on a laptop.
@dataclass
class MiniConfig:
    hidden_size: int = 128
    intermediate_size: int = 128
    moe_intermediate_size: int = 64
    num_experts: int = 8
    top_k_experts: int = 2
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    num_global_key_value_heads: int = 2
    global_head_dim: int = 32
    max_position_embeddings: int = 1024
    sliding_window: int = 64
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    vocab_size: int = 1024
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    layer_types: List[str] = field(default_factory=lambda:
        ["sliding_attention"] * 2 + ["full_attention"] * 2)
    canvas_length: int = 32

    def rope_theta_for(self, layer_type: str) -> float: return 1_000.0
    def partial_rotary_factor_for(self, layer_type: str) -> float:
        return 0.25 if layer_type == "full_attention" else 1.0
