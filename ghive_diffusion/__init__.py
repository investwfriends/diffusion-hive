"""Hive-conditioned block-diffusion model.

This package provides the implementation behind
``adaptation_plan.md`` — a Mzinga-conditioned diffusion
policy/value model for Hive. The original DiffusionGemma-style
multimodal model has been removed (Phase 1). The Hive-specific
classes live in submodules:

- :mod:`.hive_config` — :class:`HiveDiffusionConfig` and tier configs.
- :mod:`.hive_model` — :class:`HiveDiffusionModel` with value + policy heads.
- :mod:`.tokenizer` — :class:`HiveTokenizer` (Phase 4).
- :mod:`.context_builder` — :class:`HiveContextBuilder` (Phase 5).
- :mod:`.legal_scorer` — :class:`HiveLegalScorer` (Phase 5).
- :mod:`.training` — multi-objective training step (Phase 6/7).
- :mod:`.inference` — fast play + MCTS-guided search (Phase 10).
- :mod:`.dataset` — self-play + MCTS-improved dataset pipeline (Phase 9).
- :mod:`.metrics` — evaluation metrics (Phase 11).
"""

from .config import (
    DiffusionGemmaTextConfig,
    DiffusionGemmaConfig,
    MiniConfig,
)
from .model import DiffusionGemmaForBlockDiffusion, GenerationOutput

from .hive_config import (
    HiveDiffusionConfig,
    HiveSmokeConfig,
    HiveTrainableConfig,
    HiveStrongConfig,
    make_smoke_config,
)
from .hive_model import (
    HiveDiffusionModel,
    SinusoidalTimestepEmbedding,
    build_smoke_model,
    build_trainable_model,
    build_strong_model,
)
from .moe import RouterInfo

__all__ = [
    # Backwards-compatible text model (vision removed)
    "DiffusionGemmaTextConfig",
    "DiffusionGemmaConfig",
    "MiniConfig",
    "DiffusionGemmaForBlockDiffusion",
    "GenerationOutput",
    # Hive-specific
    "HiveDiffusionConfig",
    "HiveSmokeConfig",
    "HiveTrainableConfig",
    "HiveStrongConfig",
    "make_smoke_config",
    "HiveDiffusionModel",
    "SinusoidalTimestepEmbedding",
    "build_smoke_model",
    "build_trainable_model",
    "build_strong_model",
    "RouterInfo",
]