"""ghive_diffusion_lite — MacBook-trainable Hive block-diffusion model.

No MoE.  Dense gated-FFN backbone.  ~268K params.  CPU/GPU training.
Mzinga AlphaZero adapter for strong training-data generation.

Preferred entry point
---------------------
::

    python -m ghive_diffusion_lite.pipeline \\
        --dataset ghive_diffusion_lite/dataset_merged.pt \\
        --device mps --out-dir runs/lite_run1

See ``pipeline.py`` for train → eval → self-play with best checkpoints.
"""

from .hive_lite_config import HiveLiteConfig
from .hive_lite_model import HiveLiteModel, DenseMLP, LiteTransformerBlock, LiteBackbone, build_lite_model
from .lite_trainer import LiteHiveTrainer, LiteTrainingSample
from .mzinga_adapter import MzingaMCTSAdapter
from .train_lite import train_lite, train_from_dataset

__all__ = [
    "HiveLiteConfig",
    "HiveLiteModel",
    "DenseMLP",
    "LiteTransformerBlock",
    "LiteBackbone",
    "build_lite_model",
    "LiteHiveTrainer",
    "LiteTrainingSample",
    "MzingaMCTSAdapter",
    "train_lite",
    "train_from_dataset",
]
