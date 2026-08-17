import torch.nn as nn
from torch import Tensor

class MultiModalProjector(nn.Module):
    def __init__(self, v_dim: int, t_dim: int, out_tokens: int = 280):
        super().__init__()
        # simple linear projection + per-patch MLP (Gemma 4 uses a deeper projector)
        self.proj = nn.Linear(v_dim, t_dim, bias=False)
        self.out_tokens = out_tokens

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)
