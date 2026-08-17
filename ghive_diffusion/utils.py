import torch
import torch.nn as nn
from torch import Tensor

def _softcap(x: Tensor, cap: float) -> Tensor:
    """logit soft-capping: x ← cap * tanh(x / cap)."""
    return cap * torch.tanh(x / cap)

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))   # initialised to 0 → scale=1

    def forward(self, x: Tensor) -> Tensor:
        # compute in float32 for stability
        in_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return ((1.0 + self.weight) * x32.to(in_dtype))
