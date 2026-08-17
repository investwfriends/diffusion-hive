import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import GemmaRMSNorm
from .attention import _rope_freqs, _apply_rope

class Gemma4VisionAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.hd = cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        cos, sin = _rope_freqs(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos = self.rope_cos[:n].to(x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[:n].to(x.dtype).unsqueeze(0).unsqueeze(0)
        q, k = _apply_rope(q, k, cos, sin, 1.0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        return self.proj(out)


class Gemma4VisionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn  = Gemma4VisionAttention(cfg)
        self.norm2 = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Gemma4VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = nn.Conv2d(3, cfg.hidden_size,
                                     kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.layers = nn.ModuleList(Gemma4VisionBlock(cfg) for _ in range(cfg.num_hidden_layers))

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: (b, 3, H, W) with H,W divisible by patch_size
        x = self.patch_embed(pixel_values)               # (b, c, h, w)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)                 # (b, h*w, c)
        for blk in self.layers:
            x = blk(x)
        # pooling kernel 3 (Gemma 4 default)
        k = self.cfg.pooling_kernel_size
        if k > 1:
            b2, n, d = x.shape
            side = int(math.sqrt(n))
            x = x.view(b2, side, side, d).permute(0, 3, 1, 2)
            x = F.avg_pool2d(x, kernel_size=k, stride=k)
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
