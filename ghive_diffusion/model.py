"""Pure-text block-diffusion model (vision path removed).

This file preserves the original ``DiffusionGemmaForBlockDiffusion``
class name for backward compatibility with ``sanity_check.py``, but the
vision tower and multimodal projector have been removed. The class is
now text-only and uses the standard ``TextBackbone``.

For the Hive-specific model with value and policy heads, see
:mod:`ghive_diffusion.hive_model`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, List, Tuple

from .backbone import TextBackbone
from .utils import _softcap


@dataclass
class GenerationOutput:
    sequences: Tensor
    tokens_per_forward: Optional[Tensor] = None


class DiffusionGemmaForBlockDiffusion(nn.Module):
    """Pure-text block-diffusion model.

    The vision tower and multimodal projector have been removed (Phase 1
    of the adaptation plan). For Hive-specific functionality, see
    :class:`ghive_diffusion.hive_model.HiveDiffusionModel`.

    The encoder pre-fills the prompt and caches K/V; the decoder runs the
    canvas through the same backbone (with bidirectional self-attention
    and cross-attention to the encoder cache), and the
    EntropyBoundSampler iteratively refines the canvas.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # accept either a DiffusionGemmaConfig or a flat MiniConfig
        text_cfg = cfg.text if hasattr(cfg, "text") else cfg
        self.text_cfg = text_cfg
        self.text = TextBackbone(text_cfg)
        # Tied output projection
        self.lm_head = lambda h: F.linear(h, self.text.embed_tokens.weight)
        # Re-use the encoder as the decoder.
        self.decoder = self.text

    def _lm_head(self, h: Tensor) -> Tensor:
        logits = self.lm_head(h)
        return (_softcap(logits, self.text_cfg.final_logit_softcapping)
                if self.text_cfg.final_logit_softcapping else logits)

    def forward_encoder(self, input_ids: Tensor, attn_mask=None,
                        inputs_embeds=None, use_cache=True):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        h = inputs_embeds if inputs_embeds is not None else self.text.embed_tokens(input_ids)
        kv = []
        for blk in self.text.layers:
            attn_bias = torch.zeros(t, t, device=h.device, dtype=h.dtype)
            mask = torch.ones(t, t, dtype=torch.bool, device=h.device).tril()
            attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
            if attn_mask is not None:
                attn_bias = attn_bias + attn_mask[:, :t][:, :t]
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            h, new_kv = blk(h, attn_mask=attn_bias, position_ids=pos, use_cache=use_cache)
            kv.append(new_kv)
        h = self.text.norm(h)
        return h, kv

    def forward_decoder(self, decoder_input_ids: Tensor, encoder_kv: list,
                        self_conditioning_embeds: Optional[Tensor] = None,
                        decoder_position_ids: Optional[Tensor] = None,
                        canvas_attn_mask: Optional[Tensor] = None) -> Tensor:
        b, T = decoder_input_ids.shape
        T_enc = encoder_kv[0][0].size(2)

        h = self.text.embed_tokens(decoder_input_ids)
        if self_conditioning_embeds is not None:
            h = h + self_conditioning_embeds
        if decoder_position_ids is None:
            decoder_position_ids = torch.arange(T, device=h.device).unsqueeze(0) + T_enc

        for li, blk in enumerate(self.text.layers):
            cross_kv = encoder_kv[li]
            tk = T + cross_kv[0].size(2)
            attn_bias = torch.zeros(T, tk, device=h.device, dtype=h.dtype)
            if canvas_attn_mask is not None:
                attn_bias[:, -T:] = attn_bias[:, -T:] + canvas_attn_mask
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            h, _ = blk(h, attn_mask=attn_bias, position_ids=decoder_position_ids,
                       past_kv=cross_kv, use_cache=False, is_bidir=True)
        return self.text.norm(h)

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 256,
                 max_denoising_steps: int = 48,
                 t_min: float = 0.4, t_max: float = 0.8,
                 entropy_bound: float = 0.1,
                 confidence_threshold: float = 0.005,
                 stability_threshold: int = 2,
                 use_cache: bool = True) -> GenerationOutput:
        device = input_ids.device
        bsz = input_ids.size(0)
        canvas_len = self.text_cfg.canvas_length
        num_canvases = (max_new_tokens + canvas_len - 1) // canvas_len
        eos_id = self.text_cfg.eos_token_id
        history: List[Tensor] = [input_ids]
        generated = list(input_ids.unbind(0))

        for ci in range(num_canvases):
            full_input = torch.cat(history, dim=1)
            _, encoder_kv = self.forward_encoder(full_input, use_cache=True)

            canvas = torch.randint(0, self.text_cfg.vocab_size,
                                   (bsz, canvas_len), device=device)
            self_cond: Optional[Tensor] = None
            accepted_prev: Optional[Tensor] = None
            stable_count = 0

            for step in range(max_denoising_steps):
                sc_emb = None
                if self_cond is not None:
                    sc_emb = self_cond @ self.text.embed_tokens.weight

                h = self.forward_decoder(canvas, encoder_kv, sc_emb)
                logits = self._lm_head(h)

                frac = step / max(1, max_denoising_steps - 1)
                temperature = t_min + (t_max - t_min) * frac
                probs = (logits / temperature).softmax(-1)

                cand = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(bsz, canvas_len)

                ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1)
                accepted = self._accept_by_entropy_bound(ent, entropy_bound)
                new_canvas = torch.where(accepted, cand, canvas)
                new_canvas = self._renoise(new_canvas, accepted, self.text_cfg.vocab_size)

                if accepted_prev is not None and (new_canvas == accepted_prev).all():
                    stable_count += 1
                else:
                    stable_count = 0
                if ent.mean() < confidence_threshold and stable_count >= stability_threshold:
                    canvas = new_canvas
                    break
                accepted_prev = new_canvas
                canvas = new_canvas
                self_cond = logits.detach()

            history.append(canvas)
            generated[0] = torch.cat([generated[0], canvas[0]], dim=0)
            if eos_id in canvas[0].tolist():
                break

        sequences = torch.stack(generated, dim=0)
        return GenerationOutput(sequences=sequences)

    @staticmethod
    def _accept_by_entropy_bound(entropy: Tensor, bound: float) -> Tensor:
        b, t = entropy.shape
        sorted_ent, sorted_idx = entropy.sort(dim=-1)
        max_ent = sorted_ent[:, -1:]
        increments = (max_ent - sorted_ent).clamp_min(0)
        cum = increments.cumsum(-1)
        accept_count = (cum <= bound).sum(-1).clamp(min=1)
        accept_mask = torch.zeros_like(entropy, dtype=torch.bool)
        arange = torch.arange(t, device=entropy.device).unsqueeze(0).expand(b, -1)
        accept_mask.scatter_(-1, sorted_idx, arange < accept_count.unsqueeze(-1))
        return accept_mask

    @staticmethod
    def _renoise(canvas: Tensor, accept_mask: Tensor, vocab_size: int) -> Tensor:
        noise = torch.randint(0, vocab_size, canvas.shape,
                              device=canvas.device, dtype=canvas.dtype)
        return torch.where(accept_mask, canvas, noise)