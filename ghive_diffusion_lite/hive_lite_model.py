"""Lite Hive diffusion model — no MoE, dense gated-FFN backbone.

DenseMLP
    Simple gated GELU feed-forward (same arch as a single MoE Expert).

LiteTransformerBlock
    GemmaAttention + RMSNorm + DenseMLP.  Same interface as the
    original ``TransformerBlock``.

LiteBackbone
    Embedding + stack of ``LiteTransformerBlock``s + final RMSNorm.

HiveLiteModel
    Drop-in compatible with ``HiveTrainer`` from ``ghive_diffusion``.
    Provides ``begin_moe_capture`` / ``end_moe_capture`` stubs so the
    trainer can call them without errors (always returns empty list).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ghive_diffusion.attention import GemmaAttention
from ghive_diffusion.hive_model import GenerationOutput, SinusoidalTimestepEmbedding
from ghive_diffusion.utils import GemmaRMSNorm, _softcap

from .hive_lite_config import HiveLiteConfig


# ---------------------------------------------------------------------------
# Dense MLP (replaces MoELayer)
# ---------------------------------------------------------------------------


class DenseMLP(nn.Module):
    """Gated GELU FFN — same arch as a single ``Expert`` from ``moe.py``."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


# ---------------------------------------------------------------------------
# Lite Transformer Block
# ---------------------------------------------------------------------------


class LiteTransformerBlock(nn.Module):
    """Same interface as ``TransformerBlock``, but uses ``DenseMLP``."""

    def __init__(self, cfg: HiveLiteConfig, layer_idx: int,
                 head_dim: int, num_kv_heads: int):
        super().__init__()
        layer_type = cfg.layer_types[layer_idx]
        self.attn = GemmaAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=cfg.sliding_window,
            layer_type=layer_type,
            rope_theta=cfg.rope_theta_for(layer_type),
            rope_proportional=(layer_type == "full_attention"),
            partial_rotary_factor=cfg.partial_rotary_factor_for(layer_type),
            max_pos=cfg.max_position_embeddings,
            softcap=cfg.final_logit_softcapping,
        )
        self.attn_norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp_norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = DenseMLP(cfg.hidden_size, cfg.dense_intermediate_size)

    def forward(self, x: Tensor,
                attn_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                past_kv: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False,
                is_bidir: bool = False,
                cross_prefix_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        h = self.attn_norm(x)
        a, new_kv = self.attn(h, attn_mask=attn_mask,
                               position_ids=position_ids,
                               past_kv=past_kv, use_cache=use_cache,
                               is_bidir=is_bidir,
                               cross_prefix_mask=cross_prefix_mask)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Lite Backbone
# ---------------------------------------------------------------------------


class LiteBackbone(nn.Module):
    """Embedding + Transformer stack + final RMSNorm.  Same interface as
    ``TextBackbone``."""

    def __init__(self, cfg: HiveLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList()
        for i in range(cfg.num_hidden_layers):
            is_full = (cfg.layer_types[i] == "full_attention")
            head_dim = cfg.global_head_dim if is_full else cfg.head_dim
            num_kv = cfg.num_global_key_value_heads if is_full else cfg.num_key_value_heads
            self.layers.append(LiteTransformerBlock(cfg, i, head_dim, num_kv))
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids: Tensor,
                attn_mask=None, position_ids=None,
                inputs_embeds=None, is_bidir: bool = False,
                cross_prefix_mask: Optional[Tensor] = None) -> Tensor:
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for blk in self.layers:
            x, _ = blk(x, attn_mask=attn_mask, position_ids=position_ids,
                       is_bidir=is_bidir, cross_prefix_mask=cross_prefix_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# HiveLiteModel
# ---------------------------------------------------------------------------


class HiveLiteModel(nn.Module):
    """Block-diffusion model with dense FFN backbone.

    Drop-in compatible with ``HiveTrainer`` — same encoder/decoder,
    scoring, diffusion, value-head, aux-heads interface.  The only
    difference is the backbone uses ``DenseMLP`` instead of MoE.
    """

    def __init__(self, cfg: HiveLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.text = LiteBackbone(cfg)
        if not cfg.share_encoder_decoder:
            self.decoder = LiteBackbone(cfg)
        else:
            self.decoder = self.text

        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, 1),
            nn.Tanh(),
        )
        self.policy_score_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, 1),
        )

        self.aux_head_dims = [3, 5, 2, 2, 2, 2, 4, 3, 3]
        self.aux_heads = nn.ModuleList([
            nn.Linear(cfg.hidden_size, n) for n in self.aux_head_dims
        ])

        self.timestep_embed = SinusoidalTimestepEmbedding(cfg.hidden_size)
        self.self_cond_proj = nn.Linear(cfg.vocab_size, cfg.effective_self_cond_dim)
        self.lm_head = lambda h: F.linear(h, self.text.embed_tokens.weight)

    # ---------- MoE stubs (trainer compatibility) --------------------------

    def begin_moe_capture(self) -> None:
        pass

    def end_moe_capture(self) -> list:
        return []

    # ---------- Heads ------------------------------------------------------

    def _lm_head(self, h: Tensor) -> Tensor:
        logits = self.lm_head(h)
        if self.cfg.final_logit_softcapping:
            logits = _softcap(logits, self.cfg.final_logit_softcapping)
        return logits

    def encode_value(self, input_ids: Tensor,
                     attn_mask: Optional[Tensor] = None,
                     inputs_embeds: Optional[Tensor] = None) -> Tensor:
        h, _ = self.forward_encoder(input_ids, attn_mask=attn_mask,
                                    inputs_embeds=inputs_embeds, use_cache=False)
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=-1).long().clamp(min=1) - 1
            lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
            last = h.gather(1, lengths)
        else:
            last = h[:, -1:, :]
        return self.value_head(last).squeeze(-1).squeeze(-1)

    def forward_aux_heads(self, input_ids: Tensor,
                          attn_mask: Optional[Tensor] = None,
                          inputs_embeds: Optional[Tensor] = None) -> List[Tensor]:
        h, _ = self.forward_encoder(input_ids, attn_mask=attn_mask,
                                    inputs_embeds=inputs_embeds, use_cache=False)
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=-1).long().clamp(min=1) - 1
            lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
            last = h.gather(1, lengths)
        else:
            last = h[:, -1:, :]
        return [head(last).squeeze(1) for head in self.aux_heads]

    # ---------- Encoder ----------------------------------------------------

    def forward_encoder(self, input_ids: Tensor,
                        attn_mask: Optional[Tensor] = None,
                        inputs_embeds: Optional[Tensor] = None,
                        use_cache: bool = True):
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

    # ---------- Decoder ----------------------------------------------------

    def forward_decoder(self, decoder_input_ids: Tensor,
                        encoder_kv: List[Tuple[Tensor, Tensor]],
                        self_conditioning_embeds: Optional[Tensor] = None,
                        timestep: Optional[Tensor] = None,
                        decoder_position_ids: Optional[Tensor] = None,
                        canvas_attn_mask: Optional[Tensor] = None,
                        bypass_sliding_for_prefix: bool = True) -> Tensor:
        b, T = decoder_input_ids.shape
        T_enc = encoder_kv[0][0].size(2)

        h = self.decoder.embed_tokens(decoder_input_ids)
        if self_conditioning_embeds is not None:
            if self_conditioning_embeds.dim() == 3:
                sc = self.self_cond_proj(self_conditioning_embeds)
            else:
                sc = self_conditioning_embeds
            h = h + sc
        if timestep is not None:
            if timestep.dim() == 0:
                t_emb = self.timestep_embed(timestep.unsqueeze(0))
            elif timestep.dim() == 1 and timestep.size(0) == b:
                t_emb = self.timestep_embed(timestep)
            else:
                raise ValueError(
                    f"timestep must be scalar or (B,); got {timestep.shape}")
            h = h + t_emb.unsqueeze(1)

        if decoder_position_ids is None:
            decoder_position_ids = torch.arange(T, device=h.device).unsqueeze(0) + T_enc

        cross_prefix_mask = None
        if bypass_sliding_for_prefix and T_enc > 0:
            tk_total = T + T_enc
            cross_prefix_mask = torch.zeros(tk_total, dtype=torch.bool, device=h.device)
            cross_prefix_mask[-T_enc:] = True

        for li, blk in enumerate(self.decoder.layers):
            cross_kv = encoder_kv[li]
            tk = T + cross_kv[0].size(2)
            attn_bias = torch.zeros(T, tk, device=h.device, dtype=h.dtype)
            if canvas_attn_mask is not None:
                attn_bias[:, -T:] = attn_bias[:, -T:] + canvas_attn_mask
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            h, _ = blk(h, attn_mask=attn_bias, position_ids=decoder_position_ids,
                       past_kv=cross_kv, use_cache=False, is_bidir=True,
                       cross_prefix_mask=cross_prefix_mask)
        h = self.decoder.norm(h)
        return h

    # ---------- Legal-move scoring -----------------------------------------

    def score_legal_moves(self, context_ids: Tensor,
                          legal_move_ids_list: List[List[int]],
                          attn_mask: Optional[Tensor] = None,
                          self_conditioning_embeds: Optional[Tensor] = None,
                          timestep: Optional[Tensor] = None,
                          use_value_head: bool = False,
                          move_chunk_size: int = 0,
                          use_confidence_weight: bool = False,
                          self_cond_passes: int = 1,
                          encoder_kv: Optional[List[Tuple[Tensor, Tensor]]] = None
                          ) -> Tuple[Tensor, Optional[Tensor]]:
        device = context_ids.device
        b = context_ids.size(0)
        if encoder_kv is None:
            _, encoder_kv = self.forward_encoder(context_ids, attn_mask=attn_mask, use_cache=True)

        n_moves = len(legal_move_ids_list)
        if n_moves == 0:
            scores = torch.zeros(b, 0, device=device)
            value = self.encode_value(context_ids, attn_mask=attn_mask) if use_value_head else None
            return scores, value

        chunk = move_chunk_size if move_chunk_size and move_chunk_size > 0 else n_moves
        chunk = min(chunk, n_moves)

        all_scores: List[Tensor] = []

        for start in range(0, n_moves, chunk):
            end = min(start + chunk, n_moves)
            chunk_ids = legal_move_ids_list[start:end]
            chunk_n = len(chunk_ids)

            max_len = max(len(m) for m in chunk_ids)
            padded = torch.full((b * chunk_n, max_len), 0, dtype=torch.long, device=device)
            for j, mv in enumerate(chunk_ids):
                for k in range(b):
                    padded[k * chunk_n + j, :len(mv)] = torch.tensor(mv, dtype=torch.long, device=device)

            enc_kv_batch: List[Tuple[Tensor, Tensor]] = []
            for k_layer, v_layer in encoder_kv:
                ek = k_layer.repeat_interleave(chunk_n, dim=0)
                ev = v_layer.repeat_interleave(chunk_n, dim=0)
                enc_kv_batch.append((ek, ev))

            sc = None
            if self_conditioning_embeds is not None:
                sc = self_conditioning_embeds.repeat_interleave(chunk_n, dim=0)
            ts = None
            if timestep is not None:
                ts = timestep.repeat_interleave(chunk_n, dim=0)

            h = self.forward_decoder(padded, enc_kv_batch,
                                     self_conditioning_embeds=sc,
                                     timestep=ts)
            for _pass in range(1, self_cond_passes):
                logits_prev = self._lm_head(h).detach()
                sc_next = F.softmax(logits_prev, dim=-1)
                h = self.forward_decoder(padded, enc_kv_batch,
                                         self_conditioning_embeds=sc_next,
                                         timestep=ts)
            last = h[:, -1, :]
            scores_flat = self.policy_score_head(last).squeeze(-1)
            if use_confidence_weight:
                logits_all = self._lm_head(h)
                probs = F.softmax(logits_all, dim=-1)
                ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean(-1)
                scores_flat = scores_flat + 0.1 * (1.0 - ent)
            all_scores.append(scores_flat.view(b, chunk_n))

        scores = torch.cat(all_scores, dim=1)

        value = None
        if use_value_head:
            value = self.encode_value(context_ids, attn_mask=attn_mask)
        return scores, value

    # ---------- Diffusion noise --------------------------------------------

    @staticmethod
    def add_diffusion_noise(input_ids: Tensor, mask_token_id: int,
                            timesteps: Tensor, mask_prob: Tensor,
                            vocab_size: int,
                            rng: Optional[torch.Generator] = None,
                            schedule: str = "linear") -> Tensor:
        b, T = input_ids.shape
        device = input_ids.device

        if schedule == "cosine":
            s = 0.008
            t = timesteps.float()
            alpha_bar = torch.cos(
                ((t + s) / (1.0 + s)) * (math.pi / 2.0)
            ) ** 2
            effective = 1.0 - alpha_bar
            p = effective.view(-1, 1).expand(b, T)
        else:
            p = mask_prob.view(b, 1).expand(b, T)

        rand = torch.rand((b, T), device=device, generator=rng)
        do_mask = rand < p
        replace_mask = do_mask & (torch.rand((b, T), device=device, generator=rng) < 0.1)
        random_id = torch.randint(0, vocab_size, (b, T), device=device, generator=rng)
        out = torch.where(replace_mask, random_id, input_ids)
        out = torch.where(do_mask & ~replace_mask, torch.full_like(out, mask_token_id), out)
        return out

    # ---------- Generation (unused in lite training, kept for completeness) --

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
        canvas_len = self.cfg.canvas_length
        num_canvases = (max_new_tokens + canvas_len - 1) // canvas_len
        eos_id = self.cfg.eos_token_id
        history: List[Tensor] = [input_ids]
        generated = list(input_ids.unbind(0))

        for ci in range(num_canvases):
            full_input = torch.cat(history, dim=1)
            _, encoder_kv = self.forward_encoder(full_input, use_cache=True)

            canvas = torch.randint(0, self.cfg.vocab_size,
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
                new_canvas = self._renoise(new_canvas, accepted, self.cfg.vocab_size)

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


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_lite_model(**overrides) -> HiveLiteModel:
    cfg = HiveLiteConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return HiveLiteModel(cfg)
