"""Pure-text block-diffusion model for Hive.

This is the Phase-1 / Phase-3 deliverable: a vision-free
``HiveDiffusionModel`` that reuses :class:`TextBackbone` as both the
encoder and the decoder.

Key additions over the original :class:`DiffusionGemmaForBlockDiffusion`:

- ``value_head`` — Tanh scalar value from the encoder's last hidden state.
- ``policy_score_head`` — per-legal-move scoring (no prefix trie masking;
  caller supplies the legal-move candidates from Mzinga).
- ``timestep_embedding`` — sinusoidal embedding of the diffusion noise level.
- ``forward_decoder`` accepts ``cross_prefix_mask`` so sliding-window
  layers can still attend to the entire encoded prefix (Phase 3 fix).
- ``score_legal_moves`` — convenience scorer that runs the encoder once
  and reuses the KV cache for every legal move (Phase 5).

The vision path (``Gemma4VisionTower``, ``MultiModalProjector``,
``encode_images``) has been removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .backbone import TextBackbone
from .hive_config import HiveDiffusionConfig, HiveSmokeConfig, HiveTrainableConfig, HiveStrongConfig
from .moe import RouterInfo
from .utils import _softcap


@dataclass
class GenerationOutput:
    sequences: Tensor
    tokens_per_forward: Optional[Tensor] = None


class SinusoidalTimestepEmbedding(nn.Module):
    """Standard sinusoidal noise-level embedding (Diffusion-style)."""

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        # timesteps: (B,) float in [0, 1] or arbitrary scalar
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class HiveDiffusionModel(nn.Module):
    """Block-diffusion policy/value model conditioned on Hive state.

    The encoder consumes a canonical context (game type, board state,
    history, board features, legal moves) causally. The decoder runs a
    bidirectional block-diffusion canvas over candidate moves, PVs, or
    value-bucketed tokens, with full cross-attention to the encoded
    prefix.

    Final move selection is *not* the diffusion output — it is the
    ``policy_score_head`` applied to every legal move (from Mzinga) and
    ranked.
    """

    def __init__(self, cfg: HiveDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.text = TextBackbone(cfg)
        if not cfg.share_encoder_decoder:
            self.decoder = TextBackbone(cfg)
        else:
            self.decoder = self.text

        # Heads
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.GELU(),
            nn.Linear(cfg.hidden_size, 1),
            nn.Tanh(),
        )
        self.policy_score_head = nn.Linear(cfg.hidden_size, 1)

        # Auxiliary heads (NEXT_STEPS 1.2).
        # 9 classification heads on the encoder's last hidden state:
        #   0  game_phase_bucket     — 3 classes (open / midgame / endgame)
        #   1  legal_move_count_bucket — 5 classes (0-5, 6-10, 11-20, 21-40, 41+)
        #   2  queen_in_play          — 2 classes (yes / no)
        #   3  queen_placement_required — 2 classes (turn 4 imminent / not)
        #   4  noisy_move_available   — 2 classes (yes / no)
        #   5  pass_legal             — 2 classes (yes / no)
        #   6  queen_surround_count    — 4 classes (0, 1, 2, 3+)
        #   7  pinned_piece_count     — 3 classes (0, 1, 2+)
        #   8  mobility_advantage      — 3 classes (negative / neutral / positive)
        self.aux_head_dims = [3, 5, 2, 2, 2, 2, 4, 3, 3]
        self.aux_heads = nn.ModuleList([
            nn.Linear(cfg.hidden_size, n) for n in self.aux_head_dims
        ])

        # Diffusion timestep conditioning: an embedding added to every
        # canvas position before the decoder stack.
        self.timestep_embed = SinusoidalTimestepEmbedding(cfg.hidden_size)

        # Optional self-conditioning projection: the previous logits are
        # mapped back into the decoder's hidden-size space.
        self.self_cond_proj = nn.Linear(cfg.vocab_size, cfg.effective_self_cond_dim)

        # Tied output projection
        self.lm_head = lambda h: F.linear(h, self.text.embed_tokens.weight)

        # MoE router-statistics capture (Phase 6.5).
        # The trainer calls begin_moe_capture() before gradient-bearing
        # forward passes and end_moe_capture() afterwards to retrieve the
        # accumulated RouterInfo records for the load-balance loss.
        self._moe_capturing: bool = False
        self._moe_records: List[RouterInfo] = []

    # ---------- MoE router-statistics capture (Phase 6.5) -------------------

    def begin_moe_capture(self) -> None:
        """Start collecting router statistics from gradient-bearing forwards."""
        self._moe_capturing = True
        self._moe_records = []

    def end_moe_capture(self) -> List[RouterInfo]:
        """Stop collecting and return the accumulated router records."""
        self._moe_capturing = False
        records = self._moe_records
        self._moe_records = []
        return records

    def _collect_moe(self, backbone: TextBackbone, stack: str) -> None:
        """Collect router info from every MoELayer in *backbone*.

        Only called when capturing is active and autograd is enabled,
        so no-grad passes (self-conditioning, diffusion encoder) are
        excluded automatically.
        """
        if not self._moe_capturing:
            return
        if not torch.is_grad_enabled():
            return
        for i, blk in enumerate(backbone.layers):
            stats = blk.mlp.last_router_info
            if stats is not None:
                self._moe_records.append(RouterInfo(
                    top_indices=stats.top_indices,
                    top_weights=stats.top_weights,
                    all_scores=stats.all_scores,
                    layer_idx=i,
                    stack=stack,
                ))

    # ---------- Heads -------------------------------------------------------

    def _lm_head(self, h: Tensor) -> Tensor:
        logits = self.lm_head(h)
        if self.cfg.final_logit_softcapping:
            logits = _softcap(logits, self.cfg.final_logit_softcapping)
        return logits

    def encode_value(self, input_ids: Tensor, attn_mask: Optional[Tensor] = None,
                     inputs_embeds: Optional[Tensor] = None) -> Tensor:
        """Return the scalar value from the encoder's last position."""
        h, _ = self.forward_encoder(input_ids, attn_mask=attn_mask,
                                    inputs_embeds=inputs_embeds, use_cache=False)
        # Use the last non-pad position (or just the last token if no mask).
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=-1).long().clamp(min=1) - 1
            lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
            last = h.gather(1, lengths)
        else:
            last = h[:, -1:, :]
        return self.value_head(last).squeeze(-1).squeeze(-1)

    def forward_aux_heads(self, input_ids: Tensor,
                          attn_mask: Optional[Tensor] = None,
                          inputs_embeds: Optional[Tensor] = None
                          ) -> List[Tensor]:
        """Return logits from all 9 auxiliary heads on the encoder's last position.

        Returns a list of ``(B, n_classes)`` tensors, one per head.
        """
        h, _ = self.forward_encoder(input_ids, attn_mask=attn_mask,
                                    inputs_embeds=inputs_embeds, use_cache=False)
        if attn_mask is not None:
            lengths = attn_mask.sum(dim=-1).long().clamp(min=1) - 1
            lengths = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
            last = h.gather(1, lengths)
        else:
            last = h[:, -1:, :]
        return [head(last).squeeze(1) for head in self.aux_heads]

    # ---------- Encoder -----------------------------------------------------

    def forward_encoder(self, input_ids: Tensor, attn_mask: Optional[Tensor] = None,
                        inputs_embeds: Optional[Tensor] = None,
                        use_cache: bool = True):
        """Causal prefill over the context tokens. Returns hidden states + KV cache."""
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
        self._collect_moe(self.text, "encoder")
        return h, kv

    # ---------- Decoder -----------------------------------------------------

    def forward_decoder(self, decoder_input_ids: Tensor,
                        encoder_kv: List[Tuple[Tensor, Tensor]],
                        self_conditioning_embeds: Optional[Tensor] = None,
                        timestep: Optional[Tensor] = None,
                        decoder_position_ids: Optional[Tensor] = None,
                        canvas_attn_mask: Optional[Tensor] = None,
                        bypass_sliding_for_prefix: bool = True) -> Tensor:
        """Bidirectional decoder with full cross-attention to encoded prefix.

        ``bypass_sliding_for_prefix`` enables the Phase-3 fix: every key
        position belonging to the encoded prefix is exempted from the
        sliding-window rule, so the decoder can see the full context
        even when the prefix length exceeds ``sliding_window``.
        """
        b, T = decoder_input_ids.shape
        T_enc = encoder_kv[0][0].size(2)

        h = self.decoder.embed_tokens(decoder_input_ids)
        if self_conditioning_embeds is not None:
            # Project previous-step logits back to hidden size if shapes differ
            if self_conditioning_embeds.dim() == 3:
                sc = self.self_cond_proj(self_conditioning_embeds)
            else:
                sc = self_conditioning_embeds
            h = h + sc
        if timestep is not None:
            # Broadcast scalar/vector timestep to every canvas position
            if timestep.dim() == 0:
                t_emb = self.timestep_embed(timestep.unsqueeze(0))
            elif timestep.dim() == 1 and timestep.size(0) == b:
                t_emb = self.timestep_embed(timestep)
            else:
                raise ValueError(f"timestep must be scalar or (B,); got {timestep.shape}")
            h = h + t_emb.unsqueeze(1)

        if decoder_position_ids is None:
            decoder_position_ids = torch.arange(T, device=h.device).unsqueeze(0) + T_enc

        # Cross-prefix mask: shape (tk,) where tk = canvas_len + T_enc.
        # Positions corresponding to the encoded prefix (the LAST T_enc
        # entries of the full key sequence) are marked True so the
        # sliding-window rule does not prune them. The attention module
        # will OR this against its (t, tk) sliding-window mask.
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

            # ``cross_prefix_mask`` is per-layer to match the per-layer
            # attention invocation. We re-use the same boolean mask (over
            # the LAST ``T_enc`` positions of the full key sequence) for
            # every layer.
            h, _ = blk(h, attn_mask=attn_bias, position_ids=decoder_position_ids,
                       past_kv=cross_kv, use_cache=False, is_bidir=True,
                       cross_prefix_mask=cross_prefix_mask)
        h = self.decoder.norm(h)
        self._collect_moe(self.decoder, "decoder")
        return h

    # ---------- Legal-move scoring (Phase 5) --------------------------------

    def encode_legal_move(self, move_str: str, tokenizer) -> Tensor:
        """Tokenize a single legal move as a tiny canvas. Returns (1, L)."""
        ids = tokenizer.encode_move(move_str)
        return torch.tensor([ids], dtype=torch.long)

    def score_legal_moves(self, context_ids: Tensor,
                          legal_move_ids_list: List[List[int]],
                          attn_mask: Optional[Tensor] = None,
                          self_conditioning_embeds: Optional[Tensor] = None,
                          timestep: Optional[Tensor] = None,
                          use_value_head: bool = False,
                          move_chunk_size: int = 0,
                          use_confidence_weight: bool = False,
                          self_cond_passes: int = 1) -> Tuple[Tensor, Optional[Tensor]]:
        """Score every legal move given the encoded context.

        ``legal_move_ids_list`` is a list of tokenized legal moves (each a
        1-D list of ints). Padding to the longest length is done
        internally. Returns ``(scores, value)``:

        - ``scores`` is shape ``(B, n_moves)``, the scalar score from
          :attr:`policy_score_head` applied to the last canvas position.
        - ``value`` is shape ``(B,)``, the value head estimate of the
          encoded context. ``None`` if ``use_value_head=False``.

        ``move_chunk_size`` controls memory usage: if > 0, legal moves are
        scored in chunks of that size rather than all at once. This is
        important for mid-game positions with 50+ legal moves.  0 means
        no chunking (original behaviour).

        ``use_confidence_weight`` adds a per-move entropy-based confidence
        bonus (low-entropy decoded canvases get a slight boost).

        ``self_cond_passes`` > 1 enables iterative self-conditioned denoising
        per move canvas — each pass feeds the previous softmax logits as
        conditioning, refining the hidden representation before scoring.
        """
        device = context_ids.device
        b = context_ids.size(0)
        # Encode the context once and cache K/V.
        _, encoder_kv = self.forward_encoder(context_ids, attn_mask=attn_mask, use_cache=True)

        n_moves = len(legal_move_ids_list)
        if n_moves == 0:
            scores = torch.zeros(b, 0, device=device)
            value = self.encode_value(context_ids, attn_mask=attn_mask) if use_value_head else None
            return scores, value

        # Determine chunk size.
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
            # ── multi-pass self-conditioned refinement ──
            for _pass in range(1, self_cond_passes):
                logits_prev = self._lm_head(h).detach()
                sc_next = F.softmax(logits_prev, dim=-1)
                h = self.forward_decoder(padded, enc_kv_batch,
                                         self_conditioning_embeds=sc_next,
                                         timestep=ts)
            last = h[:, -1, :]  # (B*chunk_n, hidden)
            scores_flat = self.policy_score_head(last).squeeze(-1)  # (B*chunk_n,)
            # ── diffusion-confidence weighting ──
            if use_confidence_weight:
                logits_all = self._lm_head(h)  # (B*chunk_n, T, V)
                probs = F.softmax(logits_all, dim=-1)
                ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean(-1)  # per token, avg over T
                scores_flat = scores_flat + 0.1 * (1.0 - ent)
            all_scores.append(scores_flat.view(b, chunk_n))

        scores = torch.cat(all_scores, dim=1)  # (B, n_moves)

        value = None
        if use_value_head:
            value = self.encode_value(context_ids, attn_mask=attn_mask)
        return scores, value

    # ---------- Diffusion sampling (used by training and inference) ----------

    @staticmethod
    def add_diffusion_noise(input_ids: Tensor, mask_token_id: int,
                            timesteps: Tensor, mask_prob: Tensor,
                            vocab_size: int,
                            rng: Optional[torch.Generator] = None,
                            schedule: str = "linear") -> Tensor:
        """Mask / random-replace canvas tokens according to per-sample ``timesteps``.

        For ``timestep = 1.0`` every canvas token is masked. For ``0.0``
        no token is masked.

        ``schedule`` controls the mapping from ``timesteps`` (in [0, 1])
        to the effective mask probability:

        - ``"linear"`` — ``mask_prob = t`` (the default; matches the
          original implementation).
        - ``"cosine"`` — Nichol & Dhariwal cosine schedule:
          ``mask_prob = cos((t + s) / (1 + s) * pi/2)^2`` with
          ``s = 0.008``.  This biases training toward harder timesteps,
          generally improving sample quality.
        """
        b, T = input_ids.shape
        device = input_ids.device

        # Apply the noise schedule to get per-sample mask probability.
        if schedule == "cosine":
            s = 0.008
            t = timesteps.float()
            alpha_bar = torch.cos(
                ((t + s) / (1.0 + s)) * (math.pi / 2.0)
            ) ** 2
            # mask_prob = 1 - alpha_bar  (alpha_bar ≈ 1 at t=0, ≈ 0 at t=1)
            effective = 1.0 - alpha_bar
            # Broadcast to (b, T): effective may be (1,) or (b,)
            p = effective.view(-1, 1).expand(b, T)
        else:
            # Use mask_prob directly (linear schedule, original behaviour).
            p = mask_prob.view(b, 1).expand(b, T)

        rand = torch.rand((b, T), device=device, generator=rng)
        do_mask = rand < p
        # Choose randomly between mask and random replacement with prob 0.1
        # of mask (matches BERT/UL2-style corruption).
        replace_mask = do_mask & (torch.rand((b, T), device=device, generator=rng) < 0.1)
        random_id = torch.randint(0, vocab_size, (b, T), device=device, generator=rng)
        out = torch.where(replace_mask, random_id, input_ids)
        out = torch.where(do_mask & ~replace_mask, torch.full_like(out, mask_token_id), out)
        return out

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


def build_smoke_model(**overrides) -> HiveDiffusionModel:
    """Convenience builder for a smoke-test :class:`HiveDiffusionModel`."""
    cfg = HiveSmokeConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return HiveDiffusionModel(cfg)


def build_trainable_model(**overrides) -> HiveDiffusionModel:
    """Convenience builder for a trainable :class:`HiveDiffusionModel`."""
    cfg = HiveTrainableConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return HiveDiffusionModel(cfg)


def build_strong_model(**overrides) -> HiveDiffusionModel:
    """Convenience builder for a strong :class:`HiveDiffusionModel`."""
    cfg = HiveStrongConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return HiveDiffusionModel(cfg)