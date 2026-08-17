# Migration Log: gemma_diffusion → ghive_diffusion

This document records every change made to the original
`gemma_diffusion/` package while implementing
[`adaptation_plan.md`](./adaptation_plan.md). It is the single source of
truth for "what was different before" — useful for code review, for
understanding why a module exists, and for re-deriving the architecture
later.

The transformation turned a generic text+vision block-diffusion
language model into a Mzinga-conditioned Hive policy/value model.

---

## At-a-glance

| File | Before | After |
|---|---|---|
| `vision.py` | present | **deleted** |
| `projector.py` | present | **deleted** |
| `config.py` | text + vision configs | text-only configs; `DiffusionGemmaConfig` retained for `gemma_diffusion` parity |
| `model.py` | text + vision with `encode_images` | text-only `DiffusionGemmaForBlockDiffusion` |
| `attention.py` | sliding + full, no cross-prefix bypass | sliding + full + **`cross_prefix_mask`** |
| `backbone.py` | text encoder | text encoder, threads `cross_prefix_mask` |
| `__init__.py` | minimal exports | exports new Hive classes + old text model |
| `hive_config.py` | — | **new** (Phase 2) |
| `hive_model.py` | — | **new** (Phase 1, 3) |
| `tokenizer.py` | — | **new** (Phase 4) |
| `context_builder.py` | — | **new** (Phase 5) |
| `legal_scorer.py` | — | **new** (Phase 5) |
| `training.py` | — | **new** (Phase 6, 7) |
| `canvas_formats.py` | — | **new** (Phase 8) |
| `dataset.py` | — | **new** (Phase 9) |
| `inference.py` | — | **new** (Phase 10; MCTS board.clone fix in NEXT_STEPS 2.5) |
| `metrics.py` | — | **new** (Phase 11; MoE router info in NEXT_STEPS 1.1) |
| `moe.py` | — | **new** (NEXT_STEPS 1.1: `MoERouterStats` + `RouterInfo`) |
| `train_loop.py` | — | **new** (NEXT_STEPS 2.1: production training loop) |
| `eval/runner.py` | — | **new** (NEXT_STEPS 3.4: game runner + Wilson CI) |
| `tests/test_hive_diffusion.py` | — | **new** (Phase 12 + NEXT_STEPS 1–9, 90 tests) |
| `adaptation_plan.md` | — | superseded by `NEXT_STEPS.md` |
| `README.md` | — | **new** (updated for NEXT_STEPS surface) |

---

## What was removed

### `vision.py` (deleted)
The entire `Gemma4VisionTower` class (SigLIP-style ViT encoder) was
deleted. The Hive model has no images.

### `projector.py` (deleted)
The `MultiModalProjector` that mapped vision tokens to text hidden
size was deleted. No replacement: the model is text-only.

### Vision config (`config.py`)
`Gemma4VisionConfig` is still defined for backwards compatibility
with the sibling `gemma_diffusion` package, but the **default
constructor in `DiffusionGemmaConfig.__init__` no longer references
it**, and no `ghive_diffusion` code imports it.

### `encode_images` and image token assumptions
Removed from `model.py`. No `image_token_id`, `boi_token_id`,
`eoi_token_id` are needed.

### Image-aware generation paths
The old `generate()` method had no image handling; the new
`hive_model.HiveDiffusionModel.generate()` continues the same
EntropyBoundSampler behaviour but operates on the pure-text stack.

---

## What was added

### `hive_config.py` — Hive-compatible configs (Phase 2)
- `HiveDiffusionConfig` — base dataclass with every field required
  by `TextBackbone` (`global_head_dim`, `num_global_key_value_heads`,
  `layer_types`, `rope_theta_for`, `partial_rotary_factor_for`).
- `HiveSmokeConfig` — 128 hidden, 4 layers, 8 experts, 64-token canvas
  for tests.
- `HiveTrainableConfig` — 384 hidden, 8 layers, 16 experts, 128-token
  canvas.
- `HiveStrongConfig` — 768 hidden, 12 layers, 32 experts, 256-token
  canvas.
- `make_smoke_config(**overrides)` — convenience builder.
- All three tiers preserve the 3:1 sliding:full layer ratio required
  by Phase 2.

### `hive_model.py` — `HiveDiffusionModel` (Phase 1, 3)
- `HiveDiffusionModel` — vision-free block-diffusion model with:
  - `TextBackbone` reused as both encoder and decoder.
  - `value_head` — Tanh scalar value from the encoder's last hidden
    state (Phase 6.3).
  - `policy_score_head` — per-legal-move scoring (Phase 5).
  - `timestep_embed` — sinusoidal noise-level conditioning (Phase
    6.1).
  - `self_cond_proj` — projects previous-step logits back into the
    decoder hidden space (Phase 7).
  - `forward_decoder(..., bypass_sliding_for_prefix=True)` — Phase 3
    fix: the cross-attention prefix is exempt from sliding-window
    pruning via `cross_prefix_mask`.
  - `score_legal_moves(...)` — encodes the context once, replays the
    KV cache for every legal move, returns `(scores, value)`.
  - `add_diffusion_noise(...)` — mask + random-replace corruption for
    diffusion training.
  - `generate(...)` — EntropyBoundSampler preserved from the original
    text+vision model for compatibility.
  - `SinusoidalTimestepEmbedding` — reusable helper.
  - `build_smoke_model(**overrides)` — convenience builder.

### `tokenizer.py` — Hive tokenizer (Phase 4)
- `HiveTokenizer` — 138-token vocab covering:
  - specials: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<mask>`, `<sep>` (IDs
    0–5).
  - task tags: `<state>`, `<features>`, `<history>`, `<legal>`,
    `<move>`, `<pv>`, `<value>`, `<candidates>`.
  - game types: `Base`, `Base+M`, ..., `Base+MLP`.
  - board states: `NotStarted`, `InProgress`, `Draw`, `WhiteWins`,
    `BlackWins`.
  - colors: `White`, `Black`.
  - pieces: every `PieceName` (`wQ`, `wS1`, ..., `bP`).
  - separators: `;`, `[`, `]`, `/`, `\`, `-`, ` `.
  - move literals: `pass`.
  - numeric tokens `n0..n63` for counts and turns.
  - value buckets `<v-4>..<v+4>` for the move+value canvas.
- `encode_text`, `encode_move`, `encode_context`, `decode`,
  `assert_roundtrip` — full bidirectional API.
- `HiveContext` dataclass — structured carrier for context encoding.
- `build_default_tokenizer(cfg)` — convenience builder.
- Unknown-token tracking via `unknown_count` / `total_count` /
  `unknown_rate`.
- The decoder implements a small state machine to glue leading
  (`/bQ`) and trailing (`bQ/`) operators correctly.

### `context_builder.py` — canonical context from Mzinga (Phase 5)
- `HiveContextBuilder` — single source of truth for how the model
  sees a board.
- Builds the canonical text format from the plan:

  ```
  <bos> <state> Base+MLP ; InProgress ; White [ 12 ]
  <features> white_queen_in_play yes ; black_queen_in_play no ; last bA2
  <history> wB1 ; bB1 wB1/ ; ...
  <legal> wA1 /bQ ; wG2 bS1- ; pass
  <move> <target_move> <eos>
  ```
- Reads `board.game_type`, `board_state`, `current_color`,
  `current_turn`, `piece_in_play`, `last_piece_moved`, board history,
  and the live legal-move list from Mzinga.

### `legal_scorer.py` — Mzinga-backed scorer (Phase 5)
- `HiveLegalScorer` — guarantees every score corresponds to a
  Mzinga-legal move.
- `legal_move_strings(board)` — canonical strings from
  `board.get_valid_moves()`.
- `score(board, return_probs)` — one `ScoredMove` per legal move.
- `best_move(board, deterministic, temperature)` — argmax or sampled.
- `value(board)` — scalar value head estimate.

### `training.py` — multi-objective training step (Phase 6, 7)
- `TrainingSample` — one supervised sample (context_ids,
  target_move_ids, legal_move_ids, target_legal_idx, value,
  timestep).
- `HiveTrainer` — single step combining:
  - **diffusion loss** — denoising CE on the masked canvas.
  - **policy loss** — CE over legal moves against the target index.
  - **value loss** — MSE on the value head prediction.
  - **MoE load-balance loss** — placeholder (zero) until `MoELayer`
    exposes router statistics (see `NEXT_STEPS.md` §1.1).
  - **self-conditioning** — with probability `self_condition_prob`,
    first run a forward pass without grad to get self-conditioning
    logits, then re-run with the detached logits added to the decoder
    hidden states.
- Gradient clipping at norm 1.0.
- Returns a metrics dict for logging.

### `canvas_formats.py` — candidate and PV canvases (Phase 8)
- `format_single_move(move, tk)` — `<move> wA1 /bQ <eos>`.
- `format_candidate_set(moves, tk)` — `<candidates> ... <eos>`.
- `format_principal_variation(pv, tk)` — `<pv> ... <eos>`.
- `format_move_with_value(move, value, tk)` — `<move> <mv> <value>
  <vN> <eos>`.
- `value_bucket_token(value)` / `bucket_token_to_value(token)` —
  quantize scalars in `[-1, 1]` to discrete `<v-4>..<v+4>`.
- `format_canvas(fmt, ..., tk)` — dispatcher.

### `dataset.py` — sample generation (Phase 9)
- `SelfPlayGenerator` — plays games with a pluggable move policy
  (default uniform random), produces one `TrainingSample` per ply,
  fills in the value target from the final game outcome from
  side-to-move perspective.
- `GameRecordDataset` — replays a Mzinga game string, producing one
  sample per ply with correct outcome back-fill.
- `game_outcome_value(board, side_to_move_color)` — outcome from a
  specific side's perspective.

### `inference.py` — fast play and MCTS (Phase 10)
- `FastPlayer` — argmax over legal-move scores (or sampled softmax).
- `MCTSPlayer` — PUCT tree search using the diffusion model as
  policy prior and value evaluator, with Dirichlet noise at the root.

### `metrics.py` — evaluation tracking (Phase 11)
- `LegalityMetrics` — parse failures, illegal-pre / illegal-post
  projection rates, canonical roundtrip, pass misuse, expansion-piece
  misuse, top-k accuracy.
- `StrengthMetrics` — win / draw / loss against baselines.
- `DiffusionMetrics` — loss by timestep bucket, entropy by step,
  candidate diversity, accepted-per-step.
- `MoEMetrics` — expert token counts, dead-expert count, router
  entropy, top-expert share.
- `MetricsTracker` — aggregator with a single `summary()` method that
  flattens everything into a flat dict keyed by category.

### `tests/test_hive_diffusion.py` — pytest suite (Phase 12)
38 tests covering:

1. Hive configs respect `TextBackbone` field requirements.
2. Smoke config is tiny enough to run on CPU.
3. Tokenizer roundtrip on 12 representative Mzinga legal moves
   (including leading and trailing operators).
4. Unknown tokens are not silently mapped to `<pad>`.
5. Tokenizer covers all `PieceName` values.
6. Tokenizer covers all `GameType` values.
7. Pass decodes correctly.
8. Special-token IDs are stable.
9. Smoke model forward produces no NaNs.
10. Value head is Tanh-bounded.
11. **Decoder attends to early encoder tokens beyond the sliding
    window** (Phase 3 verification — gradient check).
12. Legal move scorer returns one score per legal move.
13. Fast player returns a legal move.
14. Pass is only chosen when Mzinga says it is legal.
15. Piece availability by `GameType` is respected.
16. Training step computes losses without NaNs.
17. Unknown-token tracking.
18. Context builder encodes the initial position.
19. Canvas format roundtrips.
20. Value-bucket quantization roundtrips.
21. Self-play generator produces samples.
22. Game-record dataset loads.
23–25. Metrics tracker records legality, strength, MoE correctly.
26. Original `DiffusionGemmaForBlockDiffusion` still works.
27. MCTS player returns a legal move.

### `NEXT_STEPS.md`
Roadmap of follow-up work, replacing the original
`adaptation_plan.md`.

### `README.md`
Quickstart, install instructions, layout, and how to run the test
suite.

---

## What was modified

### `attention.py` — added `cross_prefix_mask`
- New optional argument `cross_prefix_mask: Optional[Tensor]` on
  `GemmaAttention.forward`. When provided, the corresponding key
  positions are exempt from the sliding-window pruning rule.
- This is the Phase 3 fix: the decoder can attend to the full encoded
  prefix even when `sliding_window` is much smaller than the prefix
  length.

### `backbone.py` — threaded `cross_prefix_mask`
- `TransformerBlock.forward` and `TextBackbone.forward` now accept
  `cross_prefix_mask` and pass it to every `GemmaAttention` call.
- No other behavioural change.

### `model.py` — vision removed, text-only retained
- Removed `Gemma4VisionTower`, `MultiModalProjector`, `encode_images`,
  vision config fallback creation, and image token assumptions.
- Kept `DiffusionGemmaForBlockDiffusion` as a text-only class with
  the same `forward_encoder`, `forward_decoder`, `generate`,
  `_accept_by_entropy_bound`, and `_renoise` methods.
- Kept for backwards compatibility with the original `sanity_check.py`
  and any external users of the text-only model.

### `__init__.py` — re-exports the new surface
- Adds exports for `HiveDiffusionConfig`, `HiveSmokeConfig`,
  `HiveTrainableConfig`, `HiveStrongConfig`, `make_smoke_config`,
  `HiveDiffusionModel`, `SinusoidalTimestepEmbedding`,
  `build_smoke_model`.
- Keeps exports for `DiffusionGemmaTextConfig`,
  `DiffusionGemmaConfig`, `MiniConfig`,
  `DiffusionGemmaForBlockDiffusion`, `GenerationOutput`.

### `moe.py` — refactored in NEXT_STEPS 1.1
- Added `MoERouterStats` (tensors only) and `RouterInfo` (tensors +
  layer/stack metadata) dataclasses.
- `MoELayer.forward` now stores `self.last_router_info` so the trainer
  can compute a Switch-style differentiable load-balance loss and the
  metrics tracker can report expert usage, entropy, and dead experts.
- `RMSNorm` and `softcap` utilities in `utils.py` are unchanged.

### `config.py` — light edits only
- `DiffusionGemmaTextConfig` retains the same fields, including the
  `use_bidirectional_attention: str = "vision"` literal (no
  behavioural impact; just a string).
- `Gemma4VisionConfig` retained for `gemma_diffusion` parity.
- `DiffusionGemmaConfig` retains `vision: Gemma4VisionConfig` field
  for parity. No `ghive_diffusion` code reads this field.

---

## Verification

```bash
# 90 ghive tests (Phase 12 + NEXT_STEPS items 1–9)
PYTHONPATH=/Users/beshir.aissi/Desktop/Random/DiffusionHive \
  uv run --project /Users/beshir.aissi/Desktop/Random/DiffusionHive/Mzinga \
  pytest /Users/beshir.aissi/Desktop/Random/DiffusionHive/ghive_diffusion/tests/ -v
# → 90 passed

# 104 Mzinga tests still pass
uv run pytest tests/
# → 104 passed

# Sanity check (uses ghive_diffusion directly)
PYTHONPATH=/Users/beshir.aissi/Desktop/Random/DiffusionHive \
  uv run --project /Users/beshir.aissi/Desktop/Random/DiffusionHive/Mzinga \
  python sanity_check.py
# → params: 1.40M, encoder/decoder/logits shapes printed
```

---

## Architectural summary

The transformation follows the principle from the plan:

> Mzinga validates and enumerates.
> DiffusionGemma scores, evaluates, proposes, and searches.

Concretely:

- **Inputs**: Mzinga boards → canonical text via `HiveContextBuilder`.
- **Encoder**: shared `TextBackbone` reads the canonical context
  causally and caches K/V.
- **Decoder**: shared `TextBackbone` runs the canvas bidirectionally
  with full cross-attention to the encoded prefix (Phase 3 fix).
- **Heads**: `value_head` and `policy_score_head` read from encoder
  and decoder respectively.
- **Training**: multi-objective `HiveTrainer` step with self-conditioning.
- **Inference**: `FastPlayer` (legal-move scoring) and `MCTSPlayer`
  (search with model priors).
- **Datasets**: `SelfPlayGenerator` and `GameRecordDataset` produce
  Mzinga-canonicalized samples.
- **Evaluation**: `MetricsTracker` aggregates legality, strength,
  diffusion, and MoE statistics.