# Migration: gemma_diffusion → ghive_diffusion

This document tracks every architectural change between the original Gemma 4
block-diffusion model and the current Hive-specific implementations.

---

## 1. Origin — `gemma_diffusion`

The original model (`gemma_diffusion/`) was a multimodal vision+text block-diffusion
model built on the Gemma 4 architecture. Key characteristics:

| Component | Details |
|---|---|
| **Vision tower** | 27-layer ViT (`Gemma4VisionTower`) with patch embedding, 3×3 pooling |
| **Projector** | `MultiModalProjector` — linear vision→text hidden dim mapping |
| **Text backbone** | 30-layer `TextBackbone`, 5:1 sliding:full attention ratio |
| **FFN** | MoE: 128 experts, top-8, plus shared expert per layer |
| **Vocabulary** | Gemma 256K-token BPE vocabulary |
| **Config** | Nested `DiffusionGemmaConfig` wrapping `DiffusionGemmaTextConfig` + `Gemma4VisionConfig` |
| **Model** | `DiffusionGemmaForBlockDiffusion` — ViT + projector + text backbone |
| **Decoding** | Shared encoder/decoder weights; bidirectional canvas with cross-attention |
| **Inference** | `generate()` — iterative entropy-bound denoising over canvas tokens |
| **Heads** | Only LM head (tied embedding) — no game-specific heads |
| **Parameters** | ~2.6B (Gemma 4 2B variant) |
| **Training** | Standard diffusion denoising loss on masked/random-replaced tokens |

Files: `config.py`, `model.py`, `backbone.py`, `attention.py`, `moe.py`, `vision.py`, `projector.py`, `utils.py`

---

## 2. Phase 1 — Vision Removal (`ghive_diffusion`)

**What was removed:**
- `Gemma4VisionTower` (entire vision stack)
- `MultiModalProjector` (vision→text bridge)
- `encode_images()` method
- All image token IDs (BOI/EOI/image_token)
- `use_bidirectional_attention: "vision"` → `"always"`

**Why:** Hive is a purely text-based domain. Board state, move history, and legal
moves are all text tokens. Vision adds 400M+ params with no benefit.

**What was added:**
- `hive_config.py` — flat `HiveDiffusionConfig` with three presets
- `hive_model.py` — `HiveDiffusionModel` replacing `DiffusionGemmaForBlockDiffusion`
- `tokenizer.py` — `HiveTokenizer` for the Hive-specific vocabulary
- `context_builder.py` — `HiveContextBuilder` for Mzinga board → tokens
- `legal_scorer.py` — `HiveLegalScorer` bridging Mzinga legality to model scores
- `inference.py` — `FastPlayer` and `MCTSPlayer`
- `training.py` — `HiveTrainer` with multi-objective losses
- `dataset.py` — self-play data generation
- `train_loop.py` — LR scheduling utilities

---

## 3. Phase 2 — Hive Vocabulary

| | gemma_diffusion | ghive_diffusion |
|---|---|---|
| **Vocab size** | 262,144 (BPE) | 138 (deterministic) |
| **Token types** | Subword pieces | Whole tokens (pieces, operators, game types) |
| **Special tokens** | `<bos>`, `<eos>`, `<pad>` | + `<mask>`, `<sep>`, `<unk>` |
| **Task tags** | None | `<state>`, `<features>`, `<history>`, `<legal>`, `<move>`, `<pv>`, `<value>`, `<candidates>` |
| **Domain tokens** | None | 26 piece tokens, 8 game types, 5 board states, 2 colors, 7 operators, 64 numeric, 9 value buckets |

The tokenizer (`tokenizer.py`) is deterministic: same input always produces the
same tokens. Unknown input maps to `<unk>` (never `<pad>`), with tracking
statistics. It also provides per-game-type piece availability masking so the model
knows which pieces are illegal for the current ruleset.

---

## 4. Phase 3 — Config Tiers

`HiveDiffusionConfig` is a flat dataclass. Three presets:

| Field | Smoke | Trainable | Strong |
|---|---|---|---|
| `hidden_size` | 128 | 384 | 768 |
| `num_hidden_layers` | 4 | 8 | 12 |
| `num_attention_heads` | 4 | 8 | 12 |
| `num_key_value_heads` | 2 | 4 | 4 |
| `head_dim` | 32 | 48 | 64 |
| `moe_intermediate_size` | 64 | 192 | 384 |
| `num_experts` | 8 | 16 | 32 |
| `top_k_experts` | 2 | 2 | 4 |
| `sliding_window` | 128 | 256 | 512 |
| `canvas_length` | 64 | 128 | 256 |
| `max_position_embeddings` | 2048 | 4096 | 8192 |
| **Params** | 1.2M | 35.0M | 372.9M |

Unlike the original which had `Gemma4VisionConfig` nested inside `DiffusionGemmaConfig`
with `__getattr__` delegation shims, the Hive config is direct.

RoPE theta and partial rotary factor are per-layer-type (sliding vs full),
same mechanism as the original.

---

## 5. Phase 4 — Hive Heads

The original had only an LM head (tied to embedding weights). `HiveDiffusionModel`
adds 12 specialized heads, all reading from the encoder's last hidden state:

### Value Head
```
encoder_last_hidden → Linear(hidden→hidden) → GELU → Linear(hidden→1) → Tanh → [-1, 1]
```
Predicts game outcome from the current side's perspective.

### Policy Score Head
```
encoder_last_hidden → Linear(hidden→1) → scalar per legal move
```
Scores each Mzinga-legal move. Used by `score_legal_moves()` which encodes the
context once, then scores all legal moves via the decoder.

### 9 Auxiliary Classification Heads
```
encoder_last_hidden → Linear(hidden→n_classes) → class logits
```

| Head | Classes | Predicts |
|---|---|---|
| 0 | 3 | Game phase (open/midgame/endgame) |
| 1 | 5 | Legal move count bucket |
| 2 | 2 | Queen in play |
| 3 | 2 | Queen placement required (turn 4) |
| 4 | 2 | Noisy move available (beetle/pillbug) |
| 5 | 2 | Pass is legal |
| 6 | 4 | Queen surround count (0, 1, 2, 3+) |
| 7 | 3 | Pinned piece count (0, 1, 2+) |
| 8 | 3 | Mobility advantage (negative/neutral/positive) |

### Timestep Embedding
Standard sinusoidal noise-level embedding (Diffusion-style). Two linear
projections with SiLU activation. Added to every canvas position in the decoder.

### Self-Conditioning Projection
```
previous_logits → Linear(vocab→hidden) → added to decoder embeddings
```
Allows the model to iteratively refine its predictions across diffusion steps.

---

## 6. Phase 5 — Legal-Move Scoring

The original model had no concept of legal moves — it unconditionally generated
token sequences. The Hive model adds `score_legal_moves()`:

1. Encode board context once → KV cache
2. For each Mzinga-legal move, tokenize into a move canvas
3. Pad all moves to same length, run through decoder
4. Take last hidden state → `policy_score_head` → scalar per move
5. Return ranked scores

This guarantees every scored move is Mzinga-legal. Chunked scoring (e.g., 32 at a
time) manages memory for positions with 50+ legal moves.

---

## 7. Phase 6 — Multi-Objective Training

The original trained on diffusion denoising alone. The Hive trainer
(`HiveTrainer.step()`) combines 5 losses with configurable weights:

| Loss | Weight | Description |
|---|---|---|
| Diffusion | 1.0 | Cross-entropy on denoising masked canvas tokens |
| Policy | 1.0 | Cross-entropy over legal-move score distribution |
| Value | 0.5 | MSE on game outcome prediction |
| MoE Load-Balance | 0.01 | Penalizes uneven expert usage (Swish-style loss) |
| Aux | 0.1 | Average CE across 9 auxiliary heads |

Self-conditioning is enabled with a ramp-up schedule (0→50% probability over 1000 steps).

Diffusion noise schedule supports both linear (default) and cosine (Nichol & Dhariwal).

Mzinga integration: `compute_aux_targets()` computes aux head labels from any
Mzinga `Board` object. `TrainingSample` includes both context + legal move lists
so the policy loss can be computed without re-enumerating moves.

---

## 8. Phase 7 — `ghive_diffusion_lite`

A lightweight variant designed for MacBook (MPS/CPU) training. Key differences
from the main model:

| | ghive_diffusion (Smoke) | ghive_diffusion_lite |
|---|---|---|
| **FFN type** | MoE (8 experts, top-2) | Dense gated GELU |
| **Params** | 1.2M | 589K |
| **Layers** | 4 | 4 |
| **hidden_size** | 128 | 64 |
| **Attention** | GQA (4:2 heads) | MHA (2:2 heads) |
| **Encoder/decoder** | Shared | Separate stacks |
| **Training** | Requires Mzinga | Pre-generated .pt datasets |
| **MoE load-balance** | Yes | No (MoE stubs are no-ops) |
| **Default diff schedule** | Linear | Cosine |

The lite model replaces `MoELayer` with `DenseMLP` — the same gated GELU FFN as a
single MoE expert, applied unconditionally to every token. All MoE infrastructure
(router, routing statistics, load-balance loss) is removed.

`LiteHiveTrainer` is a Mzinga-free training step that depends only on torch +
tokenizer + model. `LiteTrainingSample` is a standalone dataclass so pre-generated
datasets can be loaded without importing Mzinga. Training runs on Colab, MPS, or CPU.

`MzingaMCTSAdapter` generates strong training data via Mzinga's pretrained
AlphaZero MCTS engine, producing teacher-quality move targets without requiring
Mzinga at training time.

Despite the architectural differences, the lite model maintains the same interface:
same encoder/decoder, same scoring, same diffusion, same value and aux heads.
`begin_moe_capture()` / `end_moe_capture()` are no-op stubs for compatibility.

---

## 9. Phase 8 — Inference Players

### FastPlayer
Scores every legal move in a single pass. Improvements over baseline:

- **Value-head 1-ply lookahead** (Tier 1 #1): For top-k scored moves, simulates
  playing each and evaluates the resulting position via the value head. Adjusts
  scores: `adjusted = policy_score + 0.2 × (value_after - value_before)`.
- **Pondering** (Tier 1 #2): After opponent moves, pre-computes context encoding
  and aux head predictions. Cached results skip redundant encoder passes and bias
  move selection (boost queen-placement moves when required, deprioritize
  non-queen moves when queen is threatened).
- **Diffusion confidence weighting** (Tier 2 #4): Per-move entropy-based bonus —
  moves the model confidently denoises get a slight score boost.
- **Multi-pass self-cond scoring** (Tier 2 #5, default 2 passes): Iteratively
  refines hidden representations via self-conditioning.
- **Diffusion candidate validation** (Tier 2 #6): Runs a short generative
  denoising pass (16 steps) and boosts any legal move matching the model's own
  generative output.

### MCTSPlayer
Monte Carlo Tree Search using the model for both policy priors and value
evaluations. 50 simulations per move, PUCT formula, Dirichlet exploration noise
at the root.

- **Phase-guided temperature** (Tier 1 #3): Opening phase uses temperature=1.2
  for exploration; midgame uses 0.8; endgame switches to deterministic selection.

---

## 10. Optimizations Applied

### Vocab size (1024 → 256)
The config defaulted to `vocab_size=1024` but the tokenizer only uses 138 tokens.
Reducing to 256 cut 98K params (36.6%) from the lite model and 197K (14%) from
the smoke model — elimination of dead embedding rows and self-cond projection columns.

### Layers, attention, sharing (2→4, MQA→MHA, shared→unshared)
For the lite model:
- 2→4 layers (+98K): Two layers produced only one round of sliding + one round of full attention. Four layers (2 sliding + 2 full) gives the model room to compose patterns hierarchically.
- MQA→MHA (+12K): 1 KV head → 2 KV heads. Two attention heads now have independent key/value projections rather than sharing one bottleneck.
- Shared→unshared decoder (+222K): Encoder (causal prefill) and decoder (bidirectional canvas with cross-attention) are fundamentally different operations. Separate stacks allow specialization.

Total lite model: 170K → 589K (3.5× capacity increase).

---

## 11. What Was Removed

| Component | Reason |
|---|---|
| `Gemma4VisionTower` (27-layer ViT) | Hive is text-only |
| `MultiModalProjector` | No vision → text bridge needed |
| Gemma 256K vocab | Replaced with 138-token Hive vocab |
| BOI/EOI/image token IDs | No image tokens in Hive |
| `DiffusionGemmaConfig` wrapper | Flat config is simpler |
| `encode_images()` | No vision path |
| `Gemma4VisionConfig` | No vision config |
| `__getattr__` config delegation | Flat dataclass |
| MoE (lite model only) | DenseMLP is 3× smaller per layer |

---

## 12. Config Comparison

| Field | gemma_diffusion | ghive_diffusion (Smoke) | ghive_diffusion_lite |
|---|---|---|---|
| `vocab_size` | 262,144 | 256 | 256 |
| `hidden_size` | 2,816 | 128 | 64 |
| `intermediate_size` | 2,112 (dense) | — | 256 (dense) |
| `moe_intermediate_size` | 704 | 64 | — |
| `num_experts` | 128 | 8 | — |
| `top_k_experts` | 8 | 2 | — |
| `num_hidden_layers` | 30 | 4 | 4 |
| `num_attention_heads` | 16 | 4 | 2 |
| `num_key_value_heads` | 8 | 2 | 2 |
| `head_dim` | 256 | 32 | 32 |
| `num_global_key_value_heads` | 2 | 2 | 2 |
| `global_head_dim` | 512 | 32 | 32 |
| `sliding_window` | 1,024 | 128 | 64 |
| `max_position_embeddings` | 262,144 | 2,048 | 4,096 |
| `canvas_length` | 256 | 64 | 32 |
| Layer pattern | 5:1 | 3:1 | 2:2 |
| Vision tower | Yes (27 layers) | No | No |
| Heads | LM only | +value +policy +9 aux | +value +policy +9 aux |
| Enc/dec sharing | Shared | Shared | Separate |
| Params | ~2.6B | 1.2M | 589K |

---

## 13. File Map

```
gemma_diffusion/                    ghive_diffusion/                    ghive_diffusion_lite/
├── config.py        → gone         ├── hive_config.py    (new)         ├── hive_lite_config.py  (new)
├── model.py         → replaced     ├── hive_model.py     (new)         ├── hive_lite_model.py   (new)
├── backbone.py      → reused       ├── backbone.py       (forked)      └── (uses backbone.py)
├── attention.py     → reused       ├── attention.py      (copied)      └── (uses attention.py)
├── moe.py           → reused       ├── moe.py            (copied)      └── (replaced by DenseMLP)
├── utils.py         → reused       ├── utils.py          (copied)
├── vision.py        → gone         │
├── projector.py     → gone         ├── tokenizer.py      (new)
└── __init__.py                      ├── context_builder.py (new)
                                     ├── legal_scorer.py   (new)
                                     ├── inference.py      (new)
                                     ├── training.py       (new)
                                     ├── dataset.py        (new)
                                     ├── train_loop.py     (new)
                                     └── __init__.py
```
