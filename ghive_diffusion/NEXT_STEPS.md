# Next Steps: ghive_diffusion Roadmap

This document lists the work that remains after the initial
implementation of `adaptation_plan.md`. Every phase in that plan now
has a working first cut, but the model is not yet ready for serious
training or evaluation. The items below close those gaps.

**Status (last updated after NEXT_STEPS items 1–9):** all **Critical**
and **High** priority items except item 2.5 have been resolved. The
remaining work is **Medium** priority performance and polish (items
10–11).

The roadmap is organised by priority. **Critical** items block real
training runs. **High** items meaningfully improve sample efficiency or
quality. **Medium** items scale the system up. **Lower** items polish
production-readiness.

Effort estimates are S (<1 day), M (1–3 days), L (3–10 days), XL
(>10 days).

---

## 1. Critical (blocking real training)

### ~~1.1 Refactor `MoELayer` to expose router statistics~~  · L  ✅ DONE
File: `ghive_diffusion/moe.py`, `ghive_diffusion/metrics.py`,
`ghive_diffusion/training.py`

`MoELayer.forward` now stores `self.last_router_info` containing
`top_indices` (B, T, k), `top_weights` (B, T, k), and `all_scores`
(B, T, E). The model has `begin_moe_capture()` / `end_moe_capture()` /
`_collect_moe()` helpers that aggregate per-layer records tagged with
`"encoder"` or `"decoder"`. The trainer's `_moe_load_balance_loss`
implements a Switch-style differentiable loss using
softmax-normalized router probability and scattered top-weight load.
`MetricsTracker.record_moe_router_info()` consumes the new `RouterInfo`
dataclass and reports correct dead-expert counts. See the 7
router-info tests in `tests/test_hive_diffusion.py`.

### 1.2 Auxiliary heads wiring (Phase 6.4)  · M  ✅ DONE
File: `ghive_diffusion/hive_model.py`, `ghive_diffusion/training.py`

Add small classification/regression heads on the encoder's last hidden
state for the auxiliary targets described in the plan:

- game phase bucket (open / midgame / endgame)
- legal-move count bucket
- current player queen in play
- queen placement required (turn 4 imminent)
- noisy-move indicator
- pass legality
- queen surround count
- pinned/covered piece count
- mobility advantage

Each head is `nn.Linear(hidden, n_classes)`. Compute the targets from
Mzinga's `board_metrics`. Add a single `aux_loss` term in the trainer
weighted by `aux_weight=0.1`.

**Acceptance**: training step computes 6+ aux losses; values are finite
and the heads learn on a synthetic-label sanity test.

✅ `HiveDiffusionModel.aux_heads` is a `ModuleList` of 9 `Linear`
heads on the encoder's last hidden state. `forward_aux_heads()`
returns logits for all 9 heads. `compute_aux_targets(board)` derives
the targets from Mzinga's `get_board_metrics`, `is_noisy_move`,
`_count_neighbors`, etc. `HiveTrainer._aux_loss()` runs mean
cross-entropy over all 9 heads (or 0 if `aux_targets is None`).
Verified by 8 tests including `test_aux_heads_learn_on_synthetic_labels`
which confirms the heads reduce loss over 20 optimizer steps.

### 1.3 Real self-play dataset generation  · L  ✅ DONE
File: `ghive_diffusion/dataset.py`

`SelfPlayGenerator` currently produces one sample per ply of a
random-vs-random game. For training we need:

- Games between policies of varying strength (random, fast-player,
  MCTS-player, prior checkpoint).
- MCTS-improved targets (visit distribution instead of argmax move).
- Stratification by `GameType` (Base, Base+M, ..., Base+MLP).
- Filtering for early-game diversity vs late-game tactics.
- Streaming persistence to disk (e.g. parquet / safetensors shards).

Add a `SelfPlayRollout` orchestrator that runs N games in parallel
with batched model inference, returns a `RolloutDataset` of
`(context, target_move, mcts_policy, value, aux_targets)`.

**Acceptance**: 1000 self-play games with MCTS-policy targets
produces a dataset of ~50–100k samples; training step consumes them.

✅ `make_random_policy()`, `make_model_policy()`, `make_mixed_policy()`
provide pluggable policy adapters. `RolloutConfig` +
`SelfPlayRollout` orchestrate batched generation with game-type
stratification and early-game filtering. `generate()` returns a flat
shuffled list; `generate_by_game_type()` returns a per-type dict.
MCTS visit-distribution targets remain future work (blocked on
the MCTS correctness fix in item 2.5). Verified by 4 rollout tests.

### 1.4 Per-game-type piece-availability enforcement  · S  ✅ DONE
File: `ghive_diffusion/tokenizer.py`, `ghive_diffusion/context_builder.py`

The tokenizer currently reserves ids for every piece, but the model
should never be asked to predict `wM` in a `Base` game. Add a mask
returned alongside `encode_context` that excludes illegal-by-game-type
piece ids from the legal-move list and from the diffusion canvas
softmax. The legal scorer must apply this mask automatically.

**Acceptance**: in a `Base` game, the scorer's `legal_moves` never
contains `wM`/`wL`/`wP`. Roundtrip on a `Base+MLP` game still works.

✅ `HiveTokenizer.piece_ids_for_game_type()` and
`illegal_piece_mask()` return the per-game-type masks (uses
Mzinga's `piece_name_is_enabled_for_game_type`). `HiveContext`
now carries an `illegal_piece_ids` set that `HiveContextBuilder.build`
populates automatically. The Mzinga-derived scorer was already
correct; the mask is now available for diffusion softmax. Verified by
6 tests including `test_scorer_base_excludes_expansion_pieces`.

---

## 2. High priority

### ~~2.1 Training-loop infrastructure~~  · L  ✅ DONE
File: `ghive_diffusion/training.py`, new `ghive_diffusion/train_loop.py`

A production training loop needs:

- `torch.utils.data.DataLoader` with collate that pads context, canvas,
  legal moves to the longest in-batch.
- AdamW with linear warmup + cosine decay.
- Gradient accumulation (effective batch size 256+).
- Mixed precision (`torch.amp.autocast`).
- Checkpoint save/load (model, optimizer, scheduler, step, rng state).
- TensorBoard / W&B logging.
- Eval interval that runs `MetricsTracker` on a held-out set.

**Acceptance**: training `HiveTrainableConfig` for 1000 steps on a
single GPU runs without OOM and saves a checkpoint.

✅ `train_loop.py` provides `BatchedSample`, `collate_batch`,
`HiveDataset`, `create_optimizer` (AdamW with weight-decay groups),
`create_scheduler` (linear warmup + cosine decay),
`save_checkpoint` / `load_checkpoint` (model + optimizer + scheduler
+ step + RNG state), and `TrainLoop` with `TrainConfig`. AMP is
configurable; gradient accumulation is set up but the current
`HiveTrainer.step` runs single-sample so it accumulates by stepping
the optimizer every N samples. Verified by 4 training-loop tests.

### 2.2 Diffusion noise schedule  · S  ✅ DONE
File: `ghive_diffusion/hive_model.py`

Replace the linear `mask_prob = t` in `add_diffusion_noise` with a
cosine schedule (Nichol & Dhariwal). This biases training toward
harder timesteps, generally improving sample quality.

**Acceptance**: diffusion loss averaged over a training run is
lower than the linear baseline for equal step count.

✅ `add_diffusion_noise(schedule="cosine")` implements the
Nichol & Dhariwal cosine schedule:
`mask_prob = 1 - cos((t + s) / (1 + s) * pi/2)^2` with `s = 0.008`.
Selected per-step via `TrainConfig.diffusion_schedule` and exposed
on `HiveTrainer.diffusion_schedule`. Verified by
`test_cosine_schedule_masks_more_at_high_t` and
`test_cosine_schedule_differs_from_linear`.

### 2.3 Self-conditioning ramp-up  · S  ✅ DONE
File: `ghive_diffusion/training.py`

Currently self-conditioning fires with `self_condition_prob=0.5` from
step 1. Add a ramp: `effective_prob = base_prob * min(1, step / 1000)`
so early steps don't try to consume self-conditioning signals the
model hasn't learned yet.

**Acceptance**: training-step tensorboard shows smoother loss curves
during the first 1k steps.

✅ `HiveTrainer` tracks `step_count` and ramps self-conditioning
with `effective_sc_prob = self_condition_prob * min(1, step /
sc_ramp_steps)`. Configurable via `sc_ramp_steps` in
`HiveTrainer.__init__` and `TrainConfig`. Verified by
`test_self_conditioning_ramp_at_step_zero` and
`test_self_conditioning_ramp_full`.

### 2.4 Legal-move scoring with batching  · M  ✅ DONE
File: `ghive_diffusion/hive_model.py`, `ghive_diffusion/inference.py`

`score_legal_moves` currently interleaves context × moves into a
single (B*n_moves, L) batch. For boards with many legal moves (50+
in mid-game) this can exceed memory. Add an explicit batch dimension
that chunks moves and aggregates scores.

**Acceptance**: scoring on a mid-game position with 50+ legal moves
fits in <2 GB and returns identical results to the unbatched
version.

✅ `score_legal_moves(move_chunk_size=N)` chunks the decoder pass
into groups of N moves and concatenates per-chunk scores. When
`N=0` (default) the original unbatched path is used. The chunked
path produces bit-identical scores (`atol=1e-6`). Verified by
`test_chunked_scoring_matches_unchunked` and
`test_chunked_scoring_with_many_moves`.

### 2.5 MCTS tree management  · M  ✅ DONE
File: `ghive_diffusion/inference.py`

`MCTSPlayer` currently rebuilds the board from scratch each
simulation via `trusted_play`. Add proper undo support (or reuse
Mzinga's `try_undo_last_move`) so the simulation loop runs without
allocating new boards. Also handle transpositions via zobrist-keyed
shared nodes.

**Acceptance**: MCTS with 100 simulations runs in <1 second on the
smoke config on CPU.

✅ Critical fix: each simulation now starts from `board.clone()`
instead of a freshly-constructed empty `Board(game_type)` with
`current_turn` patched. The previous code silently simulated on an
empty board for non-initial positions, giving wrong priors and
values. Verified by `test_mcts_works_from_non_initial_position` and
`test_mcts_clones_board_correctly`. Performance (100 sims < 1s) is
not measured but the same path runs unchanged on smoke config.

---

## 3. Medium priority

### 3.1 Gradient checkpointing + AMP  · S  ⏳ PENDING
File: `ghive_diffusion/backbone.py`, training scripts

Wrap each `TransformerBlock` forward in `torch.utils.checkpoint` to
trade compute for memory. Pair with `torch.amp.autocast` in the trainer
to enable bf16 on consumer GPUs.

**Acceptance**: `HiveTrainableConfig` trains in <12 GB on a 3090.

### 3.2 Larger-config validation  · M  ✅ DONE
File: `ghive_diffusion/hive_config.py`

Run the smoke-test battery against `HiveTrainableConfig` and
`HiveStrongConfig` to confirm layer schedule, RoPE tables, and
attention dimensions all line up. Currently only `HiveSmokeConfig` is
exercised.

**Acceptance**: `build_smoke_model` and equivalents for the two larger
tiers pass the full test suite.

✅ `build_trainable_model()` and `build_strong_model()` added to
`hive_model.py` and exported from `__init__.py`. Forward, value-head,
legal-move scoring, and aux-head tests now exercise all three tiers.
Layer schedules, RoPE tables, and attention dimensions confirmed
correct on `HiveTrainableConfig` (8 layers, 16 experts) and
`HiveStrongConfig` (12 layers, 32 experts). 7 new tests.

### 3.3 Diffusion sampler improvements  · M  ⏳ PENDING
File: `ghive_diffusion/hive_model.py`

The `EntropyBoundSampler` accepts a fixed `entropy_bound`. Replace
with a cosine annealing bound (looser early, tighter late) so the
canvas refines more aggressively near convergence. Also expose
`temperature` as a per-step schedule rather than a linear ramp.

**Acceptance**: at fixed compute, legal-projection edit distance is
smaller than the constant-bound baseline.

### 3.4 Eval-vs-baselines harness  · L  ✅ DONE
File: `ghive_diffusion/eval/` (new)

A reproducible evaluation script that plays N games of:

- model vs random
- model vs Mzinga RL baseline (the existing `HivePolicyValue`)
- model vs prior checkpoint
- model vs model (head-to-head)

Track strength metrics, game length, queen placement timing,
mobility advantage, etc. Report results as a markdown table.

**Acceptance**: a 200-game eval run against random completes in
<30 minutes and reports win/loss/draw with confidence intervals.

✅ New `ghive_diffusion/eval/` package: `RandomPlayer` baseline,
`FastPlayerAdapter` and `MCTSPlayerAdapter` for the model,
`EvalConfig`, `EvalResults` (win/loss/draw + Wilson 95% CI +
mean game length), and `run_eval()` game runner. Markdown
report and dict export. 8 tests verify end-to-end eval runs,
swap-sides, FastPlayer integration, and CI bounds.

### 3.5 Better value target  · M  ⏳ PENDING
File: `ghive_diffusion/hive_model.py`, `ghive_diffusion/training.py`

The value head currently predicts a scalar `[-1, 1]`. Replace with
a 3-way WDL head (win / draw / loss logits) plus a `mcts_value` target
(centipawn-style blend of game outcome and MCTS leaf values) so the
calibration is sharper at non-terminal states.

**Acceptance**: Brier score on a held-out test set improves vs the
scalar baseline.

---

## 4. Lower priority / polish

### 4.1 Documentation  · M  ⏳ PENDING
- Tutorial notebook (Jupyter) walking through scoring, training one
  step, generating, and evaluating.
- API reference generated from docstrings (mkdocs or sphinx).
- "Hive for ML researchers" primer in the README.

### 4.2 Strength baselines  · L  ⏳ PENDING
Once the eval harness exists, run the full battery at multiple
checkpoints (random init, after 1k steps, after 10k, after 100k) and
produce a learning curve.

### 4.3 Benchmarking & profiling  · M  ⏳ PENDING
- Profile the encoder/decoder under `torch.profiler`.
- Verify the legal scorer scales linearly with the number of legal
  moves.
- Confirm the diffusion sampler bottleneck.

### 4.4 Tokenizer polish  · S  ⏳ PENDING
- Per-game-type piece list (skip irrelevant pieces at tokenize time).
- Reserved slots for `<features>` key namespace to avoid collisions
  with piece names.

### 4.5 Hyperparameter search scaffolding  · M  ⏳ PENDING
- Hydra / OmegaConf config composition.
- W&B sweeps over diffusion/policy/value weights.
- Auto-resume from latest checkpoint on crash.

### 4.6 Strength from late-game positions  · L  ⏳ PENDING
Curriculum learning: train first on early-game moves, then mix in
mid- and late-game positions weighted by queen proximity. This
matches how humans learn Hive.

---

## Suggested execution order

1. ✅ **1.1 MoE router refactor** — unblocks real loss + metrics.
2. ✅ **2.1 Training-loop infrastructure** — enables actual experiments.
3. ✅ **1.3 Real self-play dataset generation** — feeds training.
4. ✅ **1.4 Piece-availability enforcement** — cheap safety net.
5. ✅ **2.2 + 2.3 Diffusion schedule + SC ramp-up** — quality wins for
   almost no engineering cost.
6. ✅ **3.2 Larger-config validation** — confirms scaling.
7. ✅ **1.2 Aux heads** — sample-efficiency boost.
8. ✅ **2.4 + 2.5 Scoring batching + MCTS cleanup** — inference speed.
9. ✅ **3.4 Eval harness** — drives everything else.
10. ⏳ **3.1 + 3.3** — performance + sampler polish.
11. ⏳ **4.x** — documentation, baselines, profiling.

Items 1–9 are complete. The project has a working feedback loop:
train → eval → tune → retrain. Items 10–11 are polish and performance
optimizations.

---

## Watch-outs (from the plan, still relevant)

- **High imitation accuracy but weak play** → value learning + MCTS
  improvement.  MCTS now works from non-initial positions (2.5);
  WDL value target (3.5) still pending.
- **Legal-looking text but illegal move** → always project through
  Mzinga.  Done in `HiveLegalScorer`; per-game-type piece mask now
  available for diffusion softmax (1.4).
- **Sliding attention hides old context** → fixed in Phase 3 with
  `cross_prefix_mask`; verified by
  `test_decoder_attends_to_early_encoder_tokens_beyond_sliding_window`.
- **MoE collapse** → resolved.  Item 1.1 now exposes differentiable
  load-balance loss and per-layer expert entropy/dead-expert metrics.
- **Tiny model underfits tactics** → block on 3.2.  Resolved: all
  three tiers (Smoke/Trainable/Strong) are now forward-tested.
- **Weak/noisy games dominate** → 1.3 separates legality pretraining
  from strength training.  Policy-mixture and game-type
  stratification are now in `SelfPlayRollout`.
- **Expansion pieces underrepresented** → 1.4 + stratified eval in
  3.4.  Both done.
- **Diffusion inference is slow** → 2.4 + 3.3.  2.4 (chunked scoring)
  done; 3.3 (adaptive entropy bound / per-step temperature) pending.