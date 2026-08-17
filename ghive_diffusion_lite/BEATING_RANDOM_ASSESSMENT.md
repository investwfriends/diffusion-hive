# Assessment: Is `ghive_diffusion_lite` Sufficient to Beat Random in Hive?

> **Update (2026-07-29):** All four root causes below have now been
> addressed: the UHP teacher replaced AlphaZero, `<features>` includes
> piece coordinates (Phase 1), the dataset has 100% terminal-outcome
> backfill (`data/dataset_10gb.pt` → `dataset_200k.pt`), checkpoint
> selection is by eval win rate, and training is batched with the
> diffusion loss down-weighted (Round 7 in `IMPROVEMENT_LOG.md`).
> The remaining bottleneck this document under-weighted was **training
> throughput**: batch-1 SGD at ~2 s/step meant no run ever saw more
> than ~15% of one epoch.
>
> **Update (2026-07-26):** The root cause of the poor win rates
> documented below was a **weak teacher** (the in-repo AlphaZero model).
> The teacher has been replaced with the **native Mzinga C# engine via
> UHP** (`mzinga_uhp_adapter.py`), which is dramatically stronger. The
> analysis below remains valid as a record of the AlphaZero-teacher era;
> retrain on UHP-generated data for current results.

## Context

The user asked me to look into the `ghive_diffusion_lite` model and assess, based on my understanding of the game Hive, whether the model is sufficient to tackle the game at a very basic level — with the explicit goal of **beating a random player consistently**.

The model sits in a multi-phase lineage documented in `MIGRATION.md`:

- `gemma_diffusion` — original 2.6B Gemma-4 block-diffusion model with vision tower + MoE
- `ghive_diffusion` — Hive-specific full model (vision removed, 138-token Hive vocab, 12 specialized heads including value + policy + 9 aux classification heads)
- **`ghive_diffusion_lite`** — a lightweight 589K-parameter variant for MacBook (MPS/CPU) training: dense gated FFN replaces MoE, separate encoder/decoder stacks, pre-generated `.pt` datasets, MoE infrastructure removed as no-op stubs

The lite model uses block-diffusion over Hive text with a dedicated `score_legal_moves` pathway that scores each Mzinga-legal move via the decoder's policy head. At inference, `FastPlayer` selects moves via argmax over those scores (with optional 1-ply value-head lookahead and aux-head biasing via pondering). `MCTSPlayer` wraps the same scorers in Monte Carlo Tree Search.

### What I reviewed

| Path | Purpose |
|---|---|
| `MIGRATION.md` | Full architectural lineage from Gemma → ghive → lite |
| `ghive_diffusion_lite/hive_lite_config.py` | 589K config: hidden=64, 4 layers, 2 heads, dense FFN |
| `ghive_diffusion_lite/hive_lite_model.py` | `DenseMLP`, `LiteTransformerBlock`, `LiteBackbone`, `HiveLiteModel` |
| `ghive_diffusion_lite/lite_trainer.py` | 5-loss training: diffusion + policy + value + aux + (MoE stub) |
| `ghive_diffusion_lite/train_lite.py` | On-the-fly / dataset training loops |
| `ghive_diffusion_lite/pipeline.py` | End-to-end train → eval → self-play orchestrator |
| `ghive_diffusion_lite/gen_data.py` | Parallel MCTS self-play data generation |
| `ghive_diffusion_lite/mzinga_adapter.py` | Wraps Mzinga's pretrained `HivePolicyValue` + MCTS as teacher |
| `ghive_diffusion_lite/IMPROVEMENT_LOG.md` | 6 rounds of training experiments and findings |
| `ghive_diffusion_lite/ROUND1_POSTMORTEM.md` | Why random self-play failed |
| `ghive_diffusion/tokenizer.py` | 138-token Hive vocabulary |
| `ghive_diffusion/context_builder.py` | Board → text context encoder |
| `ghive_diffusion/inference.py` | `FastPlayer` and `MCTSPlayer` |
| `ghive_diffusion/eval/runner.py` | Game harness, `RandomPlayer`, outcome bookkeeping |
| `Mzinga/src/mzinga/rl/model.py` | Teacher model (`HivePolicyValue`) |
| `Mzinga/src/mzinga/rl/mcts.py` | `board_to_obs()` — the 88-dim teacher observation |
| `Mzinga/AGENTS.md` | Mzinga engineering conventions |
| `ghive_diffusion/TRAINING_STRATEGY.md` | Project's own roadmap |
| All `runs/*/eval_report.json` and `runs/*/metrics.jsonl` | 11 actual eval results + training metrics |

---

# Verdict

**The architecture is fundamentally viable, but as configured today it does *not* beat random consistently — and your own run history proves it.** The good news: the bottleneck is *not* the 589K parameter count or the diffusion-transformer design. It's three fixable things: **representation, objective balance, and checkpoint selection/inference.**

## Evidence from your own runs

| Run | Steps | Win | Loss | **Draw** | WR | Mean plies (cap 400) |
|---|---|---|---|---|---|---|
| lite_run1 | 72K | 3 | 6 | **31** | 7.5% | 347 |
| dagger best | 3.5K | 9 | 3 | **28** | **22.5%** (best ever) | 330 |
| merged_mps_full | 6.5K | 8 | 2 | **30** | 20% | 350 |
| selfplay_final | 76.5K | 5 | 3 | **32** | 12.5% | 368 |

Two damning patterns:

1. **65–90% of games are draws at the 400-ply cap.** Real Hive games between competent players last 20–60 plies. The model defends its queen fine (losses are rare — the aux-head defensive biasing in `FastPlayer.ponder()` works) but is *completely unable to finish*: no plan to surround the enemy queen.
2. **More training made it worse.** The 3.5K-step DAgger checkpoint beats every 70K+ checkpoint. And the latest 20K run still shows `policy_loss=3.82` ≈ uniform over ~45 legal moves (`log(45)≈3.8`) — the policy is nearly flat (opening score spread 0.05–0.15 → softmax is near-uniform midgame).

## Root causes (ranked)

**1. Representation gap — the deepest issue.** Your teacher (`HivePolicyValue`, 145K params MLP) gets an **88-dim structured vector**: every piece's actual `(q, r, stack)` hex coordinates, handed to it (`board_to_obs` in `mcts.py`). Your 589K student gets a *text transcript*: `<history> wB1 ; bB1 wB1/ ; wQ wB1- ...` and must reconstruct the entire hex grid from relative move references before it can even ask "how close am I to their queen?" Meanwhile your `<features>` section contains only 4 trivial fields (`white_queen_in_play`, `black_queen_in_play`, `last`, `turn`) — **zero spatial information**. The model spends its capacity on geometry reconstruction instead of strategy. This is why policy loss is stuck at uniform.

**2. The diffusion objective is a tax at this scale.** Diffusion loss converged to 0.1–0.63 — it's *solved* (it just memorizes move-string syntax like `wA1 /bQ`). But `FastPlayer` never calls `generate()` to pick moves — it uses `score_legal_moves` (policy head). So the diffusion machinery (canvas, timestep embeddings, self-conditioning) consumes model capacity and gradient budget for a capability that contributes ~nothing to playing strength.

**3. Value head is flat → no offense.** v_pred hovered near 0 until recently (0.59 at 20K — promising but unverified). With a flat value head, the 1-ply lookahead in `FastPlayer` and the whole `MCTSPlayer` are dead weight. Without any position evaluation, the model can't convert "ants near their queen" into an actual surround — hence the draw epidemic.

**4. Checkpoint selection metric is broken.** You select "best" by *training policy loss*, but the 3.5K-step model outplays 72K-step models. Train-loss ≠ playing strength here (distillation targets from MCTS(50) are noisy). Selection should be by **eval-vs-random win rate**.

## One nuance in your favor

Hive move notation partially self-describes geometry: `wA1 /bQ` literally means "place white Ant 1 adjacent to black Queen." So "move ants toward the enemy queen" is learnable as a *shallow* pattern without full grid reconstruction — which is exactly the level of play needed to beat random. The model isn't even hitting that shallow bar yet, which tells me the training-signal delivery (points 2–4) is as much to blame as representation.

---

# Proposed plan

**Phase 0 — Diagnostics first (cheap, ~1–2 hrs, no retraining):**
- **Ceiling check:** eval `MzingaMCTSAdapter` itself vs random (expect ~95–100%) — confirms the harness works and quantifies the gap.
- **Control experiment:** behavior-clone the teacher's 2048-action policy on the 88-dim coordinate obs with a small MLP. If it beats random easily (it should), representation is *proven* to be the bottleneck — justifying Phase 1.

**Phase 1 — Representation fix (highest leverage):**
- Enrich `<features>` in `HiveContextBuilder`: add per-piece coordinates (your tokenizer already has `n0`–`n63`; offset-encode signed coords, e.g. `wQ n35 n30 n0`) and queen surround counts (own + enemy).
- Cost caveat: the pre-generated 10.5 GB `dataset_merged.pt` stores *pre-tokenized* contexts — coords can't be retrofitted. Regenerate a smaller enriched dataset (~50–100K samples) via `gen_data.py`, or use on-the-fly `train_lite` (builder runs live, picks up enrichment for free).

**Phase 2 — Training config:**
- Rebalance losses: policy 1.0 constant, diffusion 0.05–0.1 (or freeze it after convergence), value 1.0 with asymmetric data.
- Select checkpoints by **eval-vs-random win rate**, not train policy loss.

**Phase 3 — Inference:**
- Once v_pred shows real variance, switch eval to `MCTSPlayer` (50 sims) or `FastPlayer(lookahead_k=3)`. Random hangs its queen constantly; even a modest value signal + shallow search should convert draws into wins.

**Phase 4 — Only if needed:** scale to the `Smoke` config (1.2M, MoE) with GPU batched training per your `TRAINING_STRATEGY.md`, then the >70% gate → AlphaZero self-play loop.

---

Two questions before building this out:

1. **Is the diffusion component a research goal in itself, or just scaffolding?** If the milestone is purely "beat random," I'd slash the diffusion weight to ~0.05. If block-diffusion generation is part of the point, I'll keep it meaningful and lean harder on representation + value fixes instead.

2. **Compute path:** regenerate an enriched teacher dataset with `gen_data.py` (parallel MCTS, CPU-hours but hands-off), or on-the-fly training (slower per step, no regen), or Colab GPU per your notebook?