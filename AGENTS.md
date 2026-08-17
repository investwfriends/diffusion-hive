# AGENTS.md

Read this first. It covers the active code, commands, and conventions.

## Layout

- **`ghive_diffusion_lite/`** — THE ACTIVE CODE. Lightweight 589K-param
  block-diffusion policy/value model, trainer, data generation, pipeline.
- **`mzinga_uhp/`** — Pre-built Mzinga C# engine binary (the teacher).
  Driven via UHP subprocess. macOS arm64 binary included; Linux binary
  auto-downloaded by `setup_cloud.sh`.
- **`Mzinga/`** — Hive board engine (Python port, stdlib-only) + RL
  (AlphaZero model + MCTS, torch). Has its own `AGENTS.md` and
  `uv`-managed venv at `Mzinga/.venv/`. Required dependency.
- **`ghive_diffusion/`** — Full Hive block-diffusion model package
  (tokenizer, context builder, inference, eval harness). Imported by
  `ghive_diffusion_lite`.
- **`gemma_diffusion/`** — Legacy reference package (original Gemma 4
  model). Not imported by anything; kept for reference. Do not modify.
- **`alpha_zero_paper/`** — AlphaZero paper LaTeX source. Reference only.
- **`data/`** — Generated datasets. `dataset_10gb.pt` (1.35M samples,
  UHP teacher, 100% outcome backfill) is the master set;
  `dataset_200k.pt` is the 200K subsample actually used for training
  (the full set is ~800 CPU-h/epoch — untrainable locally).
- **`runs/`** — Training run checkpoints. `lite_batched_v1` is the
  current best recipe (batched training, see Round 7 in
  `ghive_diffusion_lite/IMPROVEMENT_LOG.md`); the other 7 runs are
  pre-batching history.
- **`scripts/`** — Shard pulling / merging utilities.

## Commands

All commands run through Mzinga's venv. Set `PYTHONPATH` so both
`mzinga` and the project packages resolve.

### Setup (one-time)

```bash
cd Mzinga && uv sync --extra gym && cd ..
# Optional: compile hot modules with mypyc (2.5-3x speedup, ~30s)
cd Mzinga && uv run python scripts/build_mypyc.py && cd ..
```

### Tests

```bash
# Mzinga engine tests (104)
Mzinga/.venv/bin/python -m pytest Mzinga/tests

# ghive_diffusion tests (90)
PYTHONPATH="$PWD:$PWD/Mzinga/src" Mzinga/.venv/bin/python \
  -m pytest ghive_diffusion/tests

# ghive_diffusion_lite tests
PYTHONPATH="$PWD:$PWD/Mzinga/src" Mzinga/.venv/bin/python \
  -m pytest ghive_diffusion_lite/tests
```

### Data generation (UHP teacher — default, recommended)

Use `gen_dataset.sh`. It is the single entrypoint and runs identically
on the local Mac and on a Linux cloud VM (interpreter auto-detected).

```bash
bash gen_dataset.sh                    # ~100K samples, good defaults
GAMES=4000 bash gen_dataset.sh         # bigger run
HISTORY_WINDOW=0 bash gen_dataset.sh   # full transcript (old behaviour)
```

It prints a dataset report at the end (sample count, context length,
`has_outcome` %). Check both before starting a long training run.

The underlying module still works directly if you need a one-off:

```bash
PYTHONPATH="$PWD:$PWD/Mzinga/src" Mzinga/.venv/bin/python \
  -m ghive_diffusion_lite.gen_data \
  --games 400 --workers 5 --output dataset.pt \
  --teacher uhp --uhp-depth 3 --random-fraction 0.5 \
  --history-window 40 --max-plies 300 --drop-unfinished
```

**Two flags decide whether the dataset is usable, and both are baked in
at generation time because contexts are stored pre-tokenized:**

- `--history-window N` caps `<history>` to the last N moves. Unbounded
  history grows contexts to ~2100 tokens by the midgame; attention is
  O(T²), so this is the dominant training cost. `40` measures ~850
  tokens mean. The floor is set by `<features>` and `<legal>`, and
  `<legal>` scales with branching factor, so below ~30 buys little.
- `--drop-unfinished` (with `--max-plies`) discards games that hit the
  ply cap. Those games get no terminal outcome backfill, so their
  samples carry only the teacher's weak root value. The old
  `dataset_enriched.pt` had `has_outcome=False` on **100%** of samples
  for this reason, which left the value head dead.

Each worker spawns its own `MzingaEngine` subprocess. The UHP teacher
is the native C# engine (negamax + quiescence), driven over stdin/stdout
via the Universal Hive Protocol. No torch needed for the teacher itself.

### Data generation (AlphaZero teacher — legacy)

```bash
PYTHONPATH="$PWD:$PWD/Mzinga/src" Mzinga/.venv/bin/python \
  -m ghive_diffusion_lite.gen_data \
  --games 400 --workers 5 --output dataset.pt --teacher alphazero
```

The parent preloads the AlphaZero model once; fork workers share weights
via copy-on-write. This teacher performed poorly (max 22.5% win rate vs
random) — prefer the UHP teacher.

### Training (full pipeline)

```bash
PYTHONPATH="$PWD:$PWD/Mzinga/src" Mzinga/.venv/bin/python \
  -m ghive_diffusion_lite.pipeline \
  --dataset data/dataset_200k.pt --device cpu --out-dir runs/my_run \
  --steps 20000 --batch-size 16 --lr 3e-4 --num-workers 4
```

Pipeline: train → eval (vs random) → self-play, with best-model
selection by eval win rate. Since Round 7 (2026-07-29) training is
**batched**: `--steps` counts optimizer steps, so
steps/epoch = n_samples / `--batch-size`. Key newer flags:
`--batch-size` (default 16), `--diffusion-weight` (default 0.1),
`--value-weight` (default 1.0), `--self-condition-prob` (default 0.0).
CPU and MPS are equally fast for this model; prefer CPU so parallel
eval workers (`--num-workers`) don't contend with MPS in subprocesses.
Use `--skip-selfplay` — step 3 still uses the legacy AlphaZero
teacher, not UHP.

### Sanity check

```bash
PYTHONPATH="$PWD:$PWD/Mzinga/src" Mzinga/.venv/bin/python sanity_check.py
```

## Key conventions

- **UHP teacher is the default.** `--teacher uhp` in `gen_data.py`.
  The binary lives in `mzinga_uhp/`. Set `MZINGA_ENGINE_PATH` to
  override the binary location.
- **Blend mode is the default.** `--random-fraction 0.5` produces 50%
  vs-random + 50% self-play games. This is critical for generalization:
  pure self-play risks the model never learning to punish random play;
  pure vs-random is draw-heavy at shallow depths.
- **Move-string format is shared** between the Python port and UHP.
  The Python `Board.get_game_string()` output is directly accepted by
  UHP `newgame <GameString>`.
- **Value labels**: the UHP teacher's negamax score (from
  `ReportIntermediateBestMoves`) is squashed to `[-1, +1]` via
  `tanh(score / 20000)`. Terminal game outcomes override this via
  backfill in `gen_data.py` (the strongest value signal).
- **Do not run `ruff`, `mypy`, or `black`** — no lint/typecheck config
  exists. Mzinga's `AGENTS.md` has its own conventions; respect them
  when touching `Mzinga/`.
- **Python >= 3.11** required. The Mzinga venv (`Mzinga/.venv/`) is
  the canonical environment for all commands.

## When touching `gen_data.py`

- The `_SHARED_*` globals are set in the parent before fork and inherited
  by workers. The UHP teacher must NOT be preloaded in the parent — each
  worker spawns its own `MzingaEngine` subprocess (a C# process cannot
  be COW-shared across fork).
- The `_worker` `finally` block closes the UHP subprocess. If you add
  new teacher types, ensure they clean up in `finally` too.
- `Board.get_move_string(m)` can raise `ValueError("Invalid move.")`
  for some valid moves — always wrap it in try/except when enumerating
  legal move strings.

## When touching `mzinga_uhp_adapter.py`

- The adapter syncs the engine to any board position via
  `newgame <GameString>` — one UHP command per `evaluate()` call.
- `ReportIntermediateBestMoves` must be enabled (done in `_start_engine`)
  to get scores. Intermediate lines have format `move;depth;score;PV`.
- The final `bestmove` result line is just the move (no score).
- `score_to_value(score, scale=20000)` controls the tanh squash. Adjust
  `scale` if you want more/less sensitivity to heuristic scores.

## Cloud VM

- `pack_for_cloud.sh` packages the repo (code + UHP binary + AlphaZero
  checkpoint + `gen_dataset.sh`) into `DiffusionHive_cloud.zip`.
- `setup_cloud.sh` runs on a fresh Ubuntu VM: installs Python 3.12,
  torch, Mzinga, and **downloads the Linux MzingaEngine binary** (the
  packed macOS binary is replaced with the platform-correct one).
- `gen_dataset.sh` then generates the dataset:
  `nohup bash gen_dataset.sh > gen.log 2>&1 &`
