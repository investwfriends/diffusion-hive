# Mzinga

A Python port of the Mzinga Hive game engine with AlphaZero-style training.

The engine is a from-scratch implementation of the Hive board game: piece
placement, all 8 bug types (Queen Bee, Spider, Beetle, Grasshopper, Soldier
Ant, Mosquito, Ladybug, Pillbug), zobrist hashing for state identity, and
move generation per game type (Base, Base+M, Base+L, ...).

The RL pipeline is an AlphaZero-style MCTS self-play loop with a residual
MLP policy-value network, plus PPO and a torch-free REINFORCE baseline.

## Quickstart

```bash
# 1. Install
uv sync --extra gym

# 2. (Optional, recommended) Compile hot modules with mypyc — ~30s, ~2.5-3x speedup
uv run python scripts/build_mypyc.py

# 3. Verify
uv run pytest                     # 104 tests pass
uv run mzinga-perft 5 Base        # expect ~1000 KN/s with mypyc, ~365 KN/s without
```

## Performance

Measured on Apple Silicon (M-series, single thread):

| Workload | Without mypyc | With mypyc |
|---|---|---|
| `mzinga-perft 5 Base` | 367 KN/s | 1020 KN/s (2.78×) |
| MCTS self-play, 200 sims/move | 827 sims/s | 934 sims/s (1.13×) |
| MCTS self-play, 50 sims/move | 779 sims/s | 870 sims/s (1.12×) |

To benchmark yourself:

```bash
uv run python scripts/bench_mcts_long.py 5 200  # ~5 min, recommended signal
uv run python scripts/bench_mcts.py            # ~1 min, quick sanity check
uv run python scripts/profile_hot.py mcts      # cProfile for top hot paths
```

See `scripts/FINAL_RESULTS.md` for the full table of optimizations tried,
what worked, and what didn't.

## Self-play / training

Three local training scripts, from fastest to slowest:

| Script | Backend | Time | Notes |
|---|---|---|---|
| `tests/train_demo.py` | numpy only | ~3 min | REINFORCE on compressed obs. Good for verifying the pipeline end-to-end. |
| `tests/train_alphazero.py` | torch CPU | ~2h | AlphaZero MCTS self-play, 50 sims/move, 256-hidden / 4-block net. |
| `tests/train_ppo.py` | torch CPU | ~2h | PPO-CLIP with GAE, same net size. |

All three import from `mzinga.rl` (MCTS, model, HiveEnv, dashboard).

For serious training runs (8-25 hours), see `colab/README.md` — runs on
Colab GPU with W&B logging and Drive checkpointing.

## Project layout

```
mzinga/
├── README.md                    ← this file
├── AGENTS.md                    ← AI agent instructions (build commands, conventions)
├── pyproject.toml               ← uv-managed; extras: [gym], [mypyc]
├── src/mzinga/
│   ├── core/                    ← engine (board, move, position, zobrist, fast_set)
│   │                              zero deps, mypyc-compiled for 2-3× speedup
│   ├── rl/                      ← model + MCTS (torch); intentionally NOT mypyc-compiled
│   ├── gym/                     ← Gymnasium env wrapper (numpy only)
│   ├── random/ ai/              ← stubs
│   └── mzinga_perft/            ← perft CLI entrypoint
├── tests/                       ← 104 unit tests + 3 training scripts
├── colab/                       ← self-contained Colab training (inlined engine)
└── scripts/                     ← benchmarks, profilers, mypyc build
    ├── bench_mcts.py            ← short MCTS benchmark
    ├── bench_mcts_long.py       ← long MCTS benchmark (recommended)
    ├── build_mypyc.py           ← one-shot mypyc compiler
    ├── profile_hot.py           ← cProfile hot-path discovery
    └── FINAL_RESULTS.md         ← optimization writeup
```

## Architecture notes

- The engine stores piece positions as packed 21-bit ints (8 bits q + 8 bits r + 5 bits stack offset) in a flat `list[int]` indexed by `(q+64)*128*8 + (r+64)*8 + stack`. This collapses 3 list lookups into one and avoids `Position` allocation in the hot path.
- Zobrist table is built once from bulk `os.urandom(N*8)` + `struct.unpack("<NQ")`, shared across all `Board` instances (class-level cache). Build cost: ~1.85s on first `Board()`.
- `Board.get_valid_moves()` is cached per-turn; `current_turn` setter invalidates. `Board.board_state` is computed lazily via queen-surrounded check, not eagerly.
- MCTS skips `try_get_move_string` (passes `""` to `trusted_play`) because the move string is only used for history serialization that MCTS doesn't read. Fixed a latent bug where UCB used undefined `P_current` from outer scope instead of the correct `P_root`.
- `mypyc` is applied to `mzinga.core.{position, move, fast_set, board}` only. MCTS was tried and *slowed down* by ~4% — its bottleneck is dict ops and torch interop, not Python loop overhead.

## License

Same as the upstream Mzinga project (MIT, see original repo).