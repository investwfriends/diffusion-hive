# AGENTS.md

## Commands

### Setup
- Install base deps: `uv sync`
- Install training deps (torch, gymnasium): `uv sync --extra gym`
- Install dev deps (pytest, mypy): `uv sync --extra gym` (already pulls mypy via dev group)
- Install in one shot: `uv sync --extra gym`

### Build (optional, big perf win)
- Compile hot-path modules to native C via mypyc (one-time, ~30s):
  `uv run python scripts/build_mypyc.py`
  Produces `.so` files in `src/mzinga/core/` that shadow the `.py` on import.
- Revert to interpreted Python: `rm src/mzinga/core/*.so`
- After mypyc, expected perft(5) speedup: ~2.6×. MCTS self-play: ~1.15×.
  See `scripts/FINAL_RESULTS.md` for measurements.

### Tests
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_board.py::test_new_board`
- Tests must pass after any code change (currently 104 passing).

### Perft CLI
- Run: `uv run mzinga-perft [depth] [game_type] [game_string]`
- Example: `uv run mzinga-perft 5 Base`

### Benchmarks (use these to verify optimizations)
- Quick MCTS benchmark (2 games, 50 sims): `uv run python scripts/bench_mcts.py`
- Long MCTS benchmark (5 games, 200 sims, recommended): `uv run python scripts/bench_mcts_long.py 5 200`
- Profile hot paths: `uv run python scripts/profile_hot.py mcts`
- All benchmark output is saved to `scripts/*.txt` for diffing.

### Training scripts (self-play)
- Quick REINFORCE demo, no torch needed, ~3 min: `uv run python tests/train_demo.py`
- AlphaZero MCTS self-play, ~2h CPU: `uv run python tests/train_alphazero.py`
- PPO self-play, ~2h CPU: `uv run python tests/train_ppo.py`
- Full AlphaZero on Colab GPU: see `colab/README.md`

## Architecture

- `src/mzinga/core/` — Hive board engine (board, move generation, perft, position, zobrist hashing).
  Hot paths here are compiled with mypyc for a 2-3× speedup. **No torch dependency.**
- `src/mzinga/random/` and `src/mzinga/ai/` — stubs (empty inits, not functional).
- `src/mzinga/rl/` — PyTorch model + MCTS for training. MCTS uses dict operations
  and torch interop — mypyc does NOT help here (slower). Compiled as `.py` only.
- `src/mzinga/gym/` — Gymnasium environment wrapper. No torch.
- `src/mzinga_perft/` — CLI performance-test tool; the only entrypoint
- `colab/` — Self-contained Colab training (engine inlined into `train_colab.py`).
- Cube coordinates `(q, r, stack)` on a flat-top hex grid.

## Critical conventions

- `Board.calculate_perft_async(depth)` returns a coroutine — wrap with `asyncio.run()` from sync code.
  Prefer `board._calculate_perft_sync(depth)` for perft at depth >= 3 (no asyncio overhead).
- `Board.play(move)` validates and raises `InvalidMoveError`;
  `Board.trusted_play(move, move_string)` skips validation (use for perft/internals).
- `Board.get_position(piece_name)` returns a `Position` (NamedTuple).
  Internally positions are stored as packed ints; `Position` is only constructed at the API boundary.
- All commands run through `uv run` (uv-managed venv); don't invoke `python` or `pytest` directly.
- Python >= 3.11 required.
- Base engine has zero dependencies (stdlib only); torch/gymnasium are optional extras.
- No lint/typecheck/formatter config; do not run `ruff`, `mypy`, or `black`.

## Optimization state (do not regress)

The board engine has been heavily optimized. If you make changes that affect
the hot path, run `uv run python scripts/bench_mcts_long.py 5 200` before and
after to verify no regression. Hot files:

- `src/mzinga/core/board.py` — uses packed-int positions and a flat grid.
  Adding dataclass `Position` allocations here will slow things down.
- `src/mzinga/core/position.py` — `pack_position` / `unpack_q/r/stack` are the
  internal helpers. `NEIGHBOR_DQ/DR/DSTACK` tables are pre-computed.
- `src/mzinga/core/zobrist.py` — uses bulk `os.urandom` + `struct.unpack` for
  the one-time table init. Do not regress to per-call `getrandbits`.
- `src/mzinga/rl/mcts.py` — uses `P_root` (not `P_current`) in UCB. Skips
  `try_get_move_string` (passes `""` to `trusted_play`).

For the history of what was tried and what worked, see `scripts/FINAL_RESULTS.md`.