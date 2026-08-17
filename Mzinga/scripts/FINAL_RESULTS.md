# Mzinga Optimization Results

## Summary

Optimized game simulation in two rounds:

1. **Layer 1 (pure-Python)** — packed position ints, flat piece grid, lazy
   queen-surrounded state, fixed latent MCTS UCB bug, bulk `os.urandom`
   zobrist table init, MCTS skips `try_get_move_string` (passes `""`).
2. **mypyc compilation** of `mzinga.core.{position, move, fast_set, board}`
   to native C extensions via `scripts/build_mypyc.py`.

MCTS (`mzinga.rl.mcts`) was deliberately NOT compiled — the bottleneck
inside MCTS is dict operations and torch interop, not Python loop overhead,
so mypyc adds C-level type checks that slowed it down by ~4%.

## Final numbers

Measured on Apple Silicon (M-series, single-threaded), `torch.set_num_threads(1)`:

| Workload | Baseline | + Layer 1 | + mypyc on Board |
|---|---|---|---|
| `mzinga-perft 5 Base` | 367 KN/s | 732 KN/s | **1020 KN/s** |
| MCTS self-play, 50 sims/move | 779 sims/s | 859 sims/s | 870 sims/s |
| MCTS self-play, 200 sims/move (3 games) | 827 sims/s | 859 sims/s | **934 sims/s** |

vs. baseline:
- **perft: 2.78× faster**
- **MCTS self-play: 1.13× faster**

## What was tried, what worked, what didn't

| Optimization | Perft impact | MCTS impact | Shipped? |
|---|---|---|---|
| `Position` → `NamedTuple` | +6% | 0% | yes |
| Flat piece grid (3D list → 1D int) + packed position ints | +20% | +20% | yes |
| MCTS skip `try_get_move_string` | 0% | +6% | yes |
| Lazy queen-surrounded state in `Board.board_state` | +5% | small | yes |
| Fix `P_current` → `P_root` in MCTS UCB (latent bug) | 0% | n/a (correctness) | yes |
| Bulk `os.urandom` for zobrist table init | 0% | +6% | yes |
| **mypyc on `mzinga.core.*`** | **+33%** | **+13%** | **yes** |
| mypyc on `mzinga.rl.mcts` | +38% | -4% (slower) | **no** |
| `FastSet` → `set[int]` of packed moves | -27% | small | **no, kept list** |
| `asyncio.run` for perft recursion | 0% (after fix) | n/a | **no, use sync** |

## How to verify

```bash
# install
uv sync --extra gym

# build the optimized C extensions (~30s, one-time)
uv run python scripts/build_mypyc.py

# verify tests still pass (104 tests)
uv run pytest

# measure perft — expect ~970 KN/s with mypyc, ~370 KN/s without
uv run mzinga-perft 5 Base

# measure MCTS self-play (5+ min for stable signal)
uv run python scripts/bench_mcts_long.py 5 200 128 2

# profile to confirm hot paths
uv run python scripts/profile_hot.py mcts
```

## Why mypyc helps Board but not MCTS

- **Board**: hot loops are integer arithmetic on a flat grid, with constant
  iteration counts. mypyc generates C code with no per-iteration Python
  dispatch — exactly what these loops need.
- **MCTS**: the inner search loop is dominated by `dict.get/set`, list
  comprehensions over `self.tree[sim_key].items()`, and torch/numpy interop
  (`model.eval`, `log_softmax`, `.cpu().numpy()`, `np.zeros`, etc.). Those
  C-API calls go through the same Python interpreter either way, and the
  additional type-checking mypyc inserts is a net loss.

Profiling confirmed: with mypyc on Board only, `Board.get_valid_moves` disappears
from the top 30 hot functions in the MCTS profile (it used to be ~14% of total).
The remaining MCTS time is mostly `dict.items()` iteration,
`torch.as_tensor`/`softmax`/`cpu().numpy()` round-trips, and Python lambda calls
inside `math.log`/`max(key=...)`.

## Scripts

| Script | Purpose | Time |
|---|---|---|
| `scripts/build_mypyc.py` | One-shot mypyc compiler for hot modules | ~30s |
| `scripts/bench_mcts.py` | Short MCTS benchmark (2 games, 50 sims) | ~30s |
| `scripts/bench_mcts_long.py` | Long MCTS benchmark (5 games, 200 sims, recommended) | ~5 min |
| `scripts/profile_hot.py` | cProfile-based hot-path discovery | varies |

## Saved benchmark output

`scripts/*.txt` — per-step and per-config benchmark output kept for diffing.
Key files:
- `scripts/baseline_perft.txt`, `scripts/true_baseline_*.txt` — pre-optimization baselines
- `scripts/long_final_200.txt` — final MCTS numbers (with mypyc)

## What's left (out of scope for this round)

- **MCTS Python loop itself** — would need Cython, or a vectorized batched MCTS
  using torch operations directly (no Python loop at all).
- **`board_to_obs`** — 28-piece loop per sim; could be a single batched tensor op.
- **PyPy** — would give a roughly 2× speedup on the engine, but PyTorch has no
  PyPy wheels, so it would block the trainer. Not pursued.
- **GPU acceleration** — would dominate the NN eval time and likely make the
  engine optimization irrelevant. Recommended next step if training time matters.

## Reverting

```bash
rm src/mzinga/core/*.so    # back to interpreted Python
```