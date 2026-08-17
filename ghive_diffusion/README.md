# ghive_diffusion

Mzinga-conditioned block-diffusion policy/value model for the board game
**Hive**, built on the DiffusionGemma architecture.

This package implements the roadmap in
[`NEXT_STEPS.md`](./NEXT_STEPS.md):

```
Mzinga validates and enumerates legal moves.
ghive_diffusion scores, evaluates, proposes, and searches.
```

## Layout

| File | What it does |
|---|---|
| `hive_config.py` | `HiveDiffusionConfig` + Smoke / Trainable / Strong tiers |
| `hive_model.py` | `HiveDiffusionModel` — pure text, value + policy + aux heads, MoE router capture |
| `moe.py` | `MoELayer` with `MoERouterStats` + `RouterInfo` for load-balancing loss |
| `tokenizer.py` | `HiveTokenizer` — every token class, roundtrip, per-game-type piece mask |
| `context_builder.py` | `HiveContextBuilder` — canonical text + illegal-piece mask from Mzinga |
| `legal_scorer.py` | `HiveLegalScorer` — score every legal move via Mzinga |
| `training.py` | `HiveTrainer` — multi-objective step, self-conditioning ramp, MoE LB loss, aux loss |
| `train_loop.py` | `TrainLoop` — DataLoader, AdamW+scheduler, grad accum, AMP, checkpointing |
| `canvas_formats.py` | Single / candidate / PV / move+value canvas formats |
| `dataset.py` | `SelfPlayGenerator` + `GameRecordDataset` + `SelfPlayRollout` (game-type stratification) |
| `inference.py` | `FastPlayer` + `MCTSPlayer` (board.clone()-based simulation) |
| `metrics.py` | `MetricsTracker` (legality / strength / diffusion / MoE) |
| `eval/runner.py` | Eval harness — game runner, baselines, Wilson CI, markdown reports |
| `tests/test_hive_diffusion.py` | 90 pytest tests covering all of the above |

`config.py` and `model.py` retain a backwards-compatible text-only
`DiffusionGemmaForBlockDiffusion` for the legacy `gemma_diffusion`
package.

## Install

The package depends on the sibling `Mzinga` package. From this
directory:

```
PYTHONPATH=/path/to/DiffusionHive \
  uv run --project /path/to/DiffusionHive/Mzinga \
  python -c "from ghive_diffusion import build_smoke_model; print('ok')"
```

Or, equivalently, install both packages in editable mode and run from
either directory.

## Quickstart

```python
import torch
from mzinga.core.board import Board
from mzinga.core.enums import GameType

from ghive_diffusion import HiveDiffusionModel, HiveSmokeConfig
from ghive_diffusion.tokenizer import build_default_tokenizer
from ghive_diffusion.legal_scorer import HiveLegalScorer
from ghive_diffusion.inference import FastPlayer, MCTSPlayer

torch.manual_seed(0)
cfg = HiveSmokeConfig()
model = HiveDiffusionModel(cfg)
tk = build_default_tokenizer(cfg)
scorer = HiveLegalScorer(model, tk)

# Score the initial Hive position.
board = Board(GameType.Base)
scored = scorer.score(board, return_probs=True)
for s in scored:
    print(f"{s.move_str:>10s}  score={s.score:+.4f}")

# Play moves with the fast player.
player = FastPlayer(model, tk)
move = player.play(board)

# Or use MCTS-guided search.
mcts = MCTSPlayer(model, tk, num_simulations=50)
move = mcts.search(board)
```

## Run a short training loop

```python
from ghive_diffusion.train_loop import TrainLoop, TrainConfig

config = TrainConfig(
    total_steps=100,
    warmup_steps=10,
    lr=3e-4,
    diffusion_schedule="cosine",  # Nichol-Dhariwal
    sc_ramp_steps=20,             # self-conditioning ramp
    log_interval=10,
)
loop = TrainLoop(model, tk, builder, train_samples, config,
                 log_fn=lambda d: print(d))
loop.run()
```

## Evaluate against baselines

```python
from ghive_diffusion.eval import EvalConfig, RandomPlayer, run_eval
from ghive_diffusion.eval.runner import FastPlayerAdapter

player = FastPlayerAdapter(FastPlayer(model, tk))
opponent = RandomPlayer()
results = run_eval(player, opponent, EvalConfig(n_games=50))
print(results.to_markdown())
```

## Testing

```
PYTHONPATH=/path/to/DiffusionHive \
  uv run --project /path/to/DiffusionHive/Mzinga \
  pytest ghive_diffusion/tests/ -v
```

All 90 tests pass; they cover the Phase-12 watch-outs from the
adaptation plan plus all NEXT_STEPS items 1–9: MoE router statistics,
training-loop infrastructure, self-play rollout, piece-availability
masking, cosine diffusion schedule, self-conditioning ramp, larger-
config validation, auxiliary heads, chunked scoring, MCTS board-clone
fix, and the eval harness.

## Roadmap

See [`NEXT_STEPS.md`](./NEXT_STEPS.md) for the ordered task list.
Items 1–9 are complete; 10 (gradient checkpointing + adaptive
sampler) and 11 (documentation, baselines, profiling) remain.

See [`MIGRATION.md`](./MIGRATION.md) for the architectural history
of the transformation from the original `gemma_diffusion` package.
