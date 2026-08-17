# Round 1 Post-Mortem — Random Self-Play Training

## What we did

Trained `HiveLiteModel` (268K params, dense FFN, 2 transformer layers) for
**2000 steps** (~14 self-play games) using **uniform-random** move selection
in `SelfPlayGenerator`.  Every legal move at each ply was equally likely.

```
train_lite  device=cpu  steps=2000  lr=0.0001  policy=random
Time: 750s (12.5 min, 0.38s/step)
```

## Results

| Metric | Start | End | Δ | Verdict |
|---|---|---|---|---|
| Total loss | 17.16 | 4.25 | −75% | improving |
| Diffusion loss | 15.73 | 0.10 | −99.4% | ✅ working |
| Policy loss | 1.42 | 4.15 | +192% | ❌ getting worse |
| Value loss | 0.02 | 0.01 | −50% | ⚠️ trivial |
| Value pred mean | — | +0.01 | — | ❌ stuck at zero |
| Gradient clip rate | — | 87% | — | ❌ severe |

## Why it failed

Random self-play provides **three training signals** but only one of them
works:

### Signal 1 — Diffusion denoising ✅

The model sees a board state and a noisy canvas representing the next move.
The task: reconstruct the clean canvas.  This works *regardless* of whether
the target move is good or bad — the model just learns the mapping from
(context, noisy canvas) → clean canvas.  This is why diffusion loss dropped
99.4%.

### Signal 2 — Policy (move selection) ❌

The policy head scores legal moves against the "target" move.  When every
target is selected uniformly at random among ~10-15 legal moves, the model
**cannot learn** which moves are better.  The best it can do is predict a
uniform distribution: CE = log(12) ≈ 2.5.  The observed final CE of 4.15
means the model is doing *worse* than uniform — it's actively unlearning.

### Signal 3 — Value (position evaluation) ❌

The value head predicts the game outcome from the current position.  When
both sides play randomly, game outcomes are dominated by randomness rather
than position quality.  There is no consistent relationship between a
position's features and who wins.  The value head therefore learns to
predict the mean outcome (~0), which is the optimal strategy when no
signal exists.

### Gradient instability ⚠️

87% of steps hit the gradient-clip threshold (1.0).  This is partly due
to the small model size (268K params) where individual samples have high
variance, and partly because the random data provides no consistent
direction for the policy and value heads to follow.

## What we're changing — Round 2

Mzinga ships a **pretrained AlphaZero agent** that plays Hive with real
strategic understanding.  We use it as the data generator:

```
Before:  random policy  →  arbitrary targets  →  no policy/value signal
After:   Mzinga MCTS    →  strong targets     →  policy & value can learn
```

### MzingaMCTSAdapter

```python
from ghive_diffusion_lite import MzingaMCTSAdapter

mz = MzingaMCTSAdapter(num_simulations=50)
# mz(board) → MCTS-chosen move string
```

The adapter loads `Mzinga/colab/mzinga_alphazero_final.pt` (145K params,
AlphaZero-trained) and wraps it in Mzinga's built-in MCTS.

### Expected improvements

| Component | Round 1 (random) | Round 2 (Mzinga AI) |
|---|---|---|
| Diffusion loss | 0.10 | similar — already learned |
| Policy loss | 4.15 (noisy) | should approach ~1.0-2.0 |
| Value pred | ~0.00 (flat) | should correlate with game flow |
| Gradient clipping | 87% | should drop with better data |
| Learning rate | 1e-4 | same |

### How to run

```bash
PYTHONPATH="/Users/beshir.aissi/Desktop/Random/DiffusionHive" \
  "/Users/beshir.aissi/Desktop/Random/DiffusionHive/Mzinga/.venv/bin/python" \
  "/Users/beshir.aissi/Desktop/Random/DiffusionHive/ghive_diffusion_lite/lite_train.py"
```

Or to fall back to random self-play:

```python
from ghive_diffusion_lite.train_lite import train_lite
train_lite(use_mz_ai=False)
```

## File changes

| File | Change |
|---|---|
| `mzinga_adapter.py` | **New** — wraps Mzinga AlphaZero + MCTS |
| `train_lite.py` | Added `use_mz_ai=True`, `mz_simulations` params |
| `lite_train.py` | Updated to use Mzinga AI by default |
| `__init__.py` | Exports `MzingaMCTSAdapter` |
