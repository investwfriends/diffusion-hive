# Mzinga AlphaZero — Colab Training

Self-contained AlphaZero training for Hive. Runs on Colab GPU with W&B logging and Google Drive checkpointing. Entire engine inlined — zero external deps beyond torch/numpy/wandb.

## Files

```
colab/
├── mzinga_colab.zip               ← Upload this to Colab (15KB)
├── train_colab.py                 ← Self-contained training script
├── mzinga_alphazero_colab.ipynb   ← VS Code Colab notebook (interactive)
├── run_colab.sh                   ← CLI convenience script
└── README.md
```

> ⚠️ **Performance note:** `train_colab.py` inlines the engine as it existed
> when this zip was last packaged. The optimized engine (packed positions,
> flat grid, mypyc-compiled) lives in the main repo at `src/mzinga/core/`
> and is **not** in this zip. For maximum self-play throughput on Colab,
> clone the repo instead of using the zip, then `uv run python scripts/build_mypyc.py`
> to compile the hot path. See the main `README.md`.

---

## CLI Workflow (VS Code — no browser needed)

### 1. Setup

```bash
./run_colab.sh setup
```

Installs the Google Colab VS Code extension, wandb, and rclone.

### 2. Open & run

```bash
./run_colab.sh open
```

Opens the notebook in VS Code. Select "Colab" as kernel, pick a GPU runtime, run cells top-to-bottom. Output streams to your VS Code terminal.

### 3. Monitor (from a separate terminal)

```bash
./run_colab.sh monitor     # Open W&B dashboard
./run_colab.sh status      # Check runtime + active runs
```

### 4. Sync checkpoints

```bash
./run_colab.sh sync        # Downloads .pt files from Google Drive
```

### 5. Stop

```bash
./run_colab.sh stop        # Instructions to stop the runtime
```

**Prerequisites**: VS Code, Google Colab extension (`code --install-extension google.colab`), Google account, W&B account.

---

## Browser Workflow (classic Colab notebook)

Open a new Colab notebook and paste each block as a cell:

### Cell 1 — Deps
```python
!pip install wandb -q
!pip install torch --quiet
```

### Cell 2 — Upload
```python
from google.colab import files
import os
if not os.path.exists("train_colab.py"):
    _ = files.upload()          # upload mzinga_colab.zip
    !unzip -o mzinga_colab.zip
print("Ready")
```

### Cell 3 — Mount Drive
```python
from google.colab import drive; drive.mount("/content/drive")
import os; os.makedirs("/content/drive/MyDrive/mzinga_checkpoints", exist_ok=True)
```

### Cell 4 — W&B login
```python
import wandb; wandb.login()
```

### Cell 5 — Train
```python
!python train_colab.py \
    --n_iterations 500 \
    --hidden_dim 512 --num_blocks 6 --num_sims 100 \
    --checkpoint_dir /content/drive/MyDrive/mzinga_checkpoints
```

### Cell 6 — Resume
```python
!python train_colab.py \
    --resume /content/drive/MyDrive/mzinga_checkpoints/checkpoint_0100.pt \
    --n_iterations 500 \
    --checkpoint_dir /content/drive/MyDrive/mzinga_checkpoints
```

> **Throughput tip:** If you cloned the repo (`git clone` instead of using the zip),
> add a Cell 2.5 to compile the engine before training:
> ```python
> !pip install mypy -q  # includes mypyc
> !python scripts/build_mypyc.py
> ```
> Expect ~15-20% higher MCTS throughput (see main README for measurements).

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n_iterations` | 500 | Total iterations |
| `--hidden_dim` | 512 | Hidden layer width |
| `--num_blocks` | 6 | Residual blocks |
| `--num_sims` | 100 | MCTS sims per move |
| `--games_per_iter` | 4 | Self-play games/iteration |
| `--batch_size` | 512 | Training batch |
| `--train_epochs` | 4 | Epochs per iteration |
| `--lr` | 1e-3 | Learning rate |
| `--buffer_size` | 500k | Replay buffer capacity |
| `--checkpoint_dir` | None | Checkpoint directory |
| `--checkpoint_every` | 10 | Save frequency |
| `--eval_every` | 25 | Eval frequency |
| `--eval_games` | 20 | Evaluation games |
| `--resume` | None | Resume from checkpoint |
| `--wandb_project` | mzinga-alphazero | W&B project name |

---

## GPU Tiers

| Tier | GPU | Session | ~Time for 500 iters |
|------|-----|---------|---------------------|
| Free | T4 (16GB) | ~12h | 17-25 hours |
| Pro ($10/mo) | T4/V100/A100 | ~24h | 8-20 hours |
| Pay-as-you-go | A100 (40GB) | Unlimited | 8-12 hours |

At 100 MCTS sims, GPU-scaled model (~2.8M params). Pro gives ~2x compute units and access to premium GPUs.

---

## W&B

Tracks: policy loss, value loss, win rate, gradient norms, MCTS entropy, learning rate, game stats. `wandb.watch()` captures parameter histograms. Visit your project at `https://wandb.ai/<you>/mzinga-alphazero`.

---

## Architecture

Single-file, zero external deps beyond `torch`/`numpy`/`wandb`. Inlines the complete Hive engine: Board, Move, Position (cube coordinates on flat-top hex grid), Zobrist hashing, residual MLP policy-value net, and MCTS with PUCT selection.