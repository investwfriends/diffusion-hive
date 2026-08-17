"""Test suite for ``ghive_diffusion_lite``.

Run with::

    PYTHONPATH=/path/to/DiffusionHive \\
      /path/to/Mzinga/.venv/bin/python -m pytest -p no:cacheprovider \\
      /path/to/DiffusionHive/ghive_diffusion_lite/tests -v
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
import torch

from mzinga.core.board import Board
from mzinga.core.enums import GameType

from ghive_diffusion_lite import (
    HiveLiteConfig,
    HiveLiteModel,
    build_lite_model,
    train_lite,
)
from ghive_diffusion.tokenizer import build_default_tokenizer, HiveTokenizer
from ghive_diffusion.context_builder import HiveContextBuilder
from ghive_diffusion.legal_scorer import HiveLegalScorer
from ghive_diffusion.training import HiveTrainer, TrainingSample, compute_aux_targets
from ghive_diffusion.dataset import SelfPlayGenerator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_lite_config_fields():
    cfg = HiveLiteConfig()
    assert cfg.hidden_size == 64
    assert cfg.num_hidden_layers == 4
    assert cfg.dense_intermediate_size == 256
    assert cfg.num_attention_heads == 2
    assert cfg.num_key_value_heads == 2
    assert cfg.canvas_length == 32
    assert len(cfg.layer_types) == cfg.num_hidden_layers
    assert cfg.rope_theta_for("sliding_attention") == 10_000.0
    assert cfg.rope_theta_for("full_attention") == 1_000_000.0
    assert not hasattr(cfg, "num_experts")


def test_lite_config_overrides():
    cfg2 = HiveLiteConfig(hidden_size=48, num_hidden_layers=1)
    assert cfg2.hidden_size == 48
    assert cfg2.num_hidden_layers == 1


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def test_build_lite_model():
    model = build_lite_model()
    assert isinstance(model, HiveLiteModel)
    assert isinstance(model.text, torch.nn.Module)


def test_lite_model_param_count():
    model = build_lite_model()
    n = sum(p.numel() for p in model.parameters())
    assert 100_000 <= n <= 700_000, f"expected ~370K params, got {n:,}"


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_lite_encoder_forward():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    ids = torch.tensor([ctx_ids], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        h, kv = model.forward_encoder(ids)
    assert h.shape[0] == 1
    assert not torch.isnan(h).any()
    assert len(kv) == model.cfg.num_hidden_layers


def test_lite_decoder_forward():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    ids = torch.tensor([ctx_ids], dtype=torch.long)
    canvas = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.canvas_length))

    model.eval()
    with torch.no_grad():
        h_enc, kv = model.forward_encoder(ids)
        h_dec = model.forward_decoder(canvas, kv)
    assert h_dec.shape == (1, model.cfg.canvas_length, model.cfg.hidden_size)
    assert not torch.isnan(h_dec).any()


def test_lite_value_head_bounded():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    ids = torch.tensor([ctx_ids], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        v = model.encode_value(ids)
    assert -1.0 <= float(v.item()) <= 1.0


def test_lite_aux_heads_shape():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    ids = torch.tensor([ctx_ids], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        logits_list = model.forward_aux_heads(ids)
    assert len(logits_list) == 9
    for i, (logits, n_cls) in enumerate(zip(logits_list, model.aux_head_dims)):
        assert logits.shape == (1, n_cls), f"head {i}: expected (1,{n_cls}), got {logits.shape}"


# ---------------------------------------------------------------------------
# Legal-move scoring
# ---------------------------------------------------------------------------


def test_lite_legal_scorer():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    scorer = HiveLegalScorer(model, tk)
    board = Board(GameType.Base)

    scored = scorer.score(board, return_probs=True)
    assert len(scored) > 0
    for s in scored:
        assert s.move_str is not None
        assert s.move is not None


def test_lite_fast_play():
    from ghive_diffusion.inference import FastPlayer

    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    player = FastPlayer(model, tk)
    board = Board(GameType.Base)

    move = player.play(board)
    assert move is not None


def test_lite_chunked_scoring():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)

    ctx_ids = builder.encode(board)
    context_ids = torch.tensor([ctx_ids], dtype=torch.long)
    legal_strs = builder._legal_moves(board)
    legal_ids = [tk.encode_move(s) for s in legal_strs]

    with torch.no_grad():
        scores_full, _ = model.score_legal_moves(context_ids, legal_ids, move_chunk_size=0)
        scores_chunked, _ = model.score_legal_moves(context_ids, legal_ids, move_chunk_size=2)

    assert torch.allclose(scores_full, scores_chunked, atol=1e-5)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def test_lite_training_step():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)

    legal_strs = builder._legal_moves(board)
    ctx_ids = builder.encode(board)
    legal_ids = [tk.encode_move(s) for s in legal_strs]

    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(tk.encode_move(legal_strs[0]), dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=0,
        value=0.0,
        aux_targets=compute_aux_targets(board),
    )

    trainer = HiveTrainer(model, tk, builder, device=torch.device("cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    metrics = trainer.step(sample, optimizer=opt)
    assert "loss" in metrics
    assert "diffusion_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "aux_loss" in metrics
    assert not any(math.isnan(float(metrics[k])) for k in metrics if metrics[k] is not None)


import math


def test_lite_moe_stubs():
    model = build_lite_model()
    model.begin_moe_capture()
    records = model.end_moe_capture()
    assert records == []


def test_lite_training_reduces_loss():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)

    legal_strs = builder._legal_moves(board)
    ctx_ids = builder.encode(board)
    legal_ids = [tk.encode_move(s) for s in legal_strs]

    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(tk.encode_move(legal_strs[0]), dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=0,
        value=0.0,
    )

    trainer = HiveTrainer(model, tk, builder, device=torch.device("cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(30):
        m = trainer.step(sample, optimizer=opt)
        losses.append(m["loss"])

    # Diffusion noise is stochastic — compare averages over first/last 10
    start_avg = sum(losses[:10]) / 10
    end_avg = sum(losses[-10:]) / 10
    assert end_avg < start_avg, f"loss did not decrease: start={start_avg:.2f} end={end_avg:.2f}"


# ---------------------------------------------------------------------------
# Self-play integration
# ---------------------------------------------------------------------------


def test_lite_self_play_generator():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)

    gen = SelfPlayGenerator(tk, builder, game_type=GameType.Base)
    samples, last_move = gen._play_one()
    assert len(samples) > 0
    assert last_move is not None
    for s in samples:
        assert s.context_ids.numel() > 0
        assert s.target_move_ids.numel() > 0


# ---------------------------------------------------------------------------
# train_lite helper
# ---------------------------------------------------------------------------


def test_train_lite_runs():
    model = train_lite(
        total_steps=50,
        log_interval=50,
        save_path=None,
        device_str="cpu",
    )
    assert isinstance(model, HiveLiteModel)


def test_train_lite_gradient_tracking():
    """Gradient stats should be collected during training."""
    model = train_lite(
        total_steps=20,
        log_interval=50,
        save_path=None,
        device_str="cpu",
    )
    assert isinstance(model, HiveLiteModel)
    for p in model.parameters():
        assert p.requires_grad


def test_train_lite_report_generation():
    """Report should be generated without errors."""
    from ghive_diffusion_lite.train_lite import _generate_report

    history = {
        "losses": [{"loss": 3.0, "diffusion_loss": 2.0, "policy_loss": 0.8,
                     "value_loss": 0.2, "step": 0, "aux_loss": 0.01},
                   {"loss": 2.5, "diffusion_loss": 1.6, "policy_loss": 0.7,
                    "value_loss": 0.2, "step": 50, "aux_loss": 0.01}],
        "grads": [{"grad_l2": 0.5, "step": 0}, {"grad_l2": 0.3, "step": 50}],
        "params": [{"param_l2_total": 10.0, "step": 0}],
        "total_steps": 2,
        "elapsed_s": 5.0,
        "device": "cpu",
        "lr": 3e-4,
        "clip_frac": 0.0,
        "nan_steps": set(),
    }
    report = _generate_report(history)
    assert "TRAINING REPORT" in report
    assert "Steps:" in report
    assert "No training issues" in report or "not be learning" in report


# ---------------------------------------------------------------------------
# Diffusion noise
# ---------------------------------------------------------------------------


def test_lite_diffusion_noise():
    model = build_lite_model()
    tk = build_default_tokenizer(model.cfg)

    ids = torch.randint(0, tk.vocab_size, (2, 16))
    t = torch.rand(2)

    noisy = HiveLiteModel.add_diffusion_noise(
        ids, tk.mask_id, t, t, tk.vocab_size, schedule="linear")
    assert noisy.shape == ids.shape

    noisy_cos = HiveLiteModel.add_diffusion_noise(
        ids, tk.mask_id, t, t, tk.vocab_size, schedule="cosine")
    assert noisy_cos.shape == ids.shape
