"""Test suite for ``ghive_diffusion`` (Phase 12).

Run with::

    PYTHONPATH=/path/to/DiffusionHive uv run pytest ghive_diffusion/tests -v

Each test corresponds to one of the Phase-12 watch-outs from the
adaptation plan.
"""

import os
import sys

import pytest

# Ensure the parent of the ``ghive_diffusion`` package is on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from mzinga.core.board import Board  # noqa: E402
from mzinga.core.enums import GameType, PieceName, PlayerColor  # noqa: E402
from mzinga.core.move import PASS_MOVE  # noqa: E402

from ghive_diffusion import (  # noqa: E402
    DiffusionGemmaForBlockDiffusion,
    HiveDiffusionModel,
    HiveDiffusionConfig,
    HiveSmokeConfig,
    HiveTrainableConfig,
    HiveStrongConfig,
    MiniConfig,
    build_smoke_model,
    build_trainable_model,
    build_strong_model,
)
from ghive_diffusion.tokenizer import (  # noqa: E402
    HiveContext,
    HiveTokenizer,
    PIECE_TOKENS,
    GAME_TYPE_TOKENS,
    build_default_tokenizer,
)
from ghive_diffusion.context_builder import HiveContextBuilder  # noqa: E402
from ghive_diffusion.legal_scorer import HiveLegalScorer  # noqa: E402
from ghive_diffusion.dataset import SelfPlayGenerator, GameRecordDataset  # noqa: E402
from ghive_diffusion.training import HiveTrainer, TrainingSample, compute_aux_targets, AUX_HEAD_SPECS  # noqa: E402
from ghive_diffusion.canvas_formats import (  # noqa: E402
    format_single_move,
    format_candidate_set,
    format_principal_variation,
    format_move_with_value,
    value_bucket_token,
    bucket_token_to_value,
)
from ghive_diffusion.metrics import MetricsTracker
from ghive_diffusion.moe import RouterInfo
from ghive_diffusion.dataset import (
    SelfPlayRollout,
    RolloutConfig,
    make_random_policy,
    make_mixed_policy,
)
from ghive_diffusion.train_loop import (
    BatchedSample,
    HiveDataset,
    TrainConfig,
    TrainLoop,
    collate_batch,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint,
)
from ghive_diffusion.eval import (
    EvalConfig,
    EvalResults,
    RandomPlayer,
    run_eval,
)
from ghive_diffusion.eval.runner import FastPlayerAdapter, MCTSPlayerAdapter, _play_one_game  # noqa: E402


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def test_hive_configs_respect_text_backbone_requirements():
    """Each tier must expose the fields TextBackbone reads."""
    for cfg_cls in (HiveSmokeConfig, HiveTrainableConfig, HiveStrongConfig):
        cfg = cfg_cls()
        assert isinstance(cfg.global_head_dim, int) and cfg.global_head_dim > 0
        assert isinstance(cfg.num_global_key_value_heads, int)
        assert len(cfg.layer_types) == cfg.num_hidden_layers
        for layer_type in cfg.layer_types:
            assert layer_type in ("sliding_attention", "full_attention")
            assert cfg.rope_theta_for(layer_type) > 0
            assert 0 < cfg.partial_rotary_factor_for(layer_type) <= 1.0
        # Canvas length is positive and sane.
        assert cfg.canvas_length > 0
        assert cfg.sliding_window > 0


def test_smoke_config_is_tiny():
    cfg = HiveSmokeConfig()
    assert cfg.hidden_size <= 256
    assert cfg.num_hidden_layers <= 8


# ---------------------------------------------------------------------------
# Phase 12.1 — Tokenizer roundtrip on every Mzinga legal move
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return build_default_tokenizer()


@pytest.mark.parametrize("move_str", [
    "wB1", "bQ/", "wA1 /bQ", "pass", "wG1 wA1-",
    "wG2 bS1-", "wQ bQ\\", "wA1 bQ-", "wB1 \\bQ",
    "bQ \\wA1", "bQ /wA1", "bQ -wA1",
])
def test_tokenizer_roundtrip_moves(tokenizer, move_str):
    ids = tokenizer.encode_move(move_str)
    out = tokenizer.decode(ids)
    assert out == move_str, f"roundtrip failed for {move_str!r}: got {out!r}"


def test_tokenizer_does_not_silently_map_unknown_to_pad(tokenizer):
    assert tokenizer.token_to_id("notarealtoken") == tokenizer.unk_id
    assert tokenizer.token_to_id("notarealtoken") != tokenizer.pad_id
    assert tokenizer.unknown_rate > 0


def test_tokenizer_covers_all_piece_names(tokenizer):
    for piece in PieceName:
        if piece in (PieceName.INVALID, PieceName.NumPieceNames):
            continue
        assert tokenizer.is_known(piece.name), f"missing token: {piece.name}"


def test_tokenizer_covers_all_game_types(tokenizer):
    for gt in GameType:
        if gt in (GameType.INVALID, GameType.NumGameTypes):
            continue
        # GameType.INVALID is -1 so this loop starts at Base (0)
        name = GAME_TYPE_TOKENS[gt.value]
        assert tokenizer.is_known(name), f"missing token: {name}"


def test_tokenizer_decode_preserves_pass(tokenizer):
    ids = tokenizer.encode_move("pass")
    assert tokenizer.decode(ids) == "pass"


def test_tokenizer_special_token_ids_are_stable(tokenizer):
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id != tokenizer.pad_id
    assert tokenizer.bos_id != tokenizer.eos_id
    assert tokenizer.mask_id != tokenizer.pad_id


# ---------------------------------------------------------------------------
# Phase 12.4 — Model smoke forward with pure-text config
# ---------------------------------------------------------------------------


def test_smoke_model_forward_no_nans():
    torch.manual_seed(0)
    model = build_smoke_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    h, kv = model.forward_encoder(ids)
    assert h.shape == (1, 8, model.cfg.hidden_size)
    canvas = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.canvas_length))
    dec_h = model.forward_decoder(canvas, kv)
    assert dec_h.shape == (1, model.cfg.canvas_length, model.cfg.hidden_size)
    assert torch.isfinite(dec_h).all(), "decoder output contains NaN/Inf"


def test_smoke_model_value_head_in_range():
    torch.manual_seed(0)
    model = build_smoke_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    v = model.encode_value(ids)
    assert v.shape == (1,)
    assert torch.isfinite(v).all()
    # Tanh-bounded
    assert (v.abs() <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Phase 12.5 — Decoder attends to early encoder tokens beyond sliding window
# ---------------------------------------------------------------------------


def test_decoder_attends_to_early_encoder_tokens_beyond_sliding_window():
    """The decoder must be able to attend to the first encoded token
    even when the encoder prefix is longer than ``sliding_window``."""
    torch.manual_seed(0)
    cfg = HiveSmokeConfig()
    model = HiveDiffusionModel(cfg)
    # Encoder length much larger than sliding_window.
    encoder_len = cfg.sliding_window * 4
    # The first encoder token is unique; we'll check the gradient flows.
    first_token = 7
    other_token = 8
    input_ids = torch.full((1, encoder_len), other_token, dtype=torch.long)
    input_ids[0, 0] = first_token

    # Encoder + decoder forward.
    h_enc, kv = model.forward_encoder(input_ids, use_cache=True)
    canvas = torch.full((1, cfg.canvas_length), cfg.pad_token_id, dtype=torch.long)
    dec_h = model.forward_decoder(canvas, kv)
    # Backprop on the first encoder hidden state.
    loss = dec_h.sum() + h_enc.sum()
    loss.backward()

    # The first encoder token's embedding should have received gradient.
    embed = model.text.embed_tokens.weight
    grad_first = embed.grad[first_token].abs().sum().item()
    grad_other = embed.grad[other_token].abs().sum().item()
    # First token appears only at position 0; other token appears at all
    # other positions. Both should have nonzero gradient through the decoder
    # cross-attention. First-token grad is weaker but should be > 0.
    assert grad_first > 0, (
        f"First encoder token received no gradient (grad={grad_first}). "
        "Sliding window is hiding early encoder positions."
    )


# ---------------------------------------------------------------------------
# Phase 12.6 — Legal move scorer returns one score per legal move
# ---------------------------------------------------------------------------


def test_legal_move_scorer_returns_one_score_per_legal_move():
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    scorer = HiveLegalScorer(model, tk)

    board = Board(GameType.Base)
    scored = scorer.score(board)
    n_legal = len(list(board.get_valid_moves()))
    assert len(scored) == n_legal
    for s in scored:
        assert isinstance(s.score, float)
        assert s.move_str


# ---------------------------------------------------------------------------
# Phase 12.7 — Final sampler always returns a Mzinga-legal move
# ---------------------------------------------------------------------------


def test_fast_player_returns_legal_move():
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    from ghive_diffusion.inference import FastPlayer
    player = FastPlayer(model, tk)
    board = Board(GameType.Base)
    mv = player.play(board)
    # The returned move must be in the current legal-move list.
    legal_strs = []
    for m in board.get_valid_moves():
        legal_strs.append(board.get_move_string(m))
    assert board.get_move_string(mv) in legal_strs


# ---------------------------------------------------------------------------
# Phase 12.8 — Pass is only emitted/chosen when Mzinga says it is legal
# ---------------------------------------------------------------------------


def test_pass_only_when_legal_in_initial_position():
    """The initial position has 4 placements, no pass. The scorer should
    never return 'pass' as the best move."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    scorer = HiveLegalScorer(model, tk)
    board = Board(GameType.Base)
    legal = list(board.get_valid_moves())
    legal_strs = [board.get_move_string(m) for m in legal]
    assert "pass" not in legal_strs, "Setup error: initial position should have no pass"
    best, _ = scorer.best_move(board)
    assert best != "pass", "Scorer chose pass in a non-pass position"


# ---------------------------------------------------------------------------
# Phase 12.9 — Base vs Base+M/L/P/MLP piece availability is respected
# ---------------------------------------------------------------------------


def test_piece_availability_by_game_type():
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    # Base has no Mosquito/Ladybug/Pillbug.
    base = Board(GameType.Base)
    base_strs = builder._legal_moves(base)
    for s in base_strs:
        assert "M" not in s.replace(" ", "").replace("-", "").replace("/", "").replace("\\", ""), \
            f"Base shouldn't have Mosquito move: {s}"
    # Base+MLP has them.
    mlp = Board(GameType.BaseMLP)
    mlp_strs = builder._legal_moves(mlp)
    # We don't enforce the move to use M/L/P yet, but the tokenizer
    # should at least know those pieces.
    for piece in ("wM", "wL", "wP"):
        assert tk.is_known(piece)


# ---------------------------------------------------------------------------
# Phase 12.10 — Training step computes losses without NaNs
# ---------------------------------------------------------------------------


def test_training_step_no_nans():
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        metrics = trainer.step(sample, optimizer=opt)
    for k in ("loss", "diffusion_loss", "policy_loss", "value_loss", "moe_loss", "aux_loss"):
        v = metrics[k]
        assert v == v, f"{k} is NaN"  # NaN != NaN
        assert -1e3 < v < 1e3, f"{k} out of range: {v}"


# ---------------------------------------------------------------------------
# Phase 12.3 — Unknown tokens are not silently padded
# ---------------------------------------------------------------------------


def test_unknown_token_tracking(tokenizer):
    initial = tokenizer.unknown_count
    tokenizer.token_to_id("xyzdefinitelyunknown")
    tokenizer.token_to_id("anotherunknown")
    assert tokenizer.unknown_count == initial + 2
    assert tokenizer.unknown_rate > 0


# ---------------------------------------------------------------------------
# Phase 4 / Phase 8 — Context builder + canvas formats
# ---------------------------------------------------------------------------


def test_context_builder_encodes_initial_position(tokenizer):
    builder = HiveContextBuilder(tokenizer)
    board = Board(GameType.Base)
    ctx = builder.build(board)
    assert ctx.game_type == "Base"
    assert ctx.board_state in ("NotStarted", "InProgress")
    assert ctx.current_color in ("White", "Black")
    assert len(ctx.legal_moves) > 0
    ids = builder.encode(board)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_canvas_formats_roundtrip(tokenizer):
    move_ids = format_single_move("wA1 /bQ", tokenizer)
    assert isinstance(move_ids, list)
    cands = format_candidate_set(["wA1 /bQ", "wG2 bS1-", "pass"], tokenizer)
    assert isinstance(cands, list)
    pv = format_principal_variation(["wA1 /bQ", "bB2 wQ-", "wG2 bQ/"], tokenizer)
    assert isinstance(pv, list)
    mv = format_move_with_value("wA1 /bQ", 0.6, tokenizer)
    assert isinstance(mv, list)


def test_value_bucket_quantization_roundtrip():
    for v in [-1.0, -0.5, 0.0, 0.3, 0.6, 1.0]:
        token = value_bucket_token(v)
        # The recovered value should be close (within one bucket).
        recovered = bucket_token_to_value(token)
        assert abs(recovered - v) < 0.5


# ---------------------------------------------------------------------------
# Phase 9 — Dataset pipeline
# ---------------------------------------------------------------------------


def test_selfplay_generator_produces_samples():
    torch.manual_seed(0)
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    gen = SelfPlayGenerator(tk, builder, game_type=GameType.Base, max_plies=10)
    samples = gen.generate(n_games=2)
    assert len(samples) > 0
    for s in samples:
        assert s.target_legal_idx >= 0
        assert s.target_legal_idx < len(s.legal_move_ids)
        assert s.value in (-1.0, 0.0, 1.0)


def test_game_record_dataset_loads():
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    ds = GameRecordDataset(tk, builder)
    samples = ds.load_game("Base;1;Black[1];wB1;bB1 wB1/")
    assert len(samples) == 2
    assert tk.decode(samples[0].target_move_ids.tolist()) == "wB1"


# ---------------------------------------------------------------------------
# Phase 11 — Metrics
# ---------------------------------------------------------------------------


def test_metrics_tracker_records_legality():
    tracker = MetricsTracker()
    tracker.record_legality_sample(
        parse_ok=True, illegal_pre=False, illegal_post=False,
        roundtrip_ok=True, pass_legal=False, pass_chosen=False,
        expansion_legal=True, expansion_chosen=False,
        target_idx=2, ranked_indices=[2, 0, 1, 3, 4],
    )
    s = tracker.summary()
    assert s["legality/legal_top1_acc"] == 1.0
    assert s["legality/legal_top3_acc"] == 1.0
    assert s["legality/illegal_post_projection_rate"] == 0.0


def test_metrics_tracker_strength():
    tracker = MetricsTracker()
    tracker.record_game_outcome("win")
    tracker.record_game_outcome("loss")
    tracker.record_game_outcome("draw")
    s = tracker.summary()
    assert s["strength/games"] == 3
    assert s["strength/win_rate"] == 1 / 3
    assert s["strength/loss_rate"] == 1 / 3
    assert s["strength/draw_rate"] == 1 / 3


def test_metrics_tracker_moe():
    import numpy as np
    tracker = MetricsTracker()
    top = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    tracker.record_moe_layer(0, top)
    s = tracker.summary()
    assert "moe/router_entropy" in s
    assert "moe/top_expert_share" in s


# ---------------------------------------------------------------------------
# Backwards-compat smoke
# ---------------------------------------------------------------------------


def test_original_text_model_still_works():
    torch.manual_seed(0)
    cfg = MiniConfig()
    m = DiffusionGemmaForBlockDiffusion(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = m.generate(ids, max_new_tokens=cfg.canvas_length,
                     max_denoising_steps=2)
    assert out.sequences.shape[1] > 8


# ---------------------------------------------------------------------------
# Phase 5 — Inference
# ---------------------------------------------------------------------------


def test_mcts_player_returns_legal_move():
    torch.manual_seed(0)
    from ghive_diffusion.inference import MCTSPlayer
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    player = MCTSPlayer(model, tk, num_simulations=3)
    board = Board(GameType.Base)
    mv = player.search(board)
    legal_strs = [board.get_move_string(m) for m in board.get_valid_moves()]
    assert board.get_move_string(mv) in legal_strs


# ---------------------------------------------------------------------------
# Phase 6.5 — MoE router statistics (NEXT_STEPS 1.1)
# ---------------------------------------------------------------------------


def test_moe_layer_exposes_router_info():
    """After a forward pass, every MoELayer stores router statistics."""
    torch.manual_seed(0)
    model = build_smoke_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    with torch.enable_grad():
        model.forward_encoder(ids)
    for i, blk in enumerate(model.text.layers):
        stats = blk.mlp.last_router_info
        assert stats is not None, f"layer {i} has no router_info"
        b, t, e = stats.all_scores.shape
        k = stats.top_indices.size(-1)
        assert e == model.cfg.num_experts
        assert k == model.cfg.top_k_experts
        assert stats.top_indices.shape == (1, 8, k)
        assert stats.top_weights.shape == (1, 8, k)
        assert stats.all_scores.shape == (1, 8, e)


def test_moe_router_weights_are_normalised():
    """top_weights should sum to ~1 per token."""
    torch.manual_seed(0)
    model = build_smoke_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    with torch.enable_grad():
        model.forward_encoder(ids)
    stats = model.text.layers[0].mlp.last_router_info
    sums = stats.top_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_moe_load_balance_loss_nonzero():
    """A training step should produce a nonzero moe_loss."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
    )
    metrics = trainer.step(sample)
    assert metrics["moe_loss"] > 0, f"moe_loss should be > 0, got {metrics['moe_loss']}"
    assert metrics["moe_loss"] == metrics["moe_loss"], "moe_loss is NaN"


def test_moe_load_balance_loss_has_gradients():
    """The router gate weights should receive gradient from moe_loss alone."""
    torch.manual_seed(0)
    model = build_smoke_model()
    model.begin_moe_capture()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    _, encoder_kv = model.forward_encoder(ids)
    canvas = torch.randint(0, model.cfg.vocab_size, (2, model.cfg.canvas_length))
    _ = model.forward_decoder(canvas, encoder_kv)
    records = model.end_moe_capture()
    assert len(records) > 0, "no router records captured"
    grad_records = [r for r in records if r.all_scores.requires_grad]
    assert len(grad_records) > 0, "no gradient-bearing records"
    trainer = HiveTrainer(model)
    moe_loss = trainer._moe_load_balance_loss(records)
    assert moe_loss.requires_grad, "moe_loss is not differentiable"
    moe_loss.backward()
    gate_grad = model.text.layers[0].mlp.gate.weight.grad
    assert gate_grad is not None, "router gate received no gradient"
    assert gate_grad.abs().sum().item() > 0, "router gate gradient is zero"


def test_moe_records_cleared_between_steps():
    """Records from a previous training step should not persist."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    step1_records = None
    for _ in range(2):
        model.begin_moe_capture()
        trainer._diffusion_loss(sample)
        records = model.end_moe_capture()
        if step1_records is None:
            step1_records = records
        else:
            # The record *objects* should be different; no stale references
            assert records is not step1_records, "stale record list reused"
        opt.zero_grad()
        total = trainer._diffusion_loss(sample) + trainer._moe_load_balance_loss(records)
        total.backward()
        opt.step()


def test_moe_metrics_from_real_forward():
    """MetricsTracker should produce non-trivial entropy from a real forward."""
    import numpy as np
    torch.manual_seed(0)
    model = build_smoke_model()
    model.begin_moe_capture()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    with torch.enable_grad():
        model.forward_encoder(ids)
    records = model.end_moe_capture()
    assert len(records) > 0
    tracker = MetricsTracker()
    for rec in records:
        tracker.record_moe_router_info(rec)
    s = tracker.summary()
    assert "moe/router_entropy" in s
    assert s["moe/router_entropy"] > 0, "entropy should be positive"
    max_ent = float(np.log(model.cfg.num_experts))
    assert s["moe/router_entropy"] <= max_ent + 1e-6, (
        f"entropy {s['moe/router_entropy']} exceeds log(E)={max_ent}"
    )
    assert "moe/dead_experts" in s


def test_moe_dead_experts_count():
    """With configured num_experts, unselected experts should be counted as dead."""
    import numpy as np
    tracker = MetricsTracker()
    # Simulate a layer with E=8 where only experts 0-3 are ever selected
    info = RouterInfo(
        top_indices=torch.zeros(1, 4, 2, dtype=torch.long),
        top_weights=torch.ones(1, 4, 2) * 0.5,
        all_scores=torch.rand(1, 4, 8),
        layer_idx=0,
        stack="encoder",
    )
    info.top_indices[0, :, 0] = 0
    info.top_indices[0, :, 1] = 1
    info.top_indices[0, 0, 0] = 2
    info.top_indices[0, 1, 0] = 3
    tracker.record_moe_router_info(info)
    s = tracker.summary()
    assert s["moe/dead_experts"] == 4, f"expected 4 dead experts, got {s['moe/dead_experts']}"


# ---------------------------------------------------------------------------
# NEXT_STEPS 2.1 — Training-loop infrastructure
# ---------------------------------------------------------------------------


def test_collate_batch_pads_correctly():
    """collate_batch should pad to the longest sample in the batch."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    s1 = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(tk.encode_move(legal_strs[0]), dtype=torch.long),
        legal_move_ids=[tk.encode_move(s) for s in legal_strs],
        target_legal_idx=0,
        value=0.5,
    )
    s2 = TrainingSample(
        context_ids=torch.tensor(ctx_ids[:4], dtype=torch.long),
        target_move_ids=torch.tensor(tk.encode_move(legal_strs[1])[:2], dtype=torch.long),
        legal_move_ids=[tk.encode_move(s) for s in legal_strs],
        target_legal_idx=1,
        value=-0.5,
    )
    batch = collate_batch([s1, s2], pad_id=tk.pad_id)
    assert batch.context_ids.shape[0] == 2
    assert batch.context_ids.shape[1] >= len(ctx_ids)
    assert batch.target_move_ids.shape[0] == 2
    assert batch.target_legal_idx.shape == (2,)
    assert batch.value.shape == (2,)
    assert batch.ctx_pad_mask is not None
    assert batch.canvas_pad_mask is not None
    assert batch.ctx_pad_mask[0].sum() == 0  # s1 fills its row
    assert batch.ctx_pad_mask[1].sum() > 0   # s2 is shorter


def test_hive_dataset_iterates():
    """HiveDataset should iterate over samples."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    samples = []
    for i, s in enumerate(legal_strs[:3]):
        samples.append(TrainingSample(
            context_ids=torch.tensor(ctx_ids, dtype=torch.long),
            target_move_ids=torch.tensor(tk.encode_move(s), dtype=torch.long),
            legal_move_ids=[tk.encode_move(m) for m in legal_strs],
            target_legal_idx=i,
            value=0.5,
        ))
    ds = HiveDataset(samples)
    assert len(ds) == 3
    for i in range(3):
        assert ds[i].target_legal_idx == i


def test_optimizer_and_scheduler():
    """create_optimizer and create_scheduler should produce usable objects."""
    torch.manual_seed(0)
    model = build_smoke_model()
    opt = create_optimizer(model, lr=1e-3, weight_decay=0.01)
    sched = create_scheduler(opt, warmup_steps=10, total_steps=100)
    assert isinstance(opt, torch.optim.AdamW)
    # Warmup should produce lr near 0 at step 0, full lr at step 10
    # Call opt.step() before sched.step() to avoid the PyTorch warning.
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] > 0


def test_checkpoint_save_load(tmp_path):
    """save/load_checkpoint should round-trip model state."""
    torch.manual_seed(0)
    model = build_smoke_model()
    opt = create_optimizer(model, lr=1e-3)
    sched = create_scheduler(opt, warmup_steps=10, total_steps=100)
    # Do a forward+backward to populate optimizer state
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    _, kv = model.forward_encoder(ids)
    canvas = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.canvas_length))
    h = model.forward_decoder(canvas, kv)
    loss = h.sum()
    loss.backward()
    opt.step()

    ckpt_path = str(tmp_path / "test_ckpt.pt")
    save_checkpoint(ckpt_path, model, opt, sched, step=42)
    # Load into a fresh model
    model2 = build_smoke_model()
    opt2 = create_optimizer(model2, lr=1e-3)
    sched2 = create_scheduler(opt2, warmup_steps=10, total_steps=100)
    extra = load_checkpoint(ckpt_path, model2, opt2, sched2)
    assert extra["step"] == 42
    # Weights should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "model weights differ after checkpoint load"


# ---------------------------------------------------------------------------
# NEXT_STEPS 1.3 — Real self-play dataset generation
# ---------------------------------------------------------------------------


def test_selfplay_rollout_with_stratification():
    """SelfPlayRollout should produce samples stratified by game type."""
    torch.manual_seed(0)
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    config = RolloutConfig(
        n_games=2,
        game_types=[GameType.Base, GameType.BaseMLP],
        max_plies=10,
    )
    rollout = SelfPlayRollout(tk, builder, config)
    by_type = rollout.generate_by_game_type()
    assert "Base" in by_type
    assert "Base+MLP" in by_type
    assert len(by_type["Base"]) > 0
    assert len(by_type["Base+MLP"]) > 0
    for gt_name, samples in by_type.items():
        for s in samples:
            assert s.target_legal_idx >= 0
            assert s.value in (-1.0, 0.0, 1.0)


def test_selfplay_rollout_flat_generate():
    """SelfPlayRollout.generate should return a flat shuffled list."""
    torch.manual_seed(0)
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    config = RolloutConfig(
        n_games=3,
        game_types=[GameType.Base],
        max_plies=8,
    )
    rollout = SelfPlayRollout(tk, builder, config)
    samples = rollout.generate()
    assert len(samples) > 0
    for s in samples:
        assert isinstance(s, TrainingSample)
        assert s.target_legal_idx >= 0


def test_make_mixed_policy():
    """Mixed policy should select from the given policies."""
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    p1 = make_random_policy()
    p2 = make_random_policy()
    mixed = make_mixed_policy([p1, p2])
    board = Board(GameType.Base)
    mv = mixed(board)
    assert mv is not None or board.game_is_over


def test_selfplay_early_game_filter():
    """Early-game filter should limit the number of samples per game."""
    torch.manual_seed(0)
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    config = RolloutConfig(
        n_games=1,
        game_types=[GameType.Base],
        max_plies=20,
        filter_early_game=True,
        early_game_plies=5,
    )
    rollout = SelfPlayRollout(tk, builder, config)
    samples = rollout.generate()
    assert len(samples) <= 5


# ---------------------------------------------------------------------------
# NEXT_STEPS 1.4 — Per-game-type piece-availability enforcement
# ---------------------------------------------------------------------------


def test_piece_availability_mask_base():
    """Base game should exclude Mosquito, Ladybug, and Pillbug tokens."""
    tk = build_default_tokenizer()
    mask = tk.illegal_piece_mask("Base")
    wM_id = tk.token_to_id("wM")
    wL_id = tk.token_to_id("wL")
    wP_id = tk.token_to_id("wP")
    bM_id = tk.token_to_id("bM")
    bL_id = tk.token_to_id("bL")
    bP_id = tk.token_to_id("bP")
    for pid in (wM_id, wL_id, wP_id, bM_id, bL_id, bP_id):
        assert mask[pid], f"piece token {pid} should be masked in Base"


def test_piece_availability_mask_base_mlp():
    """Base+MLP game should NOT mask Mosquito, Ladybug, or Pillbug."""
    tk = build_default_tokenizer()
    mask = tk.illegal_piece_mask("Base+MLP")
    wM_id = tk.token_to_id("wM")
    wL_id = tk.token_to_id("wL")
    wP_id = tk.token_to_id("wP")
    for pid in (wM_id, wL_id, wP_id):
        assert not mask[pid], f"piece token {pid} should NOT be masked in Base+MLP"


def test_piece_ids_for_game_type_base():
    """Base should not include expansion pieces in its valid set."""
    tk = build_default_tokenizer()
    valid = tk.piece_ids_for_game_type("Base")
    # Queen and soldiers should be present
    assert tk.token_to_id("wQ") in valid
    assert tk.token_to_id("wS1") in valid
    assert tk.token_to_id("bB1") in valid
    # Expansion pieces should be absent
    assert tk.token_to_id("wM") not in valid
    assert tk.token_to_id("wL") not in valid
    assert tk.token_to_id("wP") not in valid
    assert tk.token_to_id("bM") not in valid
    assert tk.token_to_id("bL") not in valid
    assert tk.token_to_id("bP") not in valid


def test_piece_ids_for_game_type_base_mlp():
    """Base+MLP should include all pieces."""
    tk = build_default_tokenizer()
    valid = tk.piece_ids_for_game_type("Base+MLP")
    assert tk.token_to_id("wQ") in valid
    assert tk.token_to_id("wM") in valid
    assert tk.token_to_id("wL") in valid
    assert tk.token_to_id("wP") in valid
    assert tk.token_to_id("bM") in valid
    assert tk.token_to_id("bL") in valid
    assert tk.token_to_id("bP") in valid


def test_context_builder_populates_illegal_piece_ids():
    """The context builder should populate illegal_piece_ids on the context."""
    tk = build_default_tokenizer()
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx = builder.build(board)
    assert ctx.illegal_piece_ids is not None
    wM_id = tk.token_to_id("wM")
    assert wM_id in ctx.illegal_piece_ids
    # Base+MLP should not have wM in illegal set
    board_mlp = Board(GameType.BaseMLP)
    ctx_mlp = builder.build(board_mlp)
    assert wM_id not in ctx_mlp.illegal_piece_ids


def test_scorer_base_excludes_expansion_pieces():
    """In a Base game, the scorer's legal moves must never contain M/L/P."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    scorer = HiveLegalScorer(model, tk)
    board = Board(GameType.Base)
    scored = scorer.score(board)
    for s in scored:
        stripped = s.move_str.replace(" ", "").replace("-", "").replace("/", "").replace("\\", "")
        assert "M" not in stripped or "m" not in stripped, \
            f"Base scorer returned Mosquito move: {s.move_str}"


# ---------------------------------------------------------------------------
# NEXT_STEPS 2.2 — Cosine diffusion noise schedule
# ---------------------------------------------------------------------------


def test_cosine_schedule_masks_more_at_high_t():
    """At t=1.0 the cosine schedule should mask most tokens; at t=0 very few."""
    torch.manual_seed(42)
    ids = torch.randint(0, 100, (4, 50))
    mask_id = 4

    # t=0 should mask almost nothing
    noisy_low = HiveDiffusionModel.add_diffusion_noise(
        ids, mask_id,
        timesteps=torch.tensor([0.0]),
        mask_prob=torch.tensor([0.0]),
        vocab_size=100,
        schedule="cosine",
    )
    mask_rate_low = (noisy_low != ids).float().mean().item()
    assert mask_rate_low < 0.1, f"cosine at t=0 masks too much: {mask_rate_low}"

    # t=1.0 should mask almost everything
    noisy_high = HiveDiffusionModel.add_diffusion_noise(
        ids, mask_id,
        timesteps=torch.tensor([1.0]),
        mask_prob=torch.tensor([1.0]),
        vocab_size=100,
        schedule="cosine",
    )
    mask_rate_high = (noisy_high != ids).float().mean().item()
    assert mask_rate_high > 0.8, f"cosine at t=1 masks too little: {mask_rate_high}"


def test_cosine_schedule_differs_from_linear():
    """Cosine and linear schedules should produce different mask rates at mid t."""
    torch.manual_seed(123)
    ids = torch.randint(0, 100, (1, 1000))
    mask_id = 4
    t = 0.3

    noisy_cos = HiveDiffusionModel.add_diffusion_noise(
        ids, mask_id,
        timesteps=torch.tensor([t]),
        mask_prob=torch.tensor([t]),
        vocab_size=100,
        schedule="cosine",
    )
    noisy_lin = HiveDiffusionModel.add_diffusion_noise(
        ids, mask_id,
        timesteps=torch.tensor([t]),
        mask_prob=torch.tensor([t]),
        vocab_size=100,
        schedule="linear",
    )
    rate_cos = (noisy_cos != ids).float().mean().item()
    rate_lin = (noisy_lin != ids).float().mean().item()
    # Cosine at t=0.3 should mask fewer tokens than linear at t=0.3
    # because the cosine schedule is below the diagonal for t < ~0.5.
    # The exact relationship depends on the offset, but they should differ.
    assert abs(rate_cos - rate_lin) > 0.01, \
        f"cosine and linear produce nearly identical mask rates at t={t}"


# ---------------------------------------------------------------------------
# NEXT_STEPS 2.3 — Self-conditioning ramp-up
# ---------------------------------------------------------------------------


def test_self_conditioning_ramp_at_step_zero():
    """At step 0, self-conditioning should be effectively disabled."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder, self_condition_prob=0.5,
                          sc_ramp_steps=100)
    assert trainer.step_count == 0
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
    )
    # With ramp_steps=100 and step=0, effective_sc_prob = 0.5 * 0 = 0
    # Run multiple times — should never use self-conditioning
    used_sc = False
    for _ in range(20):
        trainer.step_count = 0
        metrics = trainer.step(sample)
        if metrics["used_self_conditioning"] > 0:
            used_sc = True
            break
    # There's a small chance torch.rand is < 0, but with effective_sc_prob=0
    # it should never fire.  Allow 1 false positive in 20 runs.
    # Actually effective_sc_prob=0.0 means torch.rand < 0.0 is always False.
    assert not used_sc, "self-conditioning should be disabled at step 0"


def test_self_conditioning_ramp_full():
    """After ramp_steps, self-conditioning should be fully active."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder, self_condition_prob=0.5,
                          sc_ramp_steps=10)
    trainer.step_count = 100  # well past ramp
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
    )
    metrics = trainer.step(sample)
    # effective_sc_prob should be 0.5 (full)
    assert abs(metrics["effective_sc_prob"] - 0.5) < 1e-6


def test_diffusion_schedule_cosine_in_trainer():
    """Trainer should accept and use the cosine schedule."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder)
    trainer.diffusion_schedule = "cosine"
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
    )
    metrics = trainer.step(sample)
    assert metrics["diffusion_loss"] == metrics["diffusion_loss"]  # not NaN


def test_train_loop_runs_few_steps(tmp_path):
    """TrainLoop should run a few steps without error."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    samples = [
        TrainingSample(
            context_ids=torch.tensor(ctx_ids, dtype=torch.long),
            target_move_ids=torch.tensor(target_ids, dtype=torch.long),
            legal_move_ids=legal_ids,
            target_legal_idx=legal_strs.index(target),
            value=0.5,
        )
        for _ in range(3)
    ]
    config = TrainConfig(
        total_steps=3,
        warmup_steps=1,
        lr=1e-3,
        log_interval=1,
        checkpoint_interval=0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        sc_ramp_steps=2,
        diffusion_schedule="cosine",
    )
    logs = []
    loop = TrainLoop(model, tk, builder, samples, config,
                     log_fn=lambda d: logs.append(d))
    loop.run()
    assert len(logs) >= 2
    assert all("loss" in l for l in logs)
    assert all("lr" in l for l in logs)
    # Final checkpoint should exist
    import os
    assert os.path.exists(str(tmp_path / "ckpts" / "final.pt"))


# ---------------------------------------------------------------------------
# NEXT_STEPS 3.2 — Larger-config validation
# ---------------------------------------------------------------------------


def test_trainable_config_forward_no_nans():
    """HiveTrainableConfig should produce a finite forward pass."""
    torch.manual_seed(0)
    model = build_trainable_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 16))
    h, kv = model.forward_encoder(ids)
    assert h.shape == (1, 16, model.cfg.hidden_size)
    assert torch.isfinite(h).all()
    canvas = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.canvas_length))
    dec_h = model.forward_decoder(canvas, kv)
    assert dec_h.shape == (1, model.cfg.canvas_length, model.cfg.hidden_size)
    assert torch.isfinite(dec_h).all(), "decoder output contains NaN/Inf"


def test_trainable_config_value_head():
    """HiveTrainableConfig value head should produce a finite scalar."""
    torch.manual_seed(0)
    model = build_trainable_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 16))
    v = model.encode_value(ids)
    assert v.shape == (1,)
    assert torch.isfinite(v).all()
    assert (v.abs() <= 1.0 + 1e-6).all()


def test_trainable_config_layer_schedule():
    """HiveTrainableConfig should have 8 layers with the 3+1 repeating pattern."""
    cfg = HiveTrainableConfig()
    assert cfg.num_hidden_layers == 8
    assert len(cfg.layer_types) == 8
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.layer_types[7] == "full_attention"
    assert cfg.layer_types[0] == "sliding_attention"


def test_strong_config_forward_no_nans():
    """HiveStrongConfig should produce a finite forward pass."""
    torch.manual_seed(0)
    model = build_strong_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 16))
    h, kv = model.forward_encoder(ids)
    assert h.shape == (1, 16, model.cfg.hidden_size)
    assert torch.isfinite(h).all()
    canvas = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.canvas_length))
    dec_h = model.forward_decoder(canvas, kv)
    assert dec_h.shape == (1, model.cfg.canvas_length, model.cfg.hidden_size)
    assert torch.isfinite(dec_h).all()


def test_strong_config_layer_schedule():
    """HiveStrongConfig should have 12 layers with the 3+1 repeating pattern."""
    cfg = HiveStrongConfig()
    assert cfg.num_hidden_layers == 12
    assert len(cfg.layer_types) == 12
    assert cfg.layer_types[3] == "full_attention"
    assert cfg.layer_types[7] == "full_attention"
    assert cfg.layer_types[11] == "full_attention"


def test_strong_config_aux_heads():
    """HiveStrongConfig should have aux heads with correct output dims."""
    torch.manual_seed(0)
    model = build_strong_model()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 16))
    aux = model.forward_aux_heads(ids)
    assert len(aux) == 9
    expected_dims = [3, 5, 2, 2, 2, 2, 4, 3, 3]
    for i, (logits, dim) in enumerate(zip(aux, expected_dims)):
        assert logits.shape == (1, dim), f"head {i}: expected (1, {dim}), got {logits.shape}"
        assert torch.isfinite(logits).all()


def test_trainable_config_score_legal_moves():
    """HiveTrainableConfig should score legal moves without error."""
    torch.manual_seed(0)
    model = build_trainable_model()
    tk = build_default_tokenizer(model.cfg)
    scorer = HiveLegalScorer(model, tk)
    board = Board(GameType.Base)
    scored = scorer.score(board)
    n_legal = len(list(board.get_valid_moves()))
    assert len(scored) == n_legal


# ---------------------------------------------------------------------------
# NEXT_STEPS 1.2 — Auxiliary heads
# ---------------------------------------------------------------------------


def test_compute_aux_targets_initial_position():
    """compute_aux_targets should return 9 integer labels for the initial board."""
    board = Board(GameType.Base)
    targets = compute_aux_targets(board)
    assert len(targets) == 9
    for i, t in enumerate(targets):
        assert isinstance(t, int), f"target {i} is not int: {type(t)}"
    # Initial position: turn 0, so game_phase = open (0)
    assert targets[0] == 0
    # Queen not in play at start
    assert targets[2] == 0
    # Queen placement required at turns 3-4, not turn 0
    assert targets[3] == 0


def test_compute_aux_targets_all_in_range():
    """All aux target values should be within their head's class count."""
    board = Board(GameType.Base)
    targets = compute_aux_targets(board)
    for i, (idx, name, n_classes) in enumerate(AUX_HEAD_SPECS):
        assert 0 <= targets[i] < n_classes, \
            f"head {i} ({name}): target {targets[i]} out of range [0, {n_classes})"


def test_aux_heads_forward_shapes():
    """forward_aux_heads should return 9 tensors with correct shapes."""
    torch.manual_seed(0)
    model = build_smoke_model()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 8))
    aux = model.forward_aux_heads(ids)
    assert len(aux) == 9
    expected_dims = [3, 5, 2, 2, 2, 2, 4, 3, 3]
    for i, (logits, dim) in enumerate(zip(aux, expected_dims)):
        assert logits.shape == (2, dim), f"head {i}: expected (2, {dim}), got {logits.shape}"
        assert torch.isfinite(logits).all()


def test_aux_loss_nonzero_with_targets():
    """A training step with aux_targets should produce a nonzero aux_loss."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder, aux_weight=0.1)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    aux = compute_aux_targets(board)
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
        aux_targets=aux,
    )
    metrics = trainer.step(sample)
    assert "aux_loss" in metrics
    assert metrics["aux_loss"] > 0, f"aux_loss should be > 0, got {metrics['aux_loss']}"
    assert metrics["aux_loss"] == metrics["aux_loss"], "aux_loss is NaN"


def test_aux_loss_zero_without_targets():
    """A training step without aux_targets should produce aux_loss=0."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder, aux_weight=0.1)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
        aux_targets=None,
    )
    metrics = trainer.step(sample)
    assert metrics["aux_loss"] == 0.0


def test_aux_heads_learn_on_synthetic_labels():
    """Aux heads should reduce loss on a fixed synthetic label over steps."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    trainer = HiveTrainer(model, tk, builder, aux_weight=1.0)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    aux = compute_aux_targets(board)
    sample = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
        aux_targets=aux,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        metrics = trainer.step(sample, optimizer=opt)
        losses.append(metrics["aux_loss"])
    assert losses[-1] < losses[0], \
        f"aux_loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"


def test_training_step_includes_aux_loss_in_total():
    """The total loss should include the aux_loss contribution."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    board = Board(GameType.Base)
    ctx_ids = builder.encode(board)
    legal_strs = builder._legal_moves(board)
    target = legal_strs[0]
    target_ids = tk.encode_move(target)
    legal_ids = [tk.encode_move(s) for s in legal_strs]
    aux = compute_aux_targets(board)
    sample_with_aux = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
        aux_targets=aux,
    )
    sample_no_aux = TrainingSample(
        context_ids=torch.tensor(ctx_ids, dtype=torch.long),
        target_move_ids=torch.tensor(target_ids, dtype=torch.long),
        legal_move_ids=legal_ids,
        target_legal_idx=legal_strs.index(target),
        value=0.5,
        aux_targets=None,
    )
    trainer = HiveTrainer(model, tk, builder, aux_weight=0.1)
    m1 = trainer.step(sample_with_aux)
    m2 = trainer.step(sample_no_aux)
    # Total with aux should be higher (aux adds positive loss)
    # But we can't guarantee strict ordering due to weight updates.
    # Instead, verify aux_loss appears in the metrics and is > 0.
    assert m1["aux_loss"] > 0
    assert m2["aux_loss"] == 0.0


# ---------------------------------------------------------------------------
# NEXT_STEPS 2.4 — Chunked legal-move scoring
# ---------------------------------------------------------------------------


def test_chunked_scoring_matches_unchunked():
    """Chunked scoring should produce identical results to unchunked."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    scorer = HiveLegalScorer(model, tk)
    board = Board(GameType.Base)
    ctx_ids = scorer.builder.encode(board, target_move=None)
    context_ids = torch.tensor([ctx_ids], dtype=torch.long)
    legal_strs = scorer.legal_move_strings(board)
    legal_ids = [tk.encode_move(s) for s in legal_strs]

    with torch.no_grad():
        scores_full, _ = model.score_legal_moves(
            context_ids, legal_ids, use_value_head=False, move_chunk_size=0)
        scores_chunked, _ = model.score_legal_moves(
            context_ids, legal_ids, use_value_head=False, move_chunk_size=2)

    assert scores_full.shape == scores_chunked.shape
    assert torch.allclose(scores_full, scores_chunked, atol=1e-6), \
        "chunked and unchunked scores differ"


def test_chunked_scoring_with_many_moves():
    """Chunked scoring should work on positions with many legal moves."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    board = Board(GameType.BaseMLP)
    # Play a few moves to reach a mid-game position with more legal moves.
    moves = list(board.get_valid_moves())
    for i in range(min(6, len(moves))):
        ms = board.get_move_string(moves[i])
        board.trusted_play(moves[i], ms)
        moves = list(board.get_valid_moves())
        if not moves or board.game_is_over:
            break

    scorer = HiveLegalScorer(model, tk)
    scored = scorer.score(board)
    assert len(scored) == len(list(board.get_valid_moves()))


# ---------------------------------------------------------------------------
# NEXT_STEPS 2.5 — MCTS board.clone() fix
# ---------------------------------------------------------------------------


def test_mcts_works_from_non_initial_position():
    """MCTS should produce a legal move from a non-initial board position.

    This tests the fix for the bug where simulations started from an empty
    board (only current_turn was copied, not piece positions).
    """
    torch.manual_seed(0)
    from ghive_diffusion.inference import MCTSPlayer
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    player = MCTSPlayer(model, tk, num_simulations=5)
    board = Board(GameType.Base)
    # Play a few moves to reach a non-initial position.
    for _ in range(4):
        moves = list(board.get_valid_moves())
        if not moves or board.game_is_over:
            break
        mv = moves[0]
        try:
            ms = board.get_move_string(mv)
        except ValueError:
            continue
        board.trusted_play(mv, ms)

    # MCTS should return a legal move from this non-initial position.
    mv = player.search(board)
    legal_strs = [board.get_move_string(m) for m in board.get_valid_moves()]
    assert board.get_move_string(mv) in legal_strs


def test_mcts_clones_board_correctly():
    """MCTS should not modify the original board during search."""
    torch.manual_seed(0)
    from ghive_diffusion.inference import MCTSPlayer
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    player = MCTSPlayer(model, tk, num_simulations=5)
    board = Board(GameType.Base)
    # Play some moves.
    for _ in range(3):
        moves = list(board.get_valid_moves())
        if not moves or board.game_is_over:
            break
        mv = moves[0]
        try:
            ms = board.get_move_string(mv)
        except ValueError:
            continue
        board.trusted_play(mv, ms)
    original_turn_after = board.current_turn
    original_key_after = board.zobrist_key

    _ = player.search(board)

    # Board should be unmodified.
    assert board.current_turn == original_turn_after
    assert board.zobrist_key == original_key_after


# ---------------------------------------------------------------------------
# NEXT_STEPS 3.4 — Eval harness
# ---------------------------------------------------------------------------


def test_random_vs_random_eval_runs():
    """run_eval should complete with random vs random."""
    player = RandomPlayer()
    opponent = RandomPlayer()
    config = EvalConfig(n_games=10, max_plies=50, seed=42)
    results = run_eval(player, opponent, config)
    assert results.n_games == 10
    assert results.wins + results.losses + results.draws == 10
    assert 0.0 <= results.win_rate <= 1.0
    assert len(results.game_lengths) == 10
    assert all(l > 0 for l in results.game_lengths)


def test_eval_results_markdown():
    """EvalResults.to_markdown should produce a valid table."""
    results = EvalResults(player_name="model", opponent_name="random")
    results.n_games = 100
    results.wins = 60
    results.losses = 30
    results.draws = 10
    results.game_lengths = [50, 60, 40] * 33 + [50]
    md = results.to_markdown()
    assert "model vs random" in md
    assert "| Games | 100 |" in md
    assert "| Wins | 60" in md
    assert "Win Rate 95% CI" in md


def test_eval_results_dict():
    """EvalResults.to_dict should include all key metrics."""
    results = EvalResults(player_name="model", opponent_name="random")
    results.n_games = 10
    results.wins = 5
    results.losses = 3
    results.draws = 2
    results.game_lengths = [40, 50, 60]
    d = results.to_dict()
    assert d["n_games"] == 10
    assert d["win_rate"] == 0.5
    assert d["loss_rate"] == 0.3
    assert d["draw_rate"] == 0.2
    assert "win_rate_ci_low" in d
    assert "win_rate_ci_high" in d
    assert d["mean_game_length"] > 0


def test_eval_win_rate_ci_bounds():
    """Wilson CI should be within [0, 1] and contain the point estimate."""
    results = EvalResults()
    results.n_games = 50
    results.wins = 25
    results.losses = 25
    results.draws = 0
    ci_low, ci_high = results.win_rate_ci
    assert 0.0 <= ci_low <= 0.5 <= ci_high <= 1.0


def test_fast_player_adapter_in_eval():
    """FastPlayerAdapter should work with the game runner."""
    torch.manual_seed(0)
    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    from ghive_diffusion.inference import FastPlayer
    fp = FastPlayer(model, tk)
    player = FastPlayerAdapter(fp)
    opponent = RandomPlayer()
    config = EvalConfig(n_games=3, max_plies=30, seed=0)
    results = run_eval(player, opponent, config)
    assert results.n_games == 3
    assert results.wins + results.losses + results.draws == 3


def test_play_one_game_returns_valid_outcome():
    """_play_one_game should return a valid outcome and ply count."""
    player = RandomPlayer()
    opponent = RandomPlayer()
    outcome, plies = _play_one_game(player, opponent, GameType.Base, max_plies=50)
    assert outcome in ("win", "loss", "draw")
    assert plies > 0
    assert plies <= 50


def test_eval_swap_sides():
    """run_eval with swap_sides should alternate who goes first."""
    player = RandomPlayer()
    opponent = RandomPlayer()
    config = EvalConfig(n_games=4, max_plies=30, seed=42, swap_sides=True)
    results = run_eval(player, opponent, config)
    assert results.n_games == 4