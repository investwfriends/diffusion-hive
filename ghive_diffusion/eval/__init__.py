"""Evaluation harness for ``HiveDiffusionModel`` (NEXT_STEPS 3.4).

Plays N games between a model and baselines (random, prior checkpoint,
or another model) and reports win/loss/draw with confidence intervals.

Usage::

    from ghive_diffusion.eval import EvalConfig, run_eval, RandomPlayer
    from ghive_diffusion import build_smoke_model
    from ghive_diffusion.tokenizer import build_default_tokenizer

    model = build_smoke_model()
    tk = build_default_tokenizer(model.cfg)
    player = FastPlayerAdapter(model, tk)
    opponent = RandomPlayer()
    results = run_eval(player, opponent, EvalConfig(n_games=50))
    print(results.to_markdown())
"""

from .runner import (
    BasePlayer,
    RandomPlayer,
    FastPlayerAdapter,
    MCTSPlayerAdapter,
    EvalConfig,
    EvalResults,
    run_eval,
    _play_one_game,
)

__all__ = [
    "BasePlayer",
    "RandomPlayer",
    "FastPlayerAdapter",
    "MCTSPlayerAdapter",
    "EvalConfig",
    "EvalResults",
    "run_eval",
    "_play_one_game",
]
