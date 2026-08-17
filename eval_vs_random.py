#!/usr/bin/env python3
"""Standalone eval-vs-random harness for ghive_diffusion_lite checkpoints.

Compares inference strategies (greedy policy, value lookahead, value-greedy,
MCTS) against the random baseline. Does not need the Mzinga teacher binary.
"""
import argparse

import torch

from ghive_diffusion_lite import build_lite_model
from ghive_diffusion.tokenizer import build_default_tokenizer
from ghive_diffusion.inference import FastPlayer, ValuePlayer, MCTSPlayer
from ghive_diffusion.eval.runner import (
    RandomPlayer,
    run_eval,
    EvalConfig,
    FastPlayerAdapter,
    ValuePlayerAdapter,
    MCTSPlayerAdapter,
)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_lite_model()
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        step = int(ckpt.get("step", 0))
    elif isinstance(ckpt, dict) and any(
        k.startswith("text.") or k.startswith("value_head") for k in ckpt
    ):
        state = ckpt
        step = 0
    else:
        raise ValueError(f"Unrecognised checkpoint format at {path!r}")
    model.load_state_dict(state)
    model.eval()
    return model, step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--player", default="fast0",
                   choices=["fast0", "fast3", "value", "mcts"])
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-plies", type=int, default=200)
    args = p.parse_args()

    model, step = load_model(args.ckpt)
    tk = build_default_tokenizer(model.cfg)

    if args.player == "fast0":
        pl = FastPlayer(model, tk, deterministic=True,
                        lookahead_k=0, diffusion_candidates=False)
        player = FastPlayerAdapter(pl)
    elif args.player == "fast3":
        pl = FastPlayer(model, tk, deterministic=True,
                        lookahead_k=3, lookahead_weight=0.5,
                        diffusion_candidates=False)
        player = FastPlayerAdapter(pl)
    elif args.player == "value":
        pl = ValuePlayer(model, tk, deterministic=True)
        player = ValuePlayerAdapter(pl)
    elif args.player == "mcts":
        pl = MCTSPlayer(model, tk, num_simulations=16,
                        progressive_temperature=False)
        player = MCTSPlayerAdapter(pl)

    opp = RandomPlayer()
    cfg = EvalConfig(n_games=args.games, max_plies=args.max_plies,
                     swap_sides=True, seed=args.seed)
    res = run_eval(player, opp, cfg)
    print(f"RESULT step={step} player={args.player} games={res.n_games} "
          f"wins={res.wins} losses={res.losses} draws={res.draws} "
          f"win_rate={res.win_rate:.3f}", flush=True)
    print(res.to_markdown(), flush=True)


if __name__ == "__main__":
    main()
