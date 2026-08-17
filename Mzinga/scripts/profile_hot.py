"""Profile Mzinga's perft and MCTS to find hot paths."""
from __future__ import annotations

import cProfile
import pstats
import sys

import torch

from mzinga.core.board import Board
from mzinga.core.enums import GameType
from mzinga.rl.mcts import MCTS
from mzinga.rl.model import HivePolicyValue


def bench_mcts(num_sims=200, n_moves=50):
    torch.set_num_threads(1)
    model = HivePolicyValue(obs_dim=88, hidden_dim=128, num_blocks=2)
    model.eval()
    mcts = MCTS(model=model, num_simulations=num_sims)
    board = Board(GameType.Base)
    moves = 0
    for _ in range(n_moves):
        if board.game_is_over:
            break
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            break
        pi, pi_probs, best_a = mcts.search(board)
        best_a = int(best_a)
        if best_a >= len(valid_moves):
            best_a = 0
        move = valid_moves[best_a]
        move_str = board.try_get_move_string(move) or ""
        board.trusted_play(move, move_str)
        moves += 1
    return moves


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "mcts"
    profiler = cProfile.Profile()
    if which == "mcts":
        profiler.enable()
        bench_mcts()
        profiler.disable()
    else:
        print(f"unknown: {which}")
        return

    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumulative").print_stats(30)


if __name__ == "__main__":
    main()
