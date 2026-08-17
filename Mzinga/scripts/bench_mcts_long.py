"""Longer MCTS self-play benchmark for Mzinga optimization work.

Runs N self-play games with MCTS(num_sims=S) and reports throughput.
Used to measure impact of optimization (Python or mypyc) on realistic
self-play runs.

Usage:
    uv run python scripts/bench_mcts_long.py [num_games] [num_sims] [hidden_dim] [num_blocks]
"""
from __future__ import annotations

import statistics
import sys
import time

import torch

from mzinga.core.board import Board
from mzinga.core.enums import GameType
from mzinga.rl.mcts import MCTS
from mzinga.rl.model import HivePolicyValue


def play_one_game(model: torch.nn.Module, mcts: MCTS, max_steps: int = 300) -> tuple[int, float]:
    board = Board(GameType.Base)
    moves_played = 0
    t0 = time.time()
    for _ in range(max_steps):
        if board.game_is_over:
            break
        valid_moves = board.get_valid_moves()
        if len(valid_moves) == 0:
            break
        pi, pi_probs, best_a = mcts.search(board)
        best_a = int(best_a)
        if best_a >= len(valid_moves):
            best_a = 0
        move = valid_moves[best_a]
        move_str = board.try_get_move_string(move) or ""
        board.trusted_play(move, move_str)
        moves_played += 1
    elapsed = time.time() - t0
    return moves_played, elapsed


def main() -> int:
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    hidden_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    num_blocks = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    device = torch.device("cpu")
    torch.set_num_threads(1)

    model = HivePolicyValue(obs_dim=88, hidden_dim=hidden_dim, num_blocks=num_blocks)
    model.to(device)
    model.eval()

    mcts = MCTS(model=model, num_simulations=num_sims)

    print(f"Benchmark: {num_games} games, {num_sims} sims/move, hidden={hidden_dim}, blocks={num_blocks}")
    print(f"Model params: {model.param_count():,}")
    print()

    per_game_moves = []
    per_game_sims_per_sec = []
    total_moves = 0
    total_sims = 0
    total_time = 0.0
    t0_all = time.time()

    for g in range(num_games):
        moves, dt = play_one_game(model, mcts)
        sims = moves * num_sims
        per_game_moves.append(moves)
        per_game_sims_per_sec.append(sims / max(dt, 1e-6))
        total_moves += moves
        total_sims += sims
        total_time += dt
        print(f"  game {g+1}: {moves:4d} moves in {dt:7.2f}s  ({sims/max(dt,1e-6):8.0f} sims/s)")

    wall = time.time() - t0_all

    print()
    print(f"Total:     {total_moves} moves, {total_sims} sims in {total_time:.2f}s (wall {wall:.2f}s)")
    print(f"Games/sec: {num_games/total_time:.3f}")
    print(f"Moves/sec: {total_moves/total_time:.1f}")
    print(f"Sims/sec:  {total_sims/total_time:.0f}")
    print(f"Avg game:  {total_moves/num_games:.1f} moves, {total_time/num_games:.2f}s")
    if num_games >= 3:
        print(f"Median sims/s: {statistics.median(per_game_sims_per_sec):.0f}")
        print(f"Stdev sims/s:  {statistics.stdev(per_game_sims_per_sec) if num_games > 1 else 0:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
