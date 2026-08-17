"""Game runner and players for the evaluation harness (NEXT_STEPS 3.4).

Provides:
- :class:`RandomPlayer` — uniform random baseline.
- :class:`FastPlayerAdapter` — wraps :class:`FastPlayer` for the game runner.
- :class:`MCTSPlayerAdapter` — wraps :class:`MCTSPlayer`.
- :class:`EvalConfig` — configures number of games, game type, etc.
- :class:`EvalResults` — collects outcomes, computes stats, renders markdown.
- :func:`run_eval` — play N games and return :class:`EvalResults`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from mzinga.core.board import Board
from mzinga.core.enums import BoardState, GameType, PlayerColor


# ---------------------------------------------------------------------------
# Player protocol
# ---------------------------------------------------------------------------

class BasePlayer:
    """Abstract player interface for the game runner."""
    name: str = "base"

    def play(self, board: Board):
        """Return a Move object that is legal on *board*."""
        raise NotImplementedError


class RandomPlayer(BasePlayer):
    """Uniform-random baseline player."""

    name = "random"

    def play(self, board: Board):
        moves = list(board.get_valid_moves())
        valid_moves = []
        for mv in moves:
            try:
                ms = board.get_move_string(mv)
                if ms:
                    valid_moves.append(mv)
            except Exception:
                pass
        if not valid_moves:
            from mzinga.core.move import PASS_MOVE
            return PASS_MOVE
        return random.choice(valid_moves)



class FastPlayerAdapter(BasePlayer):
    """Wrap :class:`FastPlayer` for the game runner."""

    name = "fast"

    def __init__(self, fast_player):
        self.fast = fast_player

    def play(self, board: Board):
        return self.fast.play(board)


class MCTSPlayerAdapter(BasePlayer):
    """Wrap :class:`MCTSPlayer` for the game runner."""

    name = "mcts"

    def __init__(self, mcts_player):
        self.mcts = mcts_player

    def play(self, board: Board):
        return self.mcts.search(board)


class ValuePlayerAdapter(BasePlayer):
    """Wrap a value-greedy player for the game runner."""

    name = "value"

    def __init__(self, value_player):
        self.value = value_player

    def play(self, board: Board):
        return self.value.play(board)


# ---------------------------------------------------------------------------
# Config and results
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""
    n_games: int = 50
    game_type: GameType = GameType.Base
    max_plies: int = 200
    swap_sides: bool = True       # alternate who goes first
    seed: Optional[int] = None


@dataclass
class EvalResults:
    """Collects game outcomes and computes statistics."""
    player_name: str = "player"
    opponent_name: str = "opponent"
    n_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    game_lengths: List[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.n_games)

    @property
    def loss_rate(self) -> float:
        return self.losses / max(1, self.n_games)

    @property
    def draw_rate(self) -> float:
        return self.draws / max(1, self.n_games)

    @property
    def mean_game_length(self) -> float:
        return float(np.mean(self.game_lengths)) if self.game_lengths else 0.0

    def _wilson_ci(self, wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        """Wilson score 95% confidence interval for the win rate."""
        if n == 0:
            return 0.0, 0.0
        p = wins / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, center - spread), min(1.0, center + spread)

    @property
    def win_rate_ci(self) -> Tuple[float, float]:
        """95% confidence interval for the win rate."""
        return self._wilson_ci(self.wins, self.n_games)

    def to_markdown(self) -> str:
        """Render the results as a markdown table."""
        ci_low, ci_high = self.win_rate_ci
        lines = [
            f"# Eval Results: {self.player_name} vs {self.opponent_name}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Games | {self.n_games} |",
            f"| Wins | {self.wins} ({self.win_rate:.1%}) |",
            f"| Losses | {self.losses} ({self.loss_rate:.1%}) |",
            f"| Draws | {self.draws} ({self.draw_rate:.1%}) |",
            f"| Win Rate 95% CI | [{ci_low:.1%}, {ci_high:.1%}] |",
            f"| Mean Game Length | {self.mean_game_length:.1f} plies |",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        ci_low, ci_high = self.win_rate_ci
        return {
            "n_games": self.n_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
            "win_rate_ci_low": ci_low,
            "win_rate_ci_high": ci_high,
            "mean_game_length": self.mean_game_length,
        }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _play_one_game(player: BasePlayer, opponent: BasePlayer,
                   game_type: GameType, max_plies: int = 200,
                   player_first: bool = True) -> Tuple[str, int]:
    """Play one game and return ``(outcome, n_plies)``.

    *outcome* is ``"win"`` if *player* wins, ``"loss"`` if *opponent*
    wins, ``"draw"`` otherwise.
    """
    board = Board(game_type)
    plies = 0

    for ply in range(max_plies):
        if board.game_is_over:
            break
        plies += 1
        is_player_turn = (board.current_color == PlayerColor.White) == player_first
        actor = player if is_player_turn else opponent
        mv = actor.play(board)
        try:
            ms = board.get_move_string(mv)
        except Exception:
            ms = board.try_get_move_string(mv)
            if ms is None:
                valid_moves = [m for m in board.get_valid_moves() if board.try_get_move_string(m) is not None]
                if valid_moves:
                    mv = valid_moves[0]
                    ms = board.get_move_string(mv)
                else:
                    from mzinga.core.move import PASS_MOVE
                    mv = PASS_MOVE
                    ms = "pass"
        board.trusted_play(mv, ms)


    state = board.board_state
    if state == BoardState.Draw or not board.game_is_over:
        return "draw", plies
    white_wins = (state == BoardState.WhiteWins)
    player_is_white = player_first
    if (white_wins and player_is_white) or (not white_wins and not player_is_white):
        return "win", plies
    return "loss", plies


def run_eval(player: BasePlayer, opponent: BasePlayer,
             config: EvalConfig) -> EvalResults:
    """Play N games and return :class:`EvalResults`.

    *player* plays White in odd-numbered games and Black in even-numbered
    games if ``config.swap_sides`` is True.
    """
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    results = EvalResults(
        player_name=player.name,
        opponent_name=opponent.name,
    )

    for i in range(config.n_games):
        player_first = (i % 2 == 0) if config.swap_sides else True
        outcome, plies = _play_one_game(
            player, opponent, config.game_type,
            max_plies=config.max_plies, player_first=player_first,
        )
        if outcome == "win":
            results.wins += 1
        elif outcome == "loss":
            results.losses += 1
        else:
            results.draws += 1
        results.n_games += 1
        results.game_lengths.append(plies)

    return results
