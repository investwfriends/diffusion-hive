"""Dataset pipeline for ``HiveDiffusionModel`` (Phase 9, NEXT_STEPS 1.3).

Generates supervised samples from Mzinga self-play (or loaded game
records). Each sample is a :class:`TrainingSample`:

- ``context_ids`` — the canonical text context up to the current ply.
- ``target_move_ids`` — the next move tokenized.
- ``legal_move_ids`` — every legal move at this ply, tokenized.
- ``target_legal_idx`` — the index of the target move in legal_moves.
- ``value`` — game outcome from the side-to-move's perspective, in [-1, 1].

Sources supported:

- :class:`SelfPlayGenerator` — random, model-guided, or mixed self-play.
- :class:`SelfPlayRollout` — orchestrator for batched generation across
  multiple game types and policy strengths.
- :class:`GameRecordDataset` — replay loaded game strings from Mzinga.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mzinga.core.board import Board
from mzinga.core.enums import GameType

from .context_builder import HiveContextBuilder
from .tokenizer import HiveTokenizer
from .training import TrainingSample


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------


def game_outcome_value(board: Board, side_to_move_color) -> float:
    """Return the value of the game for ``side_to_move_color`` in [-1, 1].

    ``side_to_move_color`` should be the color that was about to play
    when this board state was reached. The outcome is read from
    ``board.board_state``; if the game is still in progress, returns 0.
    """
    from mzinga.core.enums import BoardState, PlayerColor
    state = board.board_state
    if state == BoardState.Draw:
        return 0.0
    if state == BoardState.WhiteWins:
        return 1.0 if side_to_move_color == PlayerColor.White else -1.0
    if state == BoardState.BlackWins:
        return 1.0 if side_to_move_color == PlayerColor.Black else -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


class SelfPlayGenerator:
    """Generate samples by playing games with a pluggable move policy.

    The ``move_policy`` callable receives the board and returns one of the
    legal-move strings (or ``None`` for uniform random).
    """

    def __init__(self, tokenizer: HiveTokenizer, builder: HiveContextBuilder,
                 game_type: GameType = GameType.Base,
                 move_policy: Optional[callable] = None,
                 max_plies: int = 400):
        self.tokenizer = tokenizer
        self.builder = builder
        self.game_type = game_type
        self.move_policy = move_policy
        self.max_plies = max_plies

    def _random_policy(self, board: Board) -> str:
        moves = list(board.get_valid_moves())
        if not moves:
            return "pass"
        mv = random.choice(moves)
        try:
            return board.get_move_string(mv)
        except Exception:
            return "pass"

    def _play_one(self) -> Tuple[List[TrainingSample], Optional[str]]:
        """Play one game and produce one sample per ply.

        Returns the list of samples and the game's final move-string for
        debugging (``None`` if the game hit max_plies).
        """
        board = Board(self.game_type)
        samples: List[TrainingSample] = []
        policy = self.move_policy or self._random_policy
        last_move_str: Optional[str] = None

        from mzinga.core.enums import PlayerColor

        for ply in range(self.max_plies):
            if board.game_is_over:
                break

            side = board.current_color
            legal_strs = self.builder._legal_moves(board)
            if not legal_strs:
                # Forced pass should already be in legal_moves; bail.
                break

            chosen_str = policy(board)
            if chosen_str is None or chosen_str not in legal_strs:
                chosen_str = random.choice(legal_strs)
            last_move_str = chosen_str

            # Use teacher value estimate if available (e.g. MCTS during gen_data).
            teacher_val = getattr(policy, 'last_value', None)
            if teacher_val is not None:
                midgame_value = float(teacher_val)
            else:
                midgame_value = 0.0

            # Build a sample *before* playing the move.
            ctx_ids = self.builder.encode(board, target_move=None)
            legal_ids = [self.tokenizer.encode_move(s) for s in legal_strs]
            target_ids = self.tokenizer.encode_move(chosen_str)
            try:
                target_idx = legal_strs.index(chosen_str)
            except ValueError:
                target_idx = 0

            samples.append(TrainingSample(
                context_ids=torch.tensor(ctx_ids, dtype=torch.long),
                target_move_ids=torch.tensor(target_ids, dtype=torch.long),
                legal_move_ids=legal_ids,
                target_legal_idx=target_idx,
                value=midgame_value,
                timestep=None,
            ))

            # Find and play the move.
            real_mv = None
            for mv in board.get_valid_moves():
                try:
                    if board.get_move_string(mv) == chosen_str:
                        real_mv = mv
                        break
                except Exception:
                    continue
            if real_mv is None:
                break
            board.trusted_play(real_mv, chosen_str)

        # Override with ground-truth outcome only for games that concluded.
        if board.game_is_over:
            final_outcome = game_outcome_value(board, PlayerColor.White)
            for sample, ply_side in zip(samples, _iter_ply_sides(self.game_type, len(samples))):
                sample.value = _sign_for_side(final_outcome, ply_side)

        return samples, last_move_str

    def generate(self, n_games: int = 1) -> List[TrainingSample]:
        all_samples: List[TrainingSample] = []
        for _ in range(n_games):
            samples, _ = self._play_one()
            all_samples.extend(samples)
        return all_samples

    def iter_batches(self, batch_size: int, n_games: int = 1
                     ) -> Iterator[List[TrainingSample]]:
        """Iterate over fixed-size batches of samples."""
        buffer: List[TrainingSample] = []
        for _ in range(n_games):
            samples, _ = self._play_one()
            buffer.extend(samples)
            while len(buffer) >= batch_size:
                yield buffer[:batch_size]
                buffer = buffer[batch_size:]
        if buffer:
            yield buffer


# ---------------------------------------------------------------------------
# Game-type stratification helpers (NEXT_STEPS 1.3)
# ---------------------------------------------------------------------------

# Maps GameType value -> (token_name, has_mosquito, has_ladybug, has_pillbug)
_GAME_TYPE_FLAGS = {
    0: ("Base",        False, False, False),
    1: ("Base+M",      True,  False, False),
    2: ("Base+L",      False, True,  False),
    3: ("Base+P",      False, False, True),
    4: ("Base+ML",     True,  True,  False),
    5: ("Base+MP",     True,  False, True),
    6: ("Base+LP",     False, True,  True),
    7: ("Base+MLP",    True,  True,  True),
}


def _all_game_types() -> List[GameType]:
    """Return all valid game types (skipping INVALID/NumGameTypes)."""
    return [gt for gt in GameType if gt not in (GameType.INVALID, GameType.NumGameTypes)]


def _filter_early_game(samples: List[TrainingSample],
                       max_plies: int = 20) -> List[TrainingSample]:
    """Keep only early-game samples for diversity."""
    return [s for i, s in enumerate(samples) if i < max_plies]


# ---------------------------------------------------------------------------
# Policy adapters (NEXT_STEPS 1.3)
# ---------------------------------------------------------------------------


def make_random_policy():
    """Uniform random move policy."""
    def _policy(board: Board) -> Optional[str]:
        moves = list(board.get_valid_moves())
        if not moves:
            return "pass"
        mv = random.choice(moves)
        try:
            return board.get_move_string(mv)
        except Exception:
            return "pass"
    return _policy


def make_model_policy(model=None, tokenizer=None, builder=None,
                      deterministic: bool = True):
    """Create a policy that uses a FastPlayer for move selection.

    Returns a callable suitable for ``SelfPlayGenerator.move_policy``.
    If *model* is None, falls back to the random policy.
    """
    if model is None:
        return make_random_policy()

    from .inference import FastPlayer
    player = FastPlayer(model, tokenizer, builder, deterministic=deterministic)

    def _policy(board: Board) -> Optional[str]:
        try:
            mv = player.play(board, deterministic=deterministic)
            return board.get_move_string(mv)
        except Exception:
            return None
    return _policy


def make_mixed_policy(policies: List, weights: Optional[List[float]] = None):
    """Mix multiple policies, sampling one per move.

    *policies* is a list of callables.  *weights* (optional) gives the
    probability of selecting each policy; defaults to uniform.
    """
    if weights is None:
        weights = [1.0 / len(policies)] * len(policies)
    total = sum(weights)
    weights = [w / total for w in weights]

    def _policy(board: Board) -> Optional[str]:
        pol = random.choices(policies, weights=weights, k=1)[0]
        return pol(board)
    return _policy


# ---------------------------------------------------------------------------
# SelfPlayRollout — orchestrator (NEXT_STEPS 1.3)
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    """Configuration for a self-play rollout session."""
    n_games: int = 100
    game_types: Optional[List[GameType]] = None   # None = all
    max_plies: int = 400
    filter_early_game: bool = False
    early_game_plies: int = 20
    policy_mixture: Optional[callable] = None     # None = random
    seed: Optional[int] = None


class SelfPlayRollout:
    """Orchestrate self-play across game types and policy strengths.

    Generates a dataset by playing N games, stratified across the
    specified game types.  Returns a flat list of
    :class:`TrainingSample` ready for training.

    Example::

        rollout = SelfPlayRollout(tokenizer, builder, RolloutConfig(
            n_games=1000,
            game_types=[GameType.Base, GameType.BaseMLP],
        ))
        samples = rollout.generate()
    """

    def __init__(self, tokenizer: HiveTokenizer, builder: HiveContextBuilder,
                 config: RolloutConfig):
        self.tokenizer = tokenizer
        self.builder = builder
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            torch.manual_seed(config.seed)

    def generate(self) -> List[TrainingSample]:
        """Run all games and return a flat list of samples."""
        cfg = self.config
        game_types = cfg.game_types or _all_game_types()
        all_samples: List[TrainingSample] = []

        for gt in game_types:
            gen = SelfPlayGenerator(
                self.tokenizer, self.builder,
                game_type=gt,
                move_policy=cfg.policy_mixture,
                max_plies=cfg.max_plies,
            )
            for _ in range(cfg.n_games):
                samples, _ = gen._play_one()
                if cfg.filter_early_game:
                    samples = _filter_early_game(
                        samples, max_plies=cfg.early_game_plies)
                all_samples.extend(samples)

        random.shuffle(all_samples)
        return all_samples

    def generate_by_game_type(self) -> Dict[str, List[TrainingSample]]:
        """Like :meth:`generate` but returns samples keyed by game type."""
        cfg = self.config
        game_types = cfg.game_types or _all_game_types()
        result: Dict[str, List[TrainingSample]] = {}

        for gt in game_types:
            gt_name = _GAME_TYPE_FLAGS.get(gt.value, ("Unknown",))[0]
            gen = SelfPlayGenerator(
                self.tokenizer, self.builder,
                game_type=gt,
                move_policy=cfg.policy_mixture,
                max_plies=cfg.max_plies,
            )
            gt_samples: List[TrainingSample] = []
            for _ in range(cfg.n_games):
                samples, _ = gen._play_one()
                if cfg.filter_early_game:
                    samples = _filter_early_game(
                        samples, max_plies=cfg.early_game_plies)
                gt_samples.extend(samples)
            result[gt_name] = gt_samples

        return result


def _iter_ply_sides(game_type: GameType, n_plies: int) -> Iterable:
    """Yield the side-to-move color for each ply in a standard game."""
    from mzinga.core.enums import PlayerColor
    return (PlayerColor.White if i % 2 == 0 else PlayerColor.Black
            for i in range(n_plies))


def _sign_for_side(white_value: float, side) -> float:
    from mzinga.core.enums import PlayerColor
    if side == PlayerColor.White:
        return white_value
    return -white_value


# ---------------------------------------------------------------------------
# Dataset from loaded game strings
# ---------------------------------------------------------------------------


class GameRecordDataset:
    """Wrap pre-collected game strings (e.g. expert records or saved replays).

    Each game is a Mzinga game string (e.g. ``"Base;InProgress;White[1];wB1;bB1 wB1/;wQ"``).
    """

    def __init__(self, tokenizer: HiveTokenizer, builder: HiveContextBuilder):
        self.tokenizer = tokenizer
        self.builder = builder
        self._samples: List[TrainingSample] = []

    def load_game(self, game_string: str) -> List[TrainingSample]:
        """Replay a game and produce one :class:`TrainingSample` per ply."""
        board = Board.try_parse_game_string(game_string)
        if board is None:
            raise ValueError(f"Could not parse game string: {game_string!r}")

        from mzinga.core.enums import PlayerColor
        samples: List[TrainingSample] = []
        # Replay by undoing then replaying forward to recover each ply.
        # Mzinga's board is already at the end state. Use the in-memory
        # `board_history` to enumerate moves in order.
        # Build per-ply positions by replaying.
        replay = Board(board.game_type)
        final_outcome = game_outcome_value(board, PlayerColor.White)

        history = list(board.board_history)
        for ply, item in enumerate(history):
            side = replay.current_color
            legal_strs = self.builder._legal_moves(replay)
            target_str = item.move_string or ""
            if not legal_strs:
                # Forced pass case.
                legal_strs = ["pass"]
                target_str = "pass"
            if target_str not in legal_strs:
                # Skip samples whose target isn't currently legal (can
                # happen after undo).
                replay.trusted_play(item.move, target_str)
                continue
            ctx_ids = self.builder.encode(replay, target_move=None)
            legal_ids = [self.tokenizer.encode_move(s) for s in legal_strs]
            target_ids = self.tokenizer.encode_move(target_str)
            target_idx = legal_strs.index(target_str)
            value = _sign_for_side(final_outcome, side)
            samples.append(TrainingSample(
                context_ids=torch.tensor(ctx_ids, dtype=torch.long),
                target_move_ids=torch.tensor(target_ids, dtype=torch.long),
                legal_move_ids=legal_ids,
                target_legal_idx=target_idx,
                value=value,
            ))
            replay.trusted_play(item.move, target_str)
        self._samples.extend(samples)
        return samples

    def load_games(self, game_strings: Sequence[str]) -> List[TrainingSample]:
        all_samples: List[TrainingSample] = []
        for gs in game_strings:
            try:
                all_samples.extend(self.load_game(gs))
            except Exception:
                continue
        return all_samples

    @property
    def samples(self) -> List[TrainingSample]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[TrainingSample]:
        return iter(self._samples)