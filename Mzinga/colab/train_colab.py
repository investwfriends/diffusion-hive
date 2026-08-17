"""AlphaZero training for Hive — self-contained Colab edition.

COLAB INSTRUCTIONS (paste each block as a separate cell):

─── Cell 1: Setup ───
    !pip install wandb -q
    !pip install torch --quiet

─── Cell 2: Upload & unzip ───
    from google.colab import files
    import os
    if not os.path.exists("train_colab.py"):
        _ = files.upload()          # upload mzinga_colab.zip
        !unzip -o mzinga_colab.zip -d mzinga_colab
        %cd mzinga_colab

─── Cell 3: Mount Drive (for checkpoints) ───
    from google.colab import drive
    drive.mount("/content/drive")
    import os; os.makedirs("/content/drive/MyDrive/mzinga_checkpoints", exist_ok=True)

─── Cell 4: W&B login ───
    import wandb; wandb.login()

─── Cell 5: Run training ───
    !python train_colab.py \
        --n_iterations 500 \
        --hidden_dim 512 \
        --num_blocks 6 \
        --num_sims 100 \
        --checkpoint_dir /content/drive/MyDrive/mzinga_checkpoints

─── Cell 6 (optional): Resume from checkpoint ───
    !python train_colab.py \
        --resume /content/drive/MyDrive/mzinga_checkpoints/checkpoint_0100.pt \
        --n_iterations 500 \
        --hidden_dim 512 \
        --num_blocks 6 \
        --num_sims 100 \
        --checkpoint_dir /content/drive/MyDrive/mzymga_checkpoints

NOTE ON PERFORMANCE:
This file inlines the engine from the mzinga_colab.zip snapshot. The
optimized engine (packed positions, flat grid, fast zobrist init) lives
in the main repo at src/mzinga/core/ and is NOT included here. For
maximum self-play throughput on Colab, prefer cloning the repo and
using tests/train_alphazero.py (or colab/train_colab.py updated from
the latest main) with the mypyc compilation step:
    uv run python scripts/build_mypyc.py
Measured speedups: perft ~2.6x, MCTS self-play ~1.15x.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

# ======================================================================
#  INLINED: mzinga/core/enums.py
# ======================================================================


class PlayerColor(IntEnum):
    White = 0
    Black = 1
    NumPlayerColors = 2


class BoardState(IntEnum):
    NotStarted = 0
    InProgress = 1
    Draw = 2
    WhiteWins = 3
    BlackWins = 4


class PieceName(IntEnum):
    INVALID = -1
    wQ = 0
    wS1 = 1
    wS2 = 2
    wB1 = 3
    wB2 = 4
    wG1 = 5
    wG2 = 6
    wG3 = 7
    wA1 = 8
    wA2 = 9
    wA3 = 10
    wM = 11
    wL = 12
    wP = 13
    bQ = 14
    bS1 = 15
    bS2 = 16
    bB1 = 17
    bB2 = 18
    bG1 = 19
    bG2 = 20
    bG3 = 21
    bA1 = 22
    bA2 = 23
    bA3 = 24
    bM = 25
    bL = 26
    bP = 27
    NumPieceNames = 28


class Direction(IntEnum):
    Up = 0
    UpRight = 1
    DownRight = 2
    Down = 3
    DownLeft = 4
    UpLeft = 5
    NumDirections = 6
    Above = 6


class BugType(IntEnum):
    INVALID = -1
    QueenBee = 0
    Spider = 1
    Beetle = 2
    Grasshopper = 3
    SoldierAnt = 4
    Mosquito = 5
    Ladybug = 6
    Pillbug = 7
    NumBugTypes = 8


class GameType(IntEnum):
    INVALID = -1
    Base = 0
    BaseM = 1
    BaseL = 2
    BaseP = 3
    BaseML = 4
    BaseMP = 5
    BaseLP = 6
    BaseMLP = 7
    NumGameTypes = 8


_NUM_FLAT_DIRECTIONS = 6

_piece_name_is_enabled_for_game_type = [
    0b1111111111100011111111111000,
    0b1111111111110011111111111100,
    0b1111111111101011111111111010,
    0b1111111111100111111111111001,
    0b1111111111111011111111111110,
    0b1111111111110111111111111101,
    0b1111111111101111111111111011,
    0b1111111111111111111111111111,
]


def _game_in_progress(board_state):
    return board_state == BoardState.NotStarted or board_state == BoardState.InProgress


def _game_is_over(board_state):
    return board_state in (BoardState.WhiteWins, BoardState.BlackWins, BoardState.Draw)


def _get_color(piece_name):
    if piece_name.value < 14:
        return PlayerColor.White
    return PlayerColor.Black


def _get_bug_type(piece_name):
    v = piece_name.value % 14
    if v == 0:
        return BugType.QueenBee
    if v in (1, 2):
        return BugType.Spider
    if v in (3, 4):
        return BugType.Beetle
    if v in (5, 6, 7):
        return BugType.Grasshopper
    if v in (8, 9, 10):
        return BugType.SoldierAnt
    if v == 11:
        return BugType.Mosquito
    if v == 12:
        return BugType.Ladybug
    if v == 13:
        return BugType.Pillbug
    return BugType.INVALID


def _piece_name_is_enabled(piece_name, game_type):
    if piece_name == PieceName.INVALID or game_type == GameType.INVALID:
        return False
    return (
        (0b1000000000000000000000000000 >> piece_name.value)
        & _piece_name_is_enabled_for_game_type[game_type.value]
    ) != 0


def _left_of(direction):
    return Direction((direction.value + _NUM_FLAT_DIRECTIONS - 1) % _NUM_FLAT_DIRECTIONS)


def _right_of(direction):
    return Direction((direction.value + 1) % _NUM_FLAT_DIRECTIONS)


def _bug_type_is_enabled(bug_type, game_type):
    if bug_type == BugType.INVALID:
        return False
    if bug_type == BugType.Mosquito:
        return game_type in (GameType.BaseM, GameType.BaseML, GameType.BaseMP, GameType.BaseMLP)
    if bug_type == BugType.Ladybug:
        return game_type in (GameType.BaseL, GameType.BaseML, GameType.BaseLP, GameType.BaseMLP)
    if bug_type == BugType.Pillbug:
        return game_type in (GameType.BaseP, GameType.BaseMP, GameType.BaseLP, GameType.BaseMLP)
    return True


def _try_parse_game_type(s):
    mapping = {
        "Base": GameType.Base,
        "Base+M": GameType.BaseM,
        "Base+L": GameType.BaseL,
        "Base+P": GameType.BaseP,
        "Base+ML": GameType.BaseML,
        "Base+MP": GameType.BaseMP,
        "Base+LP": GameType.BaseLP,
        "Base+MLP": GameType.BaseMLP,
    }
    return mapping.get(s)


def _get_game_type_string(game_type):
    mapping = {
        GameType.Base: "Base",
        GameType.BaseM: "Base+M",
        GameType.BaseL: "Base+L",
        GameType.BaseP: "Base+P",
        GameType.BaseML: "Base+ML",
        GameType.BaseMP: "Base+MP",
        GameType.BaseLP: "Base+LP",
        GameType.BaseMLP: "Base+MLP",
    }
    return mapping.get(game_type, "")


def _placing_piece_in_order(piece_name, piece_positions):
    if piece_positions[int(piece_name)].stack < 0:
        if piece_name in (
            PieceName.wS2, PieceName.wB2, PieceName.wG2, PieceName.wG3,
            PieceName.wA2, PieceName.wA3, PieceName.bS2, PieceName.bB2,
            PieceName.bG2, PieceName.bG3, PieceName.bA2, PieceName.bA3,
        ):
            return piece_positions[int(piece_name) - 1].stack >= 0
    return True


# ======================================================================
#  INLINED: mzinga/core/position.py
# ======================================================================

NEIGHBOR_DELTAS = [
    (0, -1, 0),
    (1, -1, 0),
    (1, 0, 0),
    (0, 1, 0),
    (-1, 1, 0),
    (-1, 0, 0),
    (0, 0, 1),
]
BOARD_SIZE = 128
BOARD_STACK_SIZE = 8
HALF_BOARD = BOARD_SIZE // 2


@dataclass(frozen=True, order=True)
class Position:
    q: int
    r: int
    stack: int

    def get_neighbor_at(self, direction):
        delta = NEIGHBOR_DELTAS[direction.value]
        return Position(self.q + delta[0], self.r + delta[1], self.stack)

    def get_above(self):
        return Position(self.q, self.r, self.stack + 1)

    def get_below(self):
        return Position(self.q, self.r, self.stack - 1)

    def get_bottom(self):
        if self.stack == 0:
            return self
        return Position(self.q, self.r, 0)


ORIGIN_POSITION = Position(0, 0, 0)
NULL_POSITION = Position(0, 0, -1)
ORIGIN_NEIGHBORS = [
    Position(0, -1, 0),
    Position(1, -1, 0),
    Position(1, 0, 0),
    Position(0, 1, 0),
    Position(-1, 1, 0),
    Position(-1, 0, 0),
]


# ======================================================================
#  INLINED: mzinga/core/move.py
# ======================================================================

PASS_STRING = "pass"
_DIRECTION_SEPARATORS = frozenset(["/", "\\", "-"])


@dataclass(frozen=True)
class Move:
    piece_name: PieceName
    source: Position
    destination: Position

    @staticmethod
    def _try_parse_piece_name(s):
        for member in PieceName:
            if member.name == s:
                return member
        s_lower = s.lower()
        for member in PieceName:
            if member.name.lower() == s_lower:
                return member
        return PieceName.INVALID

    @staticmethod
    def build_move_string(is_pass, start_piece, before_sep, end_piece, after_sep):
        if is_pass:
            return PASS_STRING
        result = start_piece.name
        if end_piece != PieceName.INVALID:
            result += " "
            if before_sep:
                result += f"{before_sep}{end_piece.name}"
            elif after_sep:
                result += f"{end_piece.name}{after_sep}"
            else:
                result += end_piece.name
        return result

    @staticmethod
    def try_normalize_move_string(move_string):
        piece1_chars = []
        piece2_chars = []
        before_separator = ""
        after_separator = ""
        items_found = 0
        i = 0
        while i < len(move_string):
            ch = move_string[i]
            if items_found == 0 and ch != " ":
                piece1_chars.append(ch)
                items_found = 1
            elif items_found == 1:
                if ch != " ":
                    piece1_chars.append(ch)
                else:
                    items_found = 2
            elif items_found == 2:
                if ch != " ":
                    if ch in _DIRECTION_SEPARATORS:
                        before_separator = ch
                    else:
                        piece2_chars.append(ch)
                    items_found = 3
            elif items_found == 3:
                if ch != " ":
                    if ch in _DIRECTION_SEPARATORS:
                        after_separator = ch
                        break
                    else:
                        piece2_chars.append(ch)
                else:
                    break
            i += 1
        piece1_str = "".join(piece1_chars)
        if piece1_str.lower() == PASS_STRING:
            return (True, PieceName.INVALID, "", PieceName.INVALID, "")
        start_piece = Move._try_parse_piece_name(piece1_str)
        if start_piece == PieceName.INVALID:
            return None
        piece2_str = "".join(piece2_chars)
        if not piece2_str and not before_separator and not after_separator:
            return (False, start_piece, "", PieceName.INVALID, "")
        if piece2_str:
            end_piece = Move._try_parse_piece_name(piece2_str)
            if end_piece != PieceName.INVALID:
                return (False, start_piece, before_separator, end_piece, after_separator)
        return None


PASS_MOVE = Move(PieceName.INVALID, NULL_POSITION, NULL_POSITION)


# ======================================================================
#  INLINED: mzinga/core/fast_set.py
# ======================================================================


class FastSet:
    def __init__(self, items=None):
        self._items = list(items) if items else []

    @property
    def count(self):
        return len(self._items)

    def contains(self, item):
        for i in range(len(self._items) - 1, -1, -1):
            if self._items[i] == item:
                return True
        return False

    def add(self, item):
        if self.contains(item):
            return False
        self._items.append(item)
        return True

    def fast_add(self, item):
        self._items.append(item)

    def clear(self):
        self._items.clear()

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class MoveSet(FastSet):
    def contains_piece_name(self, piece_name):
        for item in self._items:
            if item.piece_name == piece_name:
                return True
        return False


class PositionSet(set):
    pass


# ======================================================================
#  INLINED: mzinga/core/zobrist.py
# ======================================================================


class _ZobristHash:
    _next = 1
    _hash_part_by_turn_color = 0
    _hash_part_by_last_moved_piece = []
    _hash_part_by_position = []

    @staticmethod
    def _rand64():
        _ZobristHash._next = (_ZobristHash._next * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        return _ZobristHash._next

    @staticmethod
    def _init_statics():
        if _ZobristHash._hash_part_by_position:
            return
        _ZobristHash._next = 1
        _ZobristHash._hash_part_by_turn_color = _ZobristHash._rand64()
        _ZobristHash._hash_part_by_last_moved_piece = [
            _ZobristHash._rand64() for _ in range(int(PieceName.NumPieceNames))
        ]
        num_pn = int(PieceName.NumPieceNames)
        bs = BOARD_SIZE
        bss = BOARD_STACK_SIZE + 1
        _ZobristHash._hash_part_by_position = [
            [
                [
                    [_ZobristHash._rand64() for _ in range(bss)]
                    for _ in range(bs)
                ]
                for _ in range(bs)
            ]
            for _ in range(num_pn)
        ]

    def __init__(self):
        self._init_statics()
        self.value = 0

    def toggle_piece(self, piece_name, position):
        self.value ^= self._hash_part_by_position[
            int(piece_name)
        ][HALF_BOARD + position.q][HALF_BOARD + position.r][position.stack + 1]

    def toggle_last_moved_piece(self, piece_name):
        if piece_name != PieceName.INVALID:
            self.value ^= self._hash_part_by_last_moved_piece[int(piece_name)]

    def toggle_turn(self):
        self.value ^= self._hash_part_by_turn_color


# ======================================================================
#  INLINED: mzinga/core/board_history.py
# ======================================================================


@dataclass(frozen=True)
class _BoardHistoryItem:
    move: Move
    move_string: str


class _BoardHistory:
    def __init__(self):
        self._items = []

    @property
    def count(self):
        return len(self._items)

    @property
    def last_move(self):
        if self._items:
            return self._items[-1].move
        return None

    def add(self, move, move_str):
        self._items.append(_BoardHistoryItem(move, move_str))

    def undo_last(self):
        if self._items:
            self._items.pop()

    def __getitem__(self, index):
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


# ======================================================================
#  INLINED: mzinga/core/board.py (game engine)
# ======================================================================


class InvalidMoveError(Exception):
    pass


class Board:
    def __init__(self, game_type=GameType.Base):
        self.game_type = game_type
        self.board_state = BoardState.NotStarted
        self._current_turn = 0
        self._last_piece_moved = PieceName.INVALID
        self._piece_positions = [NULL_POSITION] * int(PieceName.NumPieceNames)
        self._piece_grid = [
            [[PieceName.INVALID for _ in range(BOARD_STACK_SIZE)] for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]
        self._cached_valid_placements_ready = False
        self._cached_valid_placements = PositionSet()
        self._cached_valid_moves = None
        self._cached_enemy_queen_neighbors = None
        self._part_of_hive = [False] * int(PieceName.NumPieceNames)
        self._pieces_to_look_at = deque()
        self._zobrist_hash = _ZobristHash()
        self.board_history = _BoardHistory()

    @property
    def game_in_progress(self):
        return _game_in_progress(self.board_state)

    @property
    def game_is_over(self):
        return _game_is_over(self.board_state)

    @property
    def current_turn(self):
        return self._current_turn

    @current_turn.setter
    def current_turn(self, value):
        if value < 0:
            raise ValueError("Current turn must be >= 0")
        old_color = self.current_color
        self._current_turn = value
        if old_color != self.current_color:
            self._zobrist_hash.toggle_turn()
        self._reset_state()
        self._reset_caches()

    @property
    def current_player_turn(self):
        return 1 + self._current_turn // 2

    @property
    def current_color(self):
        return PlayerColor(self._current_turn % int(PlayerColor.NumPlayerColors))

    @property
    def current_turn_queen_in_play(self):
        queen = PieceName.wQ if self.current_color == PlayerColor.White else PieceName.bQ
        return self.piece_in_play(queen)

    @property
    def last_piece_moved(self):
        return self._last_piece_moved

    @last_piece_moved.setter
    def last_piece_moved(self, value):
        old = self._last_piece_moved
        self._last_piece_moved = value
        if old != value:
            self._zobrist_hash.toggle_last_moved_piece(old)
            self._zobrist_hash.toggle_last_moved_piece(value)

    @property
    def zobrist_key(self):
        return self._zobrist_hash.value

    @staticmethod
    def parse_game_string(game_str, trusted_play=False):
        split = game_str.split(";")
        game_type = _try_parse_game_type(split[0])
        if game_type is None:
            raise ValueError(f"Unable to parse game type")
        board = Board(game_type)
        for i in range(3, len(split)):
            move, parsed = board.parse_move(split[i])
            if move is None:
                raise ValueError(f"Unable to parse move")
            if trusted_play:
                board.trusted_play(move, parsed)
            else:
                board.try_play_move(move, parsed)
        return board

    def get_game_string(self):
        parts = [f"{_get_game_type_string(self.game_type)};{self.board_state};{self.current_color.name}[{self.current_player_turn}]"]
        for item in self.board_history:
            parts.append(f";{item.move_string}")
        return "".join(parts)

    def get_valid_moves(self):
        if self._cached_valid_moves is None:
            self._cached_valid_moves = MoveSet()
            if self.game_in_progress:
                start_piece = int(PieceName.wQ if self.current_color == PlayerColor.White else PieceName.bQ)
                end_piece = int(PieceName.bQ if self.current_color == PlayerColor.White else PieceName.NumPieceNames)
                for pn in range(start_piece, end_piece):
                    self._get_valid_moves_for_piece(PieceName(pn), self._cached_valid_moves)
                if len(self._cached_valid_moves) == 0:
                    self._cached_valid_moves.fast_add(PASS_MOVE)
        return self._cached_valid_moves

    def play(self, move, move_string=""):
        if move == PASS_MOVE:
            self.pass_move()
            return
        if self.game_is_over:
            raise InvalidMoveError("Game is over")
        valid_moves = self.get_valid_moves()
        if not valid_moves.contains(move):
            raise InvalidMoveError("Invalid move")
        self.trusted_play(move, move_string)

    def pass_move(self):
        if self.game_is_over:
            raise InvalidMoveError("Game is over")
        if not self.get_valid_moves().contains(PASS_MOVE):
            raise InvalidMoveError("Can't pass with valid moves")
        self.trusted_play(PASS_MOVE, PASS_STRING)

    def try_play_move(self, move, move_string=""):
        valid_moves = self.get_valid_moves()
        if valid_moves.contains(move):
            self.trusted_play(move, move_string)
            return True
        return False

    def try_undo_last_move(self):
        if self.board_history.count > 0:
            last_item = self.board_history[-1]
            if last_item.move != PASS_MOVE:
                self._set_position(last_item.move.piece_name, last_item.move.source, True)
            self.board_history.undo_last()
            self.last_piece_moved = (
                self.board_history.last_move.piece_name
                if self.board_history.last_move is not None
                else PieceName.INVALID
            )
            self.current_turn -= 1
            return True
        return False

    def get_move_string(self, move):
        if move == PASS_MOVE:
            return PASS_STRING
        start_piece = move.piece_name.name
        if self._current_turn == 0 and move.destination == ORIGIN_POSITION:
            return start_piece
        end_piece = ""
        if move.destination.stack > 0:
            piece_below = self._get_piece_at(move.destination.get_below())
            end_piece = piece_below.name
        else:
            self._set_position(move.piece_name, NULL_POSITION, False)
            for d in range(int(Direction.NumDirections)):
                neighbor_position = move.destination.get_neighbor_at(Direction(d))
                neighbor = self.get_piece_on_top_at(neighbor_position)
                if neighbor != PieceName.INVALID:
                    end_piece = neighbor.name
                    if d == 0:
                        end_piece += "\\"
                    elif d == 1:
                        end_piece = "/" + end_piece
                    elif d == 2:
                        end_piece = "-" + end_piece
                    elif d == 3:
                        end_piece = "\\" + end_piece
                    elif d == 4:
                        end_piece += "/"
                    elif d == 5:
                        end_piece += "-"
                    break
            self._set_position(move.piece_name, move.source, False)
        if end_piece:
            return f"{start_piece} {end_piece}"
        raise ValueError("Invalid move")

    def try_get_move_string(self, move):
        try:
            return self.get_move_string(move)
        except Exception:
            return None

    def parse_move(self, move_string):
        result = Move.try_normalize_move_string(move_string)
        if result is not None:
            is_pass, start_piece, before_separator, end_piece, after_separator = result
            result_string = Move.build_move_string(is_pass, start_piece, before_separator, end_piece, after_separator)
            if is_pass:
                return PASS_MOVE, result_string
            source = self._piece_positions[int(start_piece)]
            destination = ORIGIN_POSITION
            if end_piece != PieceName.INVALID:
                target_position = self._piece_positions[int(end_piece)]
                if before_separator:
                    if before_separator == "-":
                        destination = target_position.get_neighbor_at(Direction.UpLeft).get_bottom()
                    elif before_separator == "/":
                        destination = target_position.get_neighbor_at(Direction.DownLeft).get_bottom()
                    elif before_separator == "\\":
                        destination = target_position.get_neighbor_at(Direction.Up).get_bottom()
                elif after_separator:
                    if after_separator == "-":
                        destination = target_position.get_neighbor_at(Direction.DownRight).get_bottom()
                    elif after_separator == "/":
                        destination = target_position.get_neighbor_at(Direction.UpRight).get_bottom()
                    elif after_separator == "\\":
                        destination = target_position.get_neighbor_at(Direction.Down).get_bottom()
                else:
                    destination = target_position.get_above()
                if target_position.stack < 0:
                    destination = target_position
            return Move(start_piece, source, destination), result_string
        return None, None

    def is_noisy_move(self, move):
        if move == PASS_MOVE:
            return False
        if self._cached_enemy_queen_neighbors is None:
            self._cached_enemy_queen_neighbors = PositionSet()
            enemy_queen = PieceName.bQ if self.current_color == PlayerColor.White else PieceName.wQ
            eq_pos = self.get_position(enemy_queen)
            if eq_pos != NULL_POSITION:
                for d in range(int(Direction.NumDirections)):
                    self._cached_enemy_queen_neighbors.add(eq_pos.get_neighbor_at(Direction(d)))
        return (
            move.destination in self._cached_enemy_queen_neighbors
            and self.get_position(move.piece_name) not in self._cached_enemy_queen_neighbors
        )

    def clone(self):
        board = Board(self.game_type)
        for item in self.board_history:
            board.trusted_play(item.move, item.move_string)
        return board

    def get_position(self, piece_name):
        return self._piece_positions[int(piece_name)]

    def piece_in_hand(self, piece_name):
        return self._piece_positions[int(piece_name)].stack < 0

    def piece_in_play(self, piece_name):
        return self._piece_positions[int(piece_name)].stack >= 0

    def can_move_without_breaking_hive(self, piece_name):
        piece_index = int(piece_name)
        if self._piece_positions[piece_index].stack == 0:
            gaps = 0
            last_has_piece = None
            for d in range(int(Direction.NumDirections)):
                has_piece = self._has_piece_at_dir(self._piece_positions[piece_index], Direction(d))
                if last_has_piece is not None and last_has_piece != has_piece:
                    gaps += 1
                    if gaps > 2:
                        break
                last_has_piece = has_piece
            if gaps <= 2:
                return True
            starting_position = self._piece_positions[piece_index]
            self._set_position(piece_name, NULL_POSITION, False)
            is_one = self.is_one_hive()
            self._set_position(piece_name, starting_position, False)
            return is_one
        return True

    def is_one_hive(self):
        pieces_visited = 0
        starting_piece = PieceName.INVALID
        for pn in range(int(PieceName.NumPieceNames)):
            pnn = PieceName(pn)
            if self.piece_in_hand(pnn):
                self._part_of_hive[pn] = True
                pieces_visited += 1
            else:
                self._part_of_hive[pn] = False
                if starting_piece == PieceName.INVALID and self._piece_positions[pn].stack == 0:
                    starting_piece = pnn
                    self._part_of_hive[pn] = True
                    pieces_visited += 1
        if starting_piece != PieceName.INVALID and pieces_visited < int(PieceName.NumPieceNames):
            self._pieces_to_look_at.clear()
            self._pieces_to_look_at.append(starting_piece)
            while self._pieces_to_look_at:
                current_piece = self._pieces_to_look_at.popleft()
                current_position = self._piece_positions[int(current_piece)]
                for d in range(int(Direction.NumDirections)):
                    neighbor_piece = self._get_piece_at_dir(current_position, Direction(d))
                    if neighbor_piece != PieceName.INVALID and not self._part_of_hive[int(neighbor_piece)]:
                        self._pieces_to_look_at.append(neighbor_piece)
                        self._part_of_hive[int(neighbor_piece)] = True
                        pieces_visited += 1
                piece_above = self._get_piece_at_dir(current_position, Direction.Above)
                while piece_above != PieceName.INVALID:
                    self._part_of_hive[int(piece_above)] = True
                    pieces_visited += 1
                    piece_above = self._get_piece_at_dir(
                        self._piece_positions[int(piece_above)], Direction.Above
                    )
        return pieces_visited == int(PieceName.NumPieceNames)

    def trusted_play(self, move, move_str=""):
        self.board_history.add(move, move_str)
        if move != PASS_MOVE:
            self._set_position(move.piece_name, move.destination, True)
        self.current_turn += 1
        self.last_piece_moved = move.piece_name

    def _get_valid_moves_for_piece(self, piece_name, move_set):
        if not _piece_name_is_enabled(piece_name, self.game_type):
            return
        if not _placing_piece_in_order(piece_name, self._piece_positions):
            return
        if self._current_turn == 0:
            if piece_name != PieceName.wQ:
                move_set.fast_add(Move(piece_name, NULL_POSITION, ORIGIN_POSITION))
        elif self._current_turn == 1:
            if piece_name != PieceName.bQ:
                for d in range(int(Direction.NumDirections)):
                    move_set.fast_add(Move(piece_name, NULL_POSITION, ORIGIN_NEIGHBORS[d]))
        elif self.piece_in_hand(piece_name):
            if self.current_player_turn != 4 or (
                self.current_player_turn == 4
                and (
                    self.current_turn_queen_in_play
                    or (not self.current_turn_queen_in_play and _get_bug_type(piece_name) == BugType.QueenBee)
                )
            ):
                self._calculate_valid_placements()
                for placement in self._cached_valid_placements:
                    move_set.fast_add(Move(piece_name, NULL_POSITION, placement))
        elif (
            piece_name != self._last_piece_moved
            and self.current_turn_queen_in_play
            and self._piece_is_on_top(piece_name)
        ):
            if self.can_move_without_breaking_hive(piece_name):
                bug_type = _get_bug_type(piece_name)
                if bug_type == BugType.QueenBee:
                    self._get_valid_slides(piece_name, move_set, 1)
                elif bug_type == BugType.Spider:
                    self._get_valid_slides(piece_name, move_set, 3)
                elif bug_type == BugType.Beetle:
                    self._get_valid_beetle_moves(piece_name, move_set)
                elif bug_type == BugType.Grasshopper:
                    self._get_valid_grasshopper_moves(piece_name, move_set)
                elif bug_type == BugType.SoldierAnt:
                    self._get_valid_slides(piece_name, move_set)
                elif bug_type == BugType.Mosquito:
                    self._get_valid_mosquito_moves(piece_name, move_set, False)
                elif bug_type == BugType.Ladybug:
                    self._get_valid_ladybug_moves(piece_name, move_set)
                elif bug_type == BugType.Pillbug:
                    new_moves = MoveSet()
                    self._get_valid_slides(piece_name, new_moves, 1)
                    self._get_valid_pillbug_special_moves(piece_name, new_moves)
                    for mv in new_moves:
                        move_set.add(mv)
            else:
                bug_type = _get_bug_type(piece_name)
                if bug_type == BugType.Mosquito:
                    self._get_valid_mosquito_moves(piece_name, move_set, True)
                elif bug_type == BugType.Pillbug:
                    self._get_valid_pillbug_special_moves(piece_name, move_set)

    def _calculate_valid_placements(self):
        if not self._cached_valid_placements_ready:
            self._cached_valid_placements.clear()
            start_piece = int(PieceName.wQ if self.current_color == PlayerColor.White else PieceName.bQ)
            end_piece = int(PieceName.bQ if self.current_color == PlayerColor.White else PieceName.NumPieceNames)
            for pn in range(start_piece, end_piece):
                piece_name = PieceName(pn)
                if self._piece_is_on_top(piece_name):
                    bottom_position = self._piece_positions[pn].get_bottom()
                    for d in range(int(Direction.NumDirections)):
                        neighbor = bottom_position.get_neighbor_at(Direction(d))
                        neighbor_piece = self.get_piece_on_top_at(neighbor)
                        if neighbor_piece != PieceName.INVALID:
                            if _get_color(neighbor_piece) != self.current_color:
                                d += 1
                        else:
                            original_piece_dir = (d + int(Direction.NumDirections) // 2) % int(Direction.NumDirections)
                            valid_placement = True
                            for d2 in range(int(Direction.NumDirections)):
                                if d2 != original_piece_dir:
                                    surrounding_position = neighbor.get_neighbor_at(Direction(d2))
                                    surrounding_piece = self.get_piece_on_top_at(surrounding_position)
                                    if surrounding_piece != PieceName.INVALID and _get_color(surrounding_piece) != self.current_color:
                                        valid_placement = False
                                        break
                            if valid_placement:
                                self._cached_valid_placements.add(neighbor)
            self._cached_valid_placements_ready = True

    def _get_valid_beetle_moves(self, piece_name, move_set):
        piece_index = int(piece_name)
        for direction in range(int(Direction.NumDirections)):
            new_position = self._piece_positions[piece_index].get_neighbor_at(Direction(direction))
            top_neighbor = self.get_piece_on_top_at(new_position)
            left_of_target = _left_of(Direction(direction))
            right_of_target = _right_of(Direction(direction))
            left_neighbor_position = self._piece_positions[piece_index].get_neighbor_at(left_of_target)
            right_neighbor_position = self._piece_positions[piece_index].get_neighbor_at(right_of_target)
            top_left_neighbor = self.get_piece_on_top_at(left_neighbor_position)
            top_right_neighbor = self.get_piece_on_top_at(right_neighbor_position)
            current_height = self._piece_positions[piece_index].stack + 1
            destination_height = self._piece_positions[int(top_neighbor)].stack + 1 if top_neighbor != PieceName.INVALID else 0
            top_left_height = self._piece_positions[int(top_left_neighbor)].stack + 1 if top_left_neighbor != PieceName.INVALID else 0
            top_right_height = self._piece_positions[int(top_right_neighbor)].stack + 1 if top_right_neighbor != PieceName.INVALID else 0
            current_height -= 1
            if not (current_height == 0 and destination_height == 0 and top_left_height == 0 and top_right_height == 0):
                if not (destination_height < top_left_height and destination_height < top_right_height and current_height < top_left_height and current_height < top_right_height):
                    target_move = Move(piece_name, self._piece_positions[piece_index], Position(new_position.q, new_position.r, destination_height))
                    move_set.fast_add(target_move)

    def _get_valid_grasshopper_moves(self, piece_name, move_set):
        starting_position = self._piece_positions[int(piece_name)]
        for d in range(int(Direction.NumDirections)):
            landing_position = starting_position.get_neighbor_at(Direction(d))
            distance = 0
            while self._has_piece_at(landing_position):
                landing_position = landing_position.get_neighbor_at(Direction(d))
                distance += 1
            if distance > 0:
                move_set.fast_add(Move(piece_name, starting_position, landing_position))

    def _get_valid_mosquito_moves(self, piece_name, move_set, special_ability_only):
        position = self.get_position(piece_name)
        if position.stack > 0 and not special_ability_only:
            self._get_valid_beetle_moves(piece_name, move_set)
            return
        bug_types_evaluated = [False] * int(BugType.NumBugTypes)
        for d in range(int(Direction.NumDirections)):
            neighbor_position = position.get_neighbor_at(Direction(d))
            neighbor_piece_name = self.get_piece_on_top_at(neighbor_position)
            neighbor_bug_type = _get_bug_type(neighbor_piece_name)
            if neighbor_piece_name != PieceName.INVALID and not bug_types_evaluated[int(neighbor_bug_type)]:
                new_moves = MoveSet()
                if special_ability_only:
                    if neighbor_bug_type == BugType.Pillbug:
                        self._get_valid_pillbug_special_moves(piece_name, new_moves)
                else:
                    if neighbor_bug_type == BugType.QueenBee:
                        self._get_valid_slides(piece_name, new_moves, 1)
                    elif neighbor_bug_type == BugType.Spider:
                        self._get_valid_slides(piece_name, new_moves, 3)
                    elif neighbor_bug_type == BugType.Beetle:
                        self._get_valid_beetle_moves(piece_name, new_moves)
                    elif neighbor_bug_type == BugType.Grasshopper:
                        self._get_valid_grasshopper_moves(piece_name, new_moves)
                    elif neighbor_bug_type == BugType.SoldierAnt:
                        self._get_valid_slides(piece_name, new_moves)
                    elif neighbor_bug_type == BugType.Ladybug:
                        self._get_valid_ladybug_moves(piece_name, new_moves)
                    elif neighbor_bug_type == BugType.Pillbug:
                        self._get_valid_slides(piece_name, new_moves, 1)
                        self._get_valid_pillbug_special_moves(piece_name, new_moves)
                for mv in new_moves:
                    move_set.add(mv)
                bug_types_evaluated[int(neighbor_bug_type)] = True

    def _get_valid_ladybug_moves(self, piece_name, move_set):
        starting_position = self.get_position(piece_name)
        first_moves = MoveSet()
        self._get_valid_beetle_moves(piece_name, first_moves)
        for first_move in first_moves:
            if first_move.destination.stack > 0:
                self._set_position(piece_name, first_move.destination, False)
                second_moves = MoveSet()
                self._get_valid_beetle_moves(piece_name, second_moves)
                for second_move in second_moves:
                    if second_move.destination.stack > 0:
                        self._set_position(piece_name, second_move.destination, False)
                        third_moves = MoveSet()
                        self._get_valid_beetle_moves(piece_name, third_moves)
                        for third_move in third_moves:
                            if third_move.destination.stack == 0 and third_move.destination != starting_position:
                                move_set.add(Move(piece_name, starting_position, third_move.destination))
                        self._set_position(piece_name, first_move.destination, False)
                self._set_position(piece_name, starting_position, False)

    def _get_valid_pillbug_special_moves(self, piece_name, move_set):
        position = self.get_position(piece_name)
        position_above_target_piece = position.get_above()
        for d in range(int(Direction.NumDirections)):
            neighbor_position = position.get_neighbor_at(Direction(d))
            neighbor_piece_name = self._get_piece_at(neighbor_position)
            if (
                neighbor_piece_name != PieceName.INVALID
                and neighbor_piece_name != self._last_piece_moved
                and not self._has_piece_at_dir(neighbor_position, Direction.Above)
                and self.can_move_without_breaking_hive(neighbor_piece_name)
            ):
                first_move = Move(neighbor_piece_name, neighbor_position, position_above_target_piece)
                first_moves = MoveSet()
                self._get_valid_beetle_moves(neighbor_piece_name, first_moves)
                if first_moves.contains(first_move):
                    self._set_position(neighbor_piece_name, position_above_target_piece, False)
                    second_moves = MoveSet()
                    self._get_valid_beetle_moves(neighbor_piece_name, second_moves)
                    for second_move in second_moves:
                        if second_move.destination.stack == 0 and second_move.destination != neighbor_position:
                            move_set.add(Move(neighbor_piece_name, neighbor_position, second_move.destination))
                    self._set_position(neighbor_piece_name, neighbor_position, False)

    def _get_valid_slides(self, piece_name, move_set, fixed_range=0):
        starting_position = self.get_position(piece_name)
        self._set_position(piece_name, NULL_POSITION, False)
        if fixed_range > 0:
            self._get_valid_slides_fixed(piece_name, move_set, starting_position, starting_position, starting_position, fixed_range, fixed_range == 1)
        else:
            self._get_valid_slides_unbounded(piece_name, move_set, starting_position, starting_position, starting_position)
        self._set_position(piece_name, starting_position, False)

    def _get_valid_slides_unbounded(self, piece_name, move_set, starting_position, last_position, current_position):
        for slide_direction in range(int(Direction.NumDirections)):
            slide_position = current_position.get_neighbor_at(Direction(slide_direction))
            if slide_position != last_position and slide_position != starting_position and not self._has_piece_at(slide_position):
                if self._has_piece_at_dir(current_position, _right_of(Direction(slide_direction))) != self._has_piece_at_dir(current_position, _left_of(Direction(slide_direction))):
                    move = Move(piece_name, starting_position, slide_position)
                    if move_set.add(move):
                        self._get_valid_slides_unbounded(piece_name, move_set, starting_position, current_position, slide_position)

    def _get_valid_slides_fixed(self, piece_name, move_set, starting_position, last_position, current_position, remaining_slides, fast_add):
        if remaining_slides == 0:
            move = Move(piece_name, starting_position, current_position)
            if fast_add:
                move_set.fast_add(move)
            else:
                move_set.add(move)
        else:
            for slide_direction in range(int(Direction.NumDirections)):
                slide_position = current_position.get_neighbor_at(Direction(slide_direction))
                if slide_position != last_position and slide_position != starting_position and not self._has_piece_at(slide_position):
                    if self._has_piece_at_dir(current_position, _right_of(Direction(slide_direction))) != self._has_piece_at_dir(current_position, _left_of(Direction(slide_direction))):
                        self._get_valid_slides_fixed(piece_name, move_set, starting_position, current_position, slide_position, remaining_slides - 1, fast_add)

    def _set_position(self, piece_name, position, update_zobrist):
        old_position = self._piece_positions[int(piece_name)]
        if old_position.stack >= 0:
            if update_zobrist:
                self._zobrist_hash.toggle_piece(piece_name, old_position)
            self._piece_grid[HALF_BOARD + old_position.q][HALF_BOARD + old_position.r][old_position.stack] = PieceName.INVALID
        self._piece_positions[int(piece_name)] = position
        if position.stack >= 0:
            if update_zobrist:
                self._zobrist_hash.toggle_piece(piece_name, position)
            self._piece_grid[HALF_BOARD + position.q][HALF_BOARD + position.r][position.stack] = piece_name

    def _get_piece_at(self, position):
        return self._piece_grid[HALF_BOARD + position.q][HALF_BOARD + position.r][position.stack]

    def _get_piece_at_dir(self, position, direction):
        delta = NEIGHBOR_DELTAS[direction.value]
        return self._piece_grid[HALF_BOARD + position.q + delta[0]][HALF_BOARD + position.r + delta[1]][position.stack + delta[2]]

    def get_piece_on_top_at(self, position):
        top_piece = PieceName.INVALID
        for stack in range(BOARD_STACK_SIZE):
            piece = self._piece_grid[HALF_BOARD + position.q][HALF_BOARD + position.r][stack]
            if piece == PieceName.INVALID:
                break
            top_piece = piece
        return top_piece

    def _has_piece_at(self, position):
        return self._get_piece_at(position) != PieceName.INVALID

    def _has_piece_at_dir(self, position, direction):
        return self._get_piece_at_dir(position, direction) != PieceName.INVALID

    def _piece_is_on_top(self, piece_name):
        return self.piece_in_play(piece_name) and not self._has_piece_at_dir(
            self._piece_positions[int(piece_name)], Direction.Above
        )

    def _reset_state(self):
        wq = self._count_neighbors(PieceName.wQ)[0] >= 6
        bq = self._count_neighbors(PieceName.bQ)[0] >= 6
        if wq and bq:
            self.board_state = BoardState.Draw
        elif wq:
            self.board_state = BoardState.BlackWins
        elif bq:
            self.board_state = BoardState.WhiteWins
        else:
            self.board_state = BoardState.NotStarted if self._current_turn == 0 else BoardState.InProgress

    def _reset_caches(self):
        self._cached_valid_placements_ready = False
        self._cached_valid_moves = None
        self._cached_enemy_queen_neighbors = None

    def _count_neighbors(self, piece_name):
        total = friendly = enemy = 0
        if self.piece_in_play(piece_name):
            piece_color = _get_color(piece_name)
            for d in range(int(Direction.NumDirections)):
                neighbor = self._get_piece_at_dir(self._piece_positions[int(piece_name)], Direction(d))
                if neighbor != PieceName.INVALID:
                    total += 1
                    if piece_color == _get_color(neighbor):
                        friendly += 1
                    else:
                        enemy += 1
        return total, friendly, enemy


# ======================================================================
#  INLINED: mzinga/rl/model.py + mcts.py (adapted)
# ======================================================================

MAX_MOVES = 2048
OBS_DIM = 88
BOARD_NORM = 16.0
_C_TERMINAL = -1


def board_to_obs(board):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for pn in PieceName:
        if pn in (PieceName.INVALID, PieceName.NumPieceNames):
            continue
        idx = pn.value * 3
        pos = board.get_position(pn)
        if pos.stack < 0:
            obs[idx] = 0.0
            obs[idx + 1] = 0.0
            obs[idx + 2] = _C_TERMINAL
        else:
            obs[idx] = float(np.clip(pos.q / BOARD_NORM, -1.0, 1.0))
            obs[idx + 1] = float(np.clip(pos.r / BOARD_NORM, -1.0, 1.0))
            obs[idx + 2] = pos.stack / 8.0
    obs[84] = 1.0 if board.current_color == PlayerColor.Black else -1.0
    obs[85] = board.current_turn / 100.0
    obs[86] = 1.0 if board.current_turn_queen_in_play else 0.0
    obs[87] = 1.0 if board.game_is_over else 0.0
    return obs


def game_outcome(board):
    state = board.board_state
    if state == BoardState.Draw:
        return 0.0
    if state == BoardState.WhiteWins:
        return 1.0
    if state == BoardState.BlackWins:
        return -1.0
    return 0.0


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + F.relu(self.norm(self.fc(x)))


class HivePolicyValue(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=512, num_blocks=6, num_actions=MAX_MOVES):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.input_norm = nn.LayerNorm(obs_dim)
        self.fc_in = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, obs):
        x = F.relu(self.fc_in(self.input_norm(obs)))
        x = self.blocks(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class MCTS:
    def __init__(
        self,
        model,
        num_simulations=100,
        c_puct=1.4,
        temperature=1.0,
        temperature_threshold=15,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.tree = {}
        self._sim_path = []

    def search(self, board):
        root_key = board.zobrist_key
        self._root_key = root_key
        self.tree = {}
        root_moves = list(board.get_valid_moves())
        root_move_strs = [board.try_get_move_string(m) or "" for m in root_moves]
        n_actions = len(root_moves)
        if n_actions == 0:
            return np.zeros(MAX_MOVES, np.float32), np.zeros(MAX_MOVES, np.float32), 0
        root_obs_t = torch.as_tensor(board_to_obs(board), dtype=torch.float32, device=self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device)
        root_mask = self._build_mask(root_moves)
        with torch.no_grad():
            root_logits, _ = self.model(root_obs_t.unsqueeze(0))
            root_logits_m = root_logits.squeeze(0).masked_fill(~root_mask, float("-inf"))
            root_priors = F.softmax(root_logits_m, dim=-1).cpu().numpy()
        root_priors_noisy = self._add_dirichlet(root_priors, root_mask.cpu().numpy(), n_actions)
        self.tree[root_key] = {}
        for ai in range(MAX_MOVES):
            if root_mask[ai]:
                self.tree[root_key][ai] = (0.0, 0.0, root_priors_noisy[ai], root_priors[ai])
        for _ in range(self.num_simulations):
            self._sim_path.clear()
            sim_key = root_key
            while sim_key in self.tree and self.tree[sim_key] and not board.game_is_over:
                children = self.tree[sim_key]
                best_a = None
                best_ucb = -float("inf")
                total_N = sum(c[0] for c in children.values())
                for a, (N, W, _pc, P_root) in children.items():
                    Q = W / N if N > 0 else 0.0
                    ucb = Q + self.c_puct * P_root * math.sqrt(total_N + 1e-6) / (1.0 + N)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_a = a
                if best_a is None:
                    break
                sim_moves = list(board.get_valid_moves())
                sim_strs = [board.try_get_move_string(m) or "" for m in sim_moves]
                if best_a >= len(sim_moves):
                    break
                self._sim_path.append((sim_key, best_a))
                board.trusted_play(sim_moves[best_a], sim_strs[best_a])
                sim_key = board.zobrist_key
            if board.game_is_over:
                leaf_value = self._board_terminal_value(board)
            else:
                sim_moves = list(board.get_valid_moves())
                if len(sim_moves) > 0:
                    sim_obs_t = torch.as_tensor(board_to_obs(board), dtype=torch.float32, device=next(self.model.parameters()).device)
                    sim_mask = self._build_mask(sim_moves)
                    with torch.no_grad():
                        sim_logits, sim_value = self.model(sim_obs_t.unsqueeze(0))
                        sim_logits_m = sim_logits.squeeze(0).masked_fill(~sim_mask, float("-inf"))
                        sim_priors = F.softmax(sim_logits_m, dim=-1).cpu().numpy()
                    if sim_key not in self.tree:
                        self.tree[sim_key] = {}
                    for ai in range(MAX_MOVES):
                        if sim_mask[ai] and ai not in self.tree[sim_key]:
                            self.tree[sim_key][ai] = (0.0, 0.0, sim_priors[ai], sim_priors[ai])
                    leaf_value = sim_value.item()
                else:
                    leaf_value = 0.0
            for prev_key, action in reversed(self._sim_path):
                N, W, P_current, P_root = self.tree[prev_key][action]
                N += 1.0
                W += leaf_value
                self.tree[prev_key][action] = (N, W, P_current, P_root)
                leaf_value = -leaf_value
            for _ in range(len(self._sim_path)):
                self._undo(board)
        children = self.tree[root_key]
        n_visits = {}
        for a, (N, W, P_current, P_root) in children.items():
            if N > 0:
                n_visits[a] = N
        if not n_visits:
            pi = np.zeros(MAX_MOVES, np.float32)
            pi_probs = root_priors_noisy
            best_a = int(np.argmax(root_priors))
        else:
            temp = self.temperature if board.current_turn < self.temperature_threshold else 1e-8
            indices = []
            visits_list = []
            for ai in sorted(n_visits.keys()):
                indices.append(ai)
                N = n_visits[ai]
                if temp > 1e-8:
                    visits_list.append(N ** (1.0 / temp))
                else:
                    visits_list.append(N)
            visits_arr = np.array(visits_list, np.float64)
            probs = visits_arr / visits_arr.sum()
            pi = np.zeros(MAX_MOVES, np.float32)
            pi_probs = np.zeros(MAX_MOVES, np.float32)
            for i, ai in enumerate(indices):
                pi[ai] = probs[i]
                pi_probs[ai] = N / sum(n_visits.values())
            best_a = max(n_visits, key=n_visits.get)
        return pi, pi_probs, best_a

    @staticmethod
    def _board_terminal_value(board):
        state = board.board_state
        if state == BoardState.Draw:
            return 0.0
        if state == BoardState.WhiteWins:
            return 1.0 if board.current_color == PlayerColor.White else -1.0
        if state == BoardState.BlackWins:
            return 1.0 if board.current_color == PlayerColor.Black else -1.0
        return 0.0

    @staticmethod
    def _undo(board):
        item = board.board_history._items.pop()
        move = item.move
        if move != PASS_MOVE:
            board._set_position(move.piece_name, move.source, True)
        board._current_turn = board._current_turn - 1
        board._reset_caches()

    def _build_mask(self, moves):
        mask = np.zeros(MAX_MOVES, dtype=bool)
        mask[: len(moves)] = True
        return torch.as_tensor(mask, dtype=torch.bool, device=next(self.model.parameters()).device)

    def root_visit_entropy(self):
        if not hasattr(self, "_root_key") or self._root_key not in self.tree:
            return 0.0
        children = self.tree[self._root_key]
        visits = [c[0] for c in children.values() if c[0] > 0]
        if not visits:
            return 0.0
        total = sum(visits)
        probs = [v / total for v in visits]
        return -sum(p * math.log(max(p, 1e-12)) for p in probs)

    def _add_dirichlet(self, priors, mask, n_valid):
        if n_valid <= 1:
            return priors
        noise = np.random.default_rng().dirichlet([self.dirichlet_alpha] * n_valid)
        result = priors.copy()
        j = 0
        for ai in range(MAX_MOVES):
            if mask[ai]:
                result[ai] = (1.0 - self.dirichlet_epsilon) * priors[ai] + self.dirichlet_epsilon * noise[j]
                j += 1
        return result


# ======================================================================
#  TRAINING: Replay buffer, self-play, train step, evaluation
# ======================================================================


class ReplayBuffer:
    def __init__(self, max_size=200_000):
        self.obs_list = []
        self.mask_list = []
        self.pi_list = []
        self.outcome_list = []
        self.max_size = max_size

    def add_game(self, examples):
        for obs, mask, pi, outcome in examples:
            if len(self.obs_list) >= self.max_size:
                idx = np.random.default_rng().integers(len(self.obs_list))
                self.obs_list[idx] = obs
                self.mask_list[idx] = mask
                self.pi_list[idx] = pi
                self.outcome_list[idx] = outcome
            else:
                self.obs_list.append(obs)
                self.mask_list.append(mask)
                self.pi_list.append(pi)
                self.outcome_list.append(outcome)

    def sample_batch(self, batch_size, device):
        rng = np.random.default_rng()
        indices = rng.integers(len(self.obs_list), size=batch_size)
        obs_b = np.stack([self.obs_list[i] for i in indices])
        mask_b = np.stack([self.mask_list[i] for i in indices])
        pi_b = np.stack([self.pi_list[i] for i in indices])
        out_b = np.array([self.outcome_list[i] for i in indices], dtype=np.float32)
        return (
            torch.as_tensor(obs_b, device=device),
            torch.as_tensor(mask_b, dtype=torch.bool, device=device),
            torch.as_tensor(pi_b, device=device),
            torch.as_tensor(out_b, device=device),
        )

    def __len__(self):
        return len(self.obs_list)


def self_play_game(model, mcts, max_steps=300):
    board = Board(GameType.Base)
    examples = []
    game_length = 0
    mcts_entropy_sum = 0.0
    mcts_entropy_n = 0

    for game_length in range(1, max_steps + 1):
        moves = list(board.get_valid_moves())
        if len(moves) == 0 or board.game_is_over:
            break
        pi, pi_probs, best_a = mcts.search(board)
        mcts_entropy_sum += mcts.root_visit_entropy()
        mcts_entropy_n += 1
        move = moves[best_a]
        move_str = board.try_get_move_string(move) or ""
        mask = np.zeros(MAX_MOVES, dtype=bool)
        mask[: len(moves)] = True
        obs = board_to_obs(board)
        examples.append((obs, mask, pi_probs, 0.0))
        board.trusted_play(move, move_str)

    terminated = board.game_is_over
    outcome = game_outcome(board) if terminated else 0.0

    for i in range(len(examples)):
        obs, mask, pi, _ = examples[i]
        is_black_turn = obs[84] > 0.0
        value = -outcome if is_black_turn else outcome
        examples[i] = (obs, mask, pi, value)

    mcts_entropy_avg = mcts_entropy_sum / max(1, mcts_entropy_n)
    return examples, board.board_state, game_length, terminated, mcts_entropy_avg


def train_step(model, optimizer, buffer, batch_size, device):
    if len(buffer) < batch_size:
        return {"policy_loss": 0.0, "value_loss": 0.0, "grad_norm": 0.0, "policy_entropy": 0.0}
    model.train()
    obs_b, mask_b, pi_b, out_b = buffer.sample_batch(batch_size, device)
    logits, values = model(obs_b)
    logits_m = logits.masked_fill(~mask_b, float("-inf"))
    log_probs = F.log_softmax(logits_m, dim=-1)
    policy_loss = -(pi_b * log_probs).masked_fill(~mask_b, 0.0).sum(-1).mean()
    value_loss = F.mse_loss(values, out_b)
    probs = F.softmax(logits_m, dim=-1)
    safe_log_p = log_probs.masked_fill(~mask_b, 0.0)
    entropy = -(probs * safe_log_p).sum(-1).mean().item()
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "grad_norm": gn.item() if hasattr(gn, "item") else float(gn),
        "policy_entropy": float(entropy),
    }


def evaluate(model, n_games=10, max_steps=300, device=None):
    wins = losses = draws = 0
    dev = device or next(model.parameters()).device
    for color in (PlayerColor.White, PlayerColor.Black):
        for _ in range(n_games // 2):
            board = Board(GameType.Base)
            for _ in range(max_steps):
                if board.game_is_over:
                    break
                moves = list(board.get_valid_moves())
                if len(moves) == 0:
                    break
                if board.current_color == color:
                    obs = board_to_obs(board)
                    mask = np.zeros(MAX_MOVES, dtype=bool)
                    mask[: len(moves)] = True
                    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=dev)
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                    with torch.no_grad():
                        logits, _ = model(obs_t.unsqueeze(0))
                        logits_m = logits.squeeze(0).masked_fill(~mask_t, float("-inf"))
                        action_idx = int(torch.argmax(logits_m))
                    move = moves[action_idx]
                    move_str = board.try_get_move_string(move) or ""
                    board.trusted_play(move, move_str)
                else:
                    opp_move = moves[np.random.default_rng().integers(len(moves))]
                    opp_str = board.try_get_move_string(opp_move) or ""
                    board.trusted_play(opp_move, opp_str)
            state = board.board_state
            if state == BoardState.Draw:
                draws += 1
            elif (
                (state == BoardState.WhiteWins and color == PlayerColor.White)
                or (state == BoardState.BlackWins and color == PlayerColor.Black)
            ):
                wins += 1
            else:
                losses += 1
    return wins / n_games, losses / n_games, draws / n_games


# ======================================================================
#  CHECKPOINT I/O
# ======================================================================


def save_checkpoint(model, optimizer, scheduler, iteration, path):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["iteration"]


# ======================================================================
#  MAIN
# ======================================================================


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero Hive Training (Colab)")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    p.add_argument("--n_iterations", type=int, default=500)
    p.add_argument("--games_per_iter", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--train_epochs", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_blocks", type=int, default=6)
    p.add_argument("--num_sims", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--buffer_size", type=int, default=500_000)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--eval_games", type=int, default=20)
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Dir to save checkpoints (e.g. Google Drive)")
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--wandb_project", type=str, default="mzinga-alphazero")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, nargs="*", default=["colab", "alphazero"])
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    model = HivePolicyValue(obs_dim=OBS_DIM, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks)
    model.to(device)
    print(f"Parameters: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_iterations, eta_min=args.lr_min)

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    mcts = MCTS(
        model=model,
        num_simulations=args.num_sims,
        c_puct=1.4,
        temperature=1.0,
        temperature_threshold=15,
    )

    buffer = ReplayBuffer(max_size=args.buffer_size)

    if wandb is not None:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            tags=args.wandb_tags,
            config={
                "hidden_dim": args.hidden_dim,
                "num_blocks": args.num_blocks,
                "num_sims": args.num_sims,
                "n_iterations": args.n_iterations,
                "games_per_iter": args.games_per_iter,
                "batch_size": args.batch_size,
                "train_epochs": args.train_epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "buffer_size": args.buffer_size,
                "max_steps": args.max_steps,
                "device": str(device),
                "params": model.param_count(),
                "resume_from": args.resume,
            },
        )

        wandb.define_metric("train/policy_loss", summary="min")
        wandb.define_metric("train/value_loss", summary="min")
        wandb.define_metric("train/grad_norm", summary="last")
        wandb.define_metric("train/policy_entropy", summary="last")
        wandb.define_metric("train/lr", summary="last")

        wandb.define_metric("exploration/mcts_entropy", summary="last")

        wandb.define_metric("eval/win_rate", summary="max")
        wandb.define_metric("eval/wins", summary="max")
        wandb.define_metric("eval/losses", summary="min")
        wandb.define_metric("eval/draws", summary="last")

        wandb.define_metric("game/avg_moves", summary="last")
        wandb.define_metric("game/terminal_rate", summary="last")
        wandb.define_metric("game/total_games", summary="max")
        wandb.define_metric("game/throughput_games", summary="max")
        wandb.define_metric("game/throughput_moves", summary="max")

        wandb.define_metric("system/buffer_size", summary="max")
        wandb.define_metric("system/elapsed_h", summary="last")
        wandb.define_metric("system/eta_h", summary="last")

        wandb.define_metric("*", step_metric="iteration")
        wandb.watch(model, log="gradients", log_freq=50, log_graph=False)

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    total_games = 0
    total_moves = 0
    total_terminated = 0
    win_rate = 0.0
    eval_wins = eval_losses = eval_draws = None
    t0 = time.time()

    for it in range(start_iter + 1, args.n_iterations + 1):
        it_mcts_ent = 0.0
        mcts_ent_cnt = 0

        for _ in range(args.games_per_iter):
            examples, final_state, game_len, term, mcts_ent = self_play_game(model, mcts, args.max_steps)
            buffer.add_game(examples)
            total_games += 1
            total_moves += game_len
            total_terminated += int(term)
            if mcts_ent > 0:
                it_mcts_ent += mcts_ent
                mcts_ent_cnt += 1

        mcts_ent_avg = it_mcts_ent / max(1, mcts_ent_cnt)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "grad_norm": 0.0, "policy_entropy": 0.0}
        for _ in range(args.train_epochs):
            metrics = train_step(model, optimizer, buffer, args.batch_size, device)
        scheduler.step()

        if it % args.eval_every == 0 or it == 1:
            w, l, d = evaluate(model, n_games=args.eval_games, max_steps=args.max_steps, device=device)
            win_rate = w
            eval_wins = int(w * args.eval_games)
            eval_losses = int(l * args.eval_games)
            eval_draws = int(d * args.eval_games)

        avg_moves = total_moves / max(1, total_games)
        term_rate = total_terminated / max(1, total_games)
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        it_time = elapsed / max(1, it - start_iter)
        eta = it_time * (args.n_iterations - it) if it > start_iter else 0
        games_per_sec = args.games_per_iter / max(0.001, it_time * args.n_iterations)
        tp_games = total_games / max(1e-6, elapsed)
        tp_moves = total_moves / max(1e-6, elapsed)

        log_dict = {
            "iteration": it,
            "train/policy_loss": metrics["policy_loss"],
            "train/value_loss": metrics["value_loss"],
            "train/grad_norm": metrics["grad_norm"],
            "train/policy_entropy": metrics["policy_entropy"],
            "train/lr": lr,
            "exploration/mcts_entropy": mcts_ent_avg,
            "eval/win_rate": win_rate,
            "game/avg_moves": avg_moves,
            "game/terminal_rate": term_rate,
            "game/total_games": total_games,
            "game/throughput_games": tp_games,
            "game/throughput_moves": tp_moves,
            "system/buffer_size": len(buffer),
            "system/elapsed_h": elapsed / 3600,
            "system/eta_h": eta / 3600,
        }

        if eval_wins is not None:
            log_dict["eval/wins"] = eval_wins
            log_dict["eval/losses"] = eval_losses
            log_dict["eval/draws"] = eval_draws

        if wandb is not None:
            wandb.log(log_dict, step=it)

        print(
            f"[{it:4d}/{args.n_iterations}] "
            f"p_loss={metrics['policy_loss']:.4f}  "
            f"v_loss={metrics['value_loss']:.4f}  "
            f"win={win_rate:.0%}  "
            f"buf={len(buffer):,}  "
            f"lr={lr:.2e}  "
            f"games={total_games}  "
            f"tp={tp_games:.1f}g/s  "
            f"eta={eta/60:.0f}m"
        )

        if args.checkpoint_dir and it % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{it:04d}.pt")
            save_checkpoint(model, optimizer, scheduler, it, ckpt_path)
            print(f"  -> saved {ckpt_path}")
            if wandb is not None:
                artifact = wandb.Artifact(
                    f"model-checkpoint-{it:04d}", type="model",
                    metadata={"iteration": it, "params": model.param_count()}
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact, aliases=[f"iter-{it}"])

    elapsed = time.time() - t0
    final_w, final_l, final_d = evaluate(model, n_games=50, max_steps=args.max_steps, device=device)

    final_path = os.path.join(args.checkpoint_dir or ".", "mzinga_alphazero_final.pt")
    save_checkpoint(model, optimizer, scheduler, args.n_iterations, final_path)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final eval (50 games vs random): W={final_w:.1%} L={final_l:.1%} D={final_d:.1%}")
    print(f"  Total time: {elapsed/60:.1f}m")
    print(f"  Parameters: {model.param_count():,}")
    print(f"  Buffer: {len(buffer):,}")
    print(f"  Total games: {total_games}")
    print(f"  Model saved: {final_path}")

    if wandb is not None:
        wandb.summary["final/win_rate"] = final_w
        wandb.summary["final/loss_rate"] = final_l
        wandb.summary["final/draw_rate"] = final_d
        wandb.summary["final/total_time_h"] = elapsed / 3600
        wandb.summary["final/total_games"] = total_games
        wandb.summary["final/params"] = model.param_count()
        final_artifact = wandb.Artifact(
            "mzinga-alphazero-final", type="model",
            metadata={"params": model.param_count(), "iterations": args.n_iterations}
        )
        final_artifact.add_file(final_path)
        wandb.log_artifact(final_artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
