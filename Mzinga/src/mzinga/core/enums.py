from __future__ import annotations

from enum import IntEnum


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


def game_in_progress(board_state):
    return board_state == BoardState.NotStarted or board_state == BoardState.InProgress


def game_is_over(board_state):
    return board_state in (BoardState.WhiteWins, BoardState.BlackWins, BoardState.Draw)


def get_color(piece_name):
    if piece_name.value < 14:
        return PlayerColor.White
    return PlayerColor.Black


def _get_bug_num(piece_name):
    v = piece_name.value % 14
    if v in (1, 3, 5, 8):
        return 1
    if v in (2, 4, 6, 9):
        return 2
    if v in (7, 10):
        return 3
    return 0


def try_get_bug_num(piece_name):
    return _get_bug_num(piece_name)


def left_of(direction):
    return Direction(((direction.value + _NUM_FLAT_DIRECTIONS - 1) % _NUM_FLAT_DIRECTIONS))


def right_of(direction):
    return Direction(((direction.value + 1) % _NUM_FLAT_DIRECTIONS))


def get_bug_type(piece_name):
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


def try_parse_game_type(s):
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


def get_game_type_string(game_type):
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


def piece_name_is_enabled_for_game_type(piece_name, game_type):
    if piece_name == PieceName.INVALID or game_type == GameType.INVALID:
        return False
    return ((0b1000000000000000000000000000 >> piece_name.value) & _piece_name_is_enabled_for_game_type[game_type.value]) != 0


def bug_type_is_enabled_for_game_type(bug_type, game_type):
    if bug_type == BugType.INVALID:
        return False
    if bug_type == BugType.Mosquito:
        return game_type in (GameType.BaseM, GameType.BaseML, GameType.BaseMP, GameType.BaseMLP)
    if bug_type == BugType.Ladybug:
        return game_type in (GameType.BaseL, GameType.BaseML, GameType.BaseLP, GameType.BaseMLP)
    if bug_type == BugType.Pillbug:
        return game_type in (GameType.BaseP, GameType.BaseMP, GameType.BaseLP, GameType.BaseMLP)
    return True


def enable_bug_type(bug_type, game_type, enabled):
    include_m = enabled if bug_type == BugType.Mosquito else bug_type_is_enabled_for_game_type(BugType.Mosquito, game_type)
    include_l = enabled if bug_type == BugType.Ladybug else bug_type_is_enabled_for_game_type(BugType.Ladybug, game_type)
    include_p = enabled if bug_type == BugType.Pillbug else bug_type_is_enabled_for_game_type(BugType.Pillbug, game_type)

    if include_m and include_l and include_p:
        return GameType.BaseMLP
    if include_l and include_p:
        return GameType.BaseLP
    if include_m and include_p:
        return GameType.BaseMP
    if include_m and include_l:
        return GameType.BaseML
    if include_p:
        return GameType.BaseP
    if include_l:
        return GameType.BaseL
    if include_m:
        return GameType.BaseM
    return GameType.Base


def num_piece_names(game_type):
    if game_type == GameType.Base:
        return int(PieceName.NumPieceNames) - 6
    if game_type in (GameType.BaseM, GameType.BaseL, GameType.BaseP):
        return int(PieceName.NumPieceNames) - 4
    if game_type in (GameType.BaseML, GameType.BaseMP, GameType.BaseLP):
        return int(PieceName.NumPieceNames) - 2
    return int(PieceName.NumPieceNames)


def num_bug_types(game_type):
    if game_type == GameType.Base:
        return int(BugType.NumBugTypes) - 3
    if game_type in (GameType.BaseM, GameType.BaseL, GameType.BaseP):
        return int(BugType.NumBugTypes) - 2
    if game_type in (GameType.BaseML, GameType.BaseMP, GameType.BaseLP):
        return int(BugType.NumBugTypes) - 1
    return int(BugType.NumBugTypes)
