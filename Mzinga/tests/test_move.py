from mzinga.core.move import Move, PASS_MOVE, PASS_STRING
from mzinga.core.enums import PieceName
from mzinga.core.position import Position, ORIGIN_POSITION, NULL_POSITION


def test_pass_move():
    assert PASS_MOVE.piece_name == PieceName.INVALID
    assert PASS_MOVE.source == NULL_POSITION
    assert PASS_MOVE.destination == NULL_POSITION


def test_move_creation():
    m = Move(PieceName.wB1, ORIGIN_POSITION, NULL_POSITION)
    assert m.piece_name == PieceName.wB1
    assert m.source == ORIGIN_POSITION
    assert m.destination == NULL_POSITION


def test_build_move_string_pass():
    result = Move.build_move_string(True, PieceName.INVALID, "", PieceName.INVALID, "")
    assert result == PASS_STRING


def test_build_move_string_placement():
    result = Move.build_move_string(False, PieceName.wB1, "", PieceName.INVALID, "")
    assert result == "wB1"


def test_build_move_string_with_before_sep():
    result = Move.build_move_string(False, PieceName.wB1, "/", PieceName.wQ, "")
    assert result == "wB1 /wQ"


def test_build_move_string_with_after_sep():
    result = Move.build_move_string(False, PieceName.wS1, "", PieceName.wQ, "\\")
    assert result == "wS1 wQ\\"


def test_build_move_string_on_top():
    result = Move.build_move_string(False, PieceName.wG1, "", PieceName.bQ, "")
    assert result == "wG1 bQ"


def test_try_normalize_move_string_pass():
    result = Move.try_normalize_move_string("pass")
    assert result == (True, PieceName.INVALID, "", PieceName.INVALID, "")


def test_try_normalize_move_string_placement():
    result = Move.try_normalize_move_string("wB1")
    assert result is not None
    is_pass, start_piece, before_sep, end_piece, after_sep = result
    assert is_pass is False
    assert start_piece == PieceName.wB1
    assert before_sep == ""
    assert end_piece == PieceName.INVALID
    assert after_sep == ""


def test_try_normalize_move_string_with_sep():
    result = Move.try_normalize_move_string("wB1 /wQ")
    assert result == (False, PieceName.wB1, "/", PieceName.wQ, "")

    result = Move.try_normalize_move_string("wS1 wQ\\")
    assert result == (False, PieceName.wS1, "", PieceName.wQ, "\\")

    result = Move.try_normalize_move_string("wG1 bQ")
    assert result == (False, PieceName.wG1, "", PieceName.bQ, "")

    result = Move.try_normalize_move_string("wB1 -wQ")
    assert result == (False, PieceName.wB1, "-", PieceName.wQ, "")


def test_try_normalize_move_string_invalid():
    assert Move.try_normalize_move_string("") is None
    assert Move.try_normalize_move_string("xyz") is None
    assert Move.try_normalize_move_string("wB1 xyz") is None
    assert Move.try_normalize_move_string("   ") is None


def test_move_equality():
    m1 = Move(PieceName.wB1, ORIGIN_POSITION, NULL_POSITION)
    m2 = Move(PieceName.wB1, ORIGIN_POSITION, NULL_POSITION)
    assert m1 == m2
    assert m1 == m1


def test_move_inequality():
    m1 = Move(PieceName.wB1, ORIGIN_POSITION, NULL_POSITION)
    m2 = Move(PieceName.wQ, ORIGIN_POSITION, NULL_POSITION)
    m3 = Move(PieceName.wB1, Position(0, 1, 0), NULL_POSITION)
    m4 = Move(PieceName.wB1, ORIGIN_POSITION, Position(1, 0, 0))
    assert m1 != m2
    assert m1 != m3
    assert m1 != m4
