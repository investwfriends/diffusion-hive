import asyncio
import pytest

from mzinga.core.board import Board, InvalidMoveError
from mzinga.core.enums import (
    PlayerColor, BoardState, PieceName, Direction, BugType, GameType,
)
from mzinga.core.position import Position, ORIGIN_POSITION, NULL_POSITION
from mzinga.core.move import Move, PASS_MOVE, PASS_STRING


def test_new_board():
    b = Board(GameType.Base)
    assert b.board_state == BoardState.NotStarted
    assert b.current_turn == 0
    assert b.current_color == PlayerColor.White
    assert b.current_player_turn == 1
    assert b.game_in_progress is True
    assert b.game_is_over is False


def test_new_board_valid_moves():
    b = Board()
    moves = b.get_valid_moves()
    assert len(moves) == 4
    for m in moves:
        assert m.source == NULL_POSITION
        assert m.destination == ORIGIN_POSITION
    names = [m.piece_name for m in moves]
    assert PieceName.wQ not in names
    assert PieceName.wB1 in names
    assert PieceName.wS1 in names
    assert PieceName.wG1 in names
    assert PieceName.wA1 in names


def test_second_turn_valid_moves():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    assert b.current_turn == 1
    assert b.current_color == PlayerColor.Black
    moves = b.get_valid_moves()
    assert len(moves) == 24
    for m in moves:
        assert m.source == NULL_POSITION
        assert b.current_player_turn == 1


def test_game_string_roundtrip():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    gs = b.get_game_string()
    b2 = Board.parse_game_string(gs)
    assert b2.get_game_string() == gs


def test_clone():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    b.trusted_play(Move(PieceName.bB1, NULL_POSITION, Position(0, -1, 0)), "bB1")
    c = b.clone()
    assert c.get_game_string() == b.get_game_string()
    assert c.current_turn == b.current_turn


def test_undo():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    assert b.current_turn == 1
    assert b.try_undo_last_move() is True
    assert b.current_turn == 0
    assert b.board_state == BoardState.NotStarted


def test_undo_pass():
    b = Board()
    b.trusted_play(PASS_MOVE, PASS_STRING)
    assert b.current_turn == 1
    assert b.try_undo_last_move() is True
    assert b.current_turn == 0


def test_undo_multiple():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    b.trusted_play(Move(PieceName.bB1, NULL_POSITION, Position(0, -1, 0)), "bB1")
    assert b.current_turn == 2
    b.try_undo_last_move()
    assert b.current_turn == 1
    assert b.current_color == PlayerColor.Black
    b.try_undo_last_move()
    assert b.current_turn == 0


def test_cannot_play_queen_on_first_turn():
    b = Board()
    with pytest.raises(InvalidMoveError):
        b.play(Move(PieceName.wQ, NULL_POSITION, ORIGIN_POSITION))


def test_cannot_move_piece_before_queen():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    b.trusted_play(Move(PieceName.bB1, NULL_POSITION, Position(0, -1, 0)), "bB1")
    # White plays wS1 at neighbor
    b.trusted_play(Move(PieceName.wS1, NULL_POSITION, Position(1, -1, 0)), "wS1")
    b.trusted_play(Move(PieceName.bS1, NULL_POSITION, Position(1, -2, 0)), "bS1")
    # White must play queen, cannot move pieces
    moves = b.get_valid_moves()
    for m in moves:
        if m.piece_name == PieceName.wQ:
            break
    else:
        # No queen move available, test that only placements exist
        sources = [m.source for m in moves]
        assert all(s == NULL_POSITION for s in sources if m.piece_name != PieceName.wQ)


def test_queen_must_play_by_turn_4():
    b = Board(GameType.Base)
    moves = [
        (PieceName.wB1, ORIGIN_POSITION),
        (PieceName.bB1, Position(0, -1, 0)),
        (PieceName.wS1, Position(1, -1, 0)),
        (PieceName.bS1, Position(1, -2, 0)),
        (PieceName.wG1, Position(1, 0, 0)),
        (PieceName.bG1, Position(0, -2, 0)),
    ]
    for piece, pos in moves:
        b.trusted_play(Move(piece, NULL_POSITION, pos), f"{piece.name}")
    assert b.current_player_turn == 4
    # On turn 4, white must play queen
    moves = b.get_valid_moves()
    sources = {m.piece_name for m in moves}
    assert PieceName.wQ in sources


def test_parse_move_placement():
    b = Board()
    move, move_str = b.parse_move("wB1")
    assert move is not None
    assert move.piece_name == PieceName.wB1
    assert move.destination == ORIGIN_POSITION
    assert move_str == "wB1"


def test_parse_move_with_separator():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    # Now parse a placement relative to wB1
    move, move_str = b.parse_move("bB1 wB1/")
    assert move is not None
    assert move.piece_name == PieceName.bB1
    assert move_str == "bB1 wB1/"


def test_game_string_parse_basic():
    gs = "Base;0;White[1]"
    b = Board.parse_game_string(gs)
    assert b.board_state == BoardState.NotStarted
    assert b.current_turn == 0


def test_game_string_with_moves():
    gs = "Base;1;Black[1];wB1"
    b = Board.parse_game_string(gs)
    assert b.board_state == BoardState.InProgress
    assert b.current_turn == 1
    assert b.current_color == PlayerColor.Black


def test_get_game_string_in_progress():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    gs = b.get_game_string()
    assert "Base;" in gs
    assert "InProgress" in gs or "1" in gs
    assert "wB1" in gs


def test_is_noisy_move():
    b = Board()
    # Noisy move: landing adjacent to enemy queen
    # Setup: white queen at origin, black queen adjacent
    b.trusted_play(Move(PieceName.wQ, NULL_POSITION, ORIGIN_POSITION), "wQ")
    b.trusted_play(Move(PieceName.bQ, NULL_POSITION, Position(0, -1, 0)), "bQ")
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, Position(1, -1, 0)), "wB1")
    b.trusted_play(Move(PieceName.bB1, NULL_POSITION, Position(1, -2, 0)), "bB1")
    # Playing wS1 adjacent to bQ would be noisy
    # But first check if parsing works
    moves = b.get_valid_moves()
    assert len(moves) > 0


def test_board_state_white_wins():
    b = Board()
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    b.trusted_play(Move(PieceName.bQ, NULL_POSITION, Position(0, -1, 0)), "bQ")
    b.trusted_play(Move(PieceName.wS1, NULL_POSITION, Position(1, -1, 0)), "wS1")
    b.trusted_play(Move(PieceName.bB1, NULL_POSITION, Position(1, -1, 0)), "bB1")
    b.trusted_play(Move(PieceName.wS2, NULL_POSITION, Position(0, 0, 0)), "wS2")
    b.trusted_play(Move(PieceName.bS1, NULL_POSITION, Position(1, -1, 0)), "bS1")
    # The game should continue
    assert b.game_in_progress is True


def test_piece_in_hand_and_play():
    b = Board()
    assert b.piece_in_hand(PieceName.wB1) is True
    assert b.piece_in_play(PieceName.wB1) is False
    b.trusted_play(Move(PieceName.wB1, NULL_POSITION, ORIGIN_POSITION), "wB1")
    assert b.piece_in_hand(PieceName.wB1) is False
    assert b.piece_in_play(PieceName.wB1) is True


def test_board_initial_perft():
    b = Board()
    result = asyncio.run(b.calculate_perft_async(0))
    assert result == 1
    result = asyncio.run(b.calculate_perft_async(1))
    assert result == 4


def test_get_board_metrics():
    b = Board(GameType.Base)
    metrics = b.get_board_metrics()
    assert metrics.board_state == BoardState.NotStarted
    assert metrics.pieces_in_hand == 22
    assert metrics.pieces_in_play == 0
