"""Build canonical Hive contexts from Mzinga boards (Phase 5).

Every context is rendered as the canonical example from
``adaptation_plan.md``::

    <bos> <state> Base+MLP ; InProgress ; White [ 12 ]
    <features> white_queen_in_play yes ; black_queen_in_play yes ; last bA2
    <history> wB1 ; bB1 wB1/ ; wQ wB1- ; ...
    <legal> wA1 /bQ ; wG2 bS1- ; pass
    <move> <target_move> <eos>

The :class:`HiveContextBuilder` is the single source of truth for how
the model sees the board.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from mzinga.core.board import Board
from mzinga.core.enums import (
    BoardState,
    GameType,
    PieceName,
    PlayerColor,
)
from mzinga.core.move import Move

from .tokenizer import HiveContext, HiveTokenizer


def _game_type_token(game_type: GameType) -> str:
    from mzinga.core import enums as Enums
    return Enums.get_game_type_string(game_type) or "Base"


def _board_state_token(state: BoardState) -> str:
    mapping = {
        BoardState.NotStarted: "NotStarted",
        BoardState.InProgress: "InProgress",
        BoardState.Draw: "Draw",
        BoardState.WhiteWins: "WhiteWins",
        BoardState.BlackWins: "BlackWins",
    }
    return mapping.get(state, "NotStarted")


def _color_token(color: PlayerColor) -> str:
    return "White" if color == PlayerColor.White else "Black"


def _piece_name_str(piece: PieceName) -> str:
    return piece.name


def _try_get_move_string(board: Board, move: Move) -> Optional[str]:
    """Return the canonical move string for a Move, or None if unavailable."""
    # Mzinga's `get_move_string` requires state and is built for the current
    # board, so we use it directly. For PASS moves it returns "pass".
    try:
        return board.get_move_string(move)
    except Exception:
        return None


class HiveContextBuilder:
    """Build :class:`HiveContext` objects from Mzinga boards.

    The builder is stateless. Use :meth:`build` to produce a context for
    the board's current position, optionally supplying a target move
    string for supervised training samples.
    """

    # Coordinate tokens use the existing n0-n63 numeric vocab.
    # We offset signed (q, r) by 32, clip to [-31, 30], and reserve n63
    # for "piece not in play".
    _COORD_OFFSET = 32
    _COORD_MIN = -31
    _COORD_MAX = 30
    _COORD_SENTINEL = 63

    def __init__(self, tokenizer: HiveTokenizer, include_features: bool = True,
                 history_window: Optional[int] = None,
                 include_piece_coords: bool = True):
        self.tokenizer = tokenizer
        self.include_features = include_features
        self.history_window = history_window
        self.include_piece_coords = include_piece_coords

    # ----- feature extraction ---------------------------------------------

    def _coord_token(self, v: int) -> str:
        clipped = max(self._COORD_MIN, min(self._COORD_MAX, v))
        return f"n{clipped + self._COORD_OFFSET}"

    def _piece_feature_value(self, board: Board, pn: PieceName) -> str:
        pos = board.get_position(pn)
        is_queen = pn in (PieceName.wQ, PieceName.bQ)
        if pos.stack < 0:
            # Not in play: sentinel for coords (and queen counts).
            return " ".join([f"n{self._COORD_SENTINEL}"] * (6 if is_queen else 3))

        q_tok = self._coord_token(pos.q)
        r_tok = self._coord_token(pos.r)
        s_tok = f"n{max(0, min(self._COORD_SENTINEL - 1, pos.stack))}"
        if is_queen:
            total, friendly, enemy = board._count_neighbors(pn)
            t_tok = f"n{max(0, min(self._COORD_SENTINEL - 1, total))}"
            f_tok = f"n{max(0, min(self._COORD_SENTINEL - 1, friendly))}"
            e_tok = f"n{max(0, min(self._COORD_SENTINEL - 1, enemy))}"
            return f"{q_tok} {r_tok} {s_tok} {t_tok} {f_tok} {e_tok}"
        return f"{q_tok} {r_tok} {s_tok}"

    def _features(self, board: Board) -> List[tuple]:
        feats: List[tuple] = []
        feats.append(("white_queen_in_play", "yes" if board.piece_in_play(PieceName.wQ) else "no"))
        feats.append(("black_queen_in_play", "yes" if board.piece_in_play(PieceName.bQ) else "no"))
        if board.last_piece_moved != PieceName.INVALID:
            feats.append(("last", _piece_name_str(board.last_piece_moved)))
        # Turn counter (white is player 1 of each pair); clamp to numeric vocab range
        feats.append(("turn", f"n{max(0, min(63, board.current_turn))}"))

        if self.include_piece_coords:
            for pn in PieceName:
                if pn in (PieceName.INVALID, PieceName.NumPieceNames):
                    continue
                feats.append((pn.name, self._piece_feature_value(board, pn)))
        return feats

    def _history(self, board: Board) -> List[str]:
        moves: List[str] = []
        for item in board.board_history:
            ms = item.move_string or ""
            if ms:
                moves.append(ms)
        if self.history_window is not None and len(moves) > self.history_window:
            moves = moves[-self.history_window:]
        return moves

    def _legal_moves(self, board: Board) -> List[str]:
        moves = list(board.get_valid_moves())
        out: List[str] = []
        for mv in moves:
            s = _try_get_move_string(board, mv)
            if s is None:
                continue
            out.append(s)
        return out

    # ----- main builder ---------------------------------------------------

    def build(self, board: Board, target_move: Optional[str] = None,
              legal_moves: Optional[Sequence[str]] = None,
              features: Optional[Sequence[tuple]] = None) -> HiveContext:
        if not isinstance(board.game_type, GameType):
            raise TypeError("board.game_type must be a mzinga GameType")
        gt_token = _game_type_token(board.game_type)
        ctx = HiveContext(
            game_type=gt_token,
            board_state=_board_state_token(board.board_state),
            current_color=_color_token(board.current_color),
            current_turn=board.current_turn,
        )
        ctx.illegal_piece_ids = self.tokenizer.illegal_piece_ids_for_game_type(gt_token)
        if self.include_features:
            ctx.features = list(features) if features is not None else self._features(board)
        ctx.history = self._history(board)
        if legal_moves is not None:
            ctx.legal_moves = list(legal_moves)
        else:
            ctx.legal_moves = self._legal_moves(board)
        ctx.target_move = target_move
        return ctx

    def encode(self, board: Board, target_move: Optional[str] = None,
               legal_moves: Optional[Sequence[str]] = None,
               features: Optional[Sequence[tuple]] = None) -> List[int]:
        ctx = self.build(board, target_move=target_move,
                         legal_moves=legal_moves, features=features)
        return self.tokenizer.encode_context(ctx)