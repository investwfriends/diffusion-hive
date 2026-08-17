from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from mzinga.core.enums import (
    PlayerColor,
    BoardState,
    PieceName,
    Direction,
    BugType,
    GameType,
)
from mzinga.core import enums as Enums
from mzinga.core.position import (
    Position,
    NULL_POSITION,
    ORIGIN_POSITION,
    ORIGIN_NEIGHBORS,
    BOARD_SIZE,
    BOARD_STACK_SIZE,
    HALF_BOARD,
    NEIGHBOR_DELTAS,
    NEIGHBOR_DQ,
    NEIGHBOR_DR,
    NEIGHBOR_DSTACK,
    NUM_NEIGHBOR_DIRECTIONS,
    pack_position,
    unpack_q,
    unpack_r,
    unpack_stack,
    PACKED_NULL,
    PACKED_ORIGIN,
    _COORD_MASK,
    _STACK_MASK,
    _STACK_SHIFT,
    _COORD_OFFSET as _POS_COORD_OFFSET,
)
from mzinga.core.move import Move, PASS_MOVE, PASS_STRING
from mzinga.core.fast_set import MoveSet
from mzinga.core.position_set import PositionSet
from mzinga.core.zobrist import ZobristHash
from mzinga.core.board_metrics import BoardMetrics


# Flat grid index: BOARD_SIZE * BOARD_SIZE * BOARD_STACK_SIZE = 128*128*8 = 131072
# Index = ((q + 64) * 128 + (r + 64)) * 8 + stack, with stack in 0..7.
_GRID_TOTAL = BOARD_SIZE * BOARD_SIZE * BOARD_STACK_SIZE
_GRID_STRIDE_R = BOARD_SIZE * BOARD_STACK_SIZE
_GRID_STRICE_Q = BOARD_STACK_SIZE


def _grid_idx(q: int, r: int, stack: int) -> int:
    return ((q + HALF_BOARD) * _GRID_STRIDE_R) + ((r + HALF_BOARD) * BOARD_STACK_SIZE) + stack


# Pre-compute grid index offsets for each of the 6 flat neighbor directions.
# This collapses the 3-axis neighbor lookup to a single add.
_GRID_NEIGHBOR_OFFSETS = [
    _grid_idx(NEIGHBOR_DQ[d], NEIGHBOR_DR[d], NEIGHBOR_DSTACK[d]) for d in range(7)
]


# Pre-compute the (q, r) column stride for get_piece_on_top_at.
_COL_STRIDE = BOARD_STACK_SIZE


class InvalidMoveError(Exception):
    def __init__(self, move: Move, message: str = "You can't move that piece there."):
        self.move = move
        super().__init__(message)


# Pre-compute commonly used enum values to avoid Enum.__call__ overhead in hot paths.
_NUM_PIECE_NAMES = int(PieceName.NumPieceNames)
_INVALID_PIECE = int(PieceName.INVALID)
_W_QUEEN = int(PieceName.wQ)
_B_QUEEN = int(PieceName.bQ)
_QUEEN_PIECE = (_W_QUEEN, _B_QUEEN)
_DIRECTION_UP = int(Direction.Up)
_DIRECTION_UPRIGHT = int(Direction.UpRight)
_DIRECTION_DOWNRIGHT = int(Direction.DownRight)
_DIRECTION_DOWN = int(Direction.Down)
_DIRECTION_DOWNLEFT = int(Direction.DownLeft)
_DIRECTION_UPLEFT = int(Direction.UpLeft)
_DIRECTION_ABOVE = int(Direction.Above)
_NUM_DIRECTIONS = int(Direction.NumDirections)
_HALF_DIRECTIONS = _NUM_DIRECTIONS // 2

# Pre-compute piece -> bug type for the in-hand check (for ordering rules).
# Order:  wS2=2, wB2=4, wG2=6, wG3=7, wA2=9, wA3=10 (and black equivalents)
# These are the pieces that have a "must play in order" rule.
_ORDERED_PIECE_NAMES = frozenset({
    int(PieceName.wS2), int(PieceName.wB2), int(PieceName.wG2), int(PieceName.wG3),
    int(PieceName.wA2), int(PieceName.wA3), int(PieceName.bS2), int(PieceName.bB2),
    int(PieceName.bG2), int(PieceName.bG3), int(PieceName.bA2), int(PieceName.bA3),
})


def _get_color_int(piece_name_int: int) -> int:
    return 0 if piece_name_int < 14 else 1  # 0 = White, 1 = Black


class Board:
    def __init__(self, game_type: GameType = GameType.Base):
        self.game_type = game_type
        self.board_state: BoardState = BoardState.NotStarted
        self._current_turn: int = 0
        self._last_piece_moved: int = _INVALID_PIECE  # packed int of PieceName
        self._piece_positions: list[int] = [PACKED_NULL] * _NUM_PIECE_NAMES  # packed ints

        # Flat piece grid: one int per (q, r, stack) cell. Indexed by _grid_idx().
        # PieceName.INVALID (= -1) is stored as -1. We use -1 directly so we can
        # compare grid[i] != -1 to check for occupancy without going through enum calls.
        self._piece_grid: list[int] = [-1] * _GRID_TOTAL

        self._cached_valid_placements_ready: bool = False
        self._cached_valid_placements: PositionSet = PositionSet()
        self._cached_valid_moves: Optional[MoveSet] = None
        self._cached_enemy_queen_neighbors: Optional[PositionSet] = None

        self._part_of_hive: list[bool] = [False] * _NUM_PIECE_NAMES
        self._pieces_to_look_at: deque = deque()

        self._zobrist_hash: ZobristHash = ZobristHash()

        # Lazy queen-surrounded state (avoid computing on every turn).
        self._queen_surrounded_dirty: bool = True
        self._cached_board_state: BoardState = BoardState.NotStarted

        from mzinga.core.board_history import BoardHistory
        self.board_history = BoardHistory()

    @property
    def game_in_progress(self) -> bool:
        return self._cached_board_state == BoardState.NotStarted or self._cached_board_state == BoardState.InProgress

    @property
    def game_is_over(self) -> bool:
        state = self._cached_board_state
        return state == BoardState.WhiteWins or state == BoardState.BlackWins or state == BoardState.Draw

    @property
    def current_turn(self) -> int:
        return self._current_turn

    @current_turn.setter
    def current_turn(self, value: int) -> None:
        if value < 0:
            raise ValueError("Current turn must be >= 0")
        old_color = self._current_turn & 1
        self._current_turn = value
        new_color = value & 1
        if old_color != new_color:
            self._zobrist_hash.toggle_turn()
        self._queen_surrounded_dirty = True
        self._reset_caches()

    @property
    def current_player_turn(self) -> int:
        return 1 + (self._current_turn >> 1)

    @property
    def current_color(self) -> PlayerColor:
        return PlayerColor(self._current_turn & 1)

    @property
    def current_color_int(self) -> int:
        return self._current_turn & 1

    @property
    def current_turn_queen_in_play(self) -> bool:
        queen = _W_QUEEN if (self._current_turn & 1) == 0 else _B_QUEEN
        return self._piece_positions[queen] != PACKED_NULL

    @property
    def last_piece_moved(self) -> PieceName:
        return PieceName(self._last_piece_moved)

    @last_piece_moved.setter
    def last_piece_moved(self, value: PieceName) -> None:
        v = int(value)
        old = self._last_piece_moved
        self._last_piece_moved = v
        if old != v:
            self._zobrist_hash.toggle_last_moved_piece(PieceName(old))
            self._zobrist_hash.toggle_last_moved_piece(value)

    @property
    def zobrist_key(self) -> int:
        return self._zobrist_hash.value

    def _board_state_cached(self) -> BoardState:
        if self._queen_surrounded_dirty:
            wq_surrounded = self._count_neighbors_packed(_W_QUEEN) >= 6
            bq_surrounded = self._count_neighbors_packed(_B_QUEEN) >= 6
            if wq_surrounded and bq_surrounded:
                self._cached_board_state = BoardState.Draw
            elif wq_surrounded:
                self._cached_board_state = BoardState.BlackWins
            elif bq_surrounded:
                self._cached_board_state = BoardState.WhiteWins
            else:
                self._cached_board_state = BoardState.NotStarted if self._current_turn == 0 else BoardState.InProgress
            self._queen_surrounded_dirty = False
        return self._cached_board_state

    @property
    def board_state(self) -> BoardState:
        return self._board_state_cached()

    @board_state.setter
    def board_state(self, value: BoardState) -> None:
        self._cached_board_state = value
        self._queen_surrounded_dirty = False

    @staticmethod
    def parse_game_string(game_str: str, trusted_play: bool = False) -> Board:
        split = game_str.split(";")
        game_type_str = split[0]
        game_type = Enums.try_parse_game_type(game_type_str)
        if game_type is None:
            raise ValueError(f"Unable to parse '{game_type_str}' in GameString.")
        board = Board(game_type)
        for i in range(3, len(split)):
            move_str = split[i]
            move, parsed_move_str = board.parse_move(move_str)
            if move is None:
                raise ValueError(f"Unable to parse '{move_str}' in GameString.")
            assert parsed_move_str is not None
            if trusted_play:
                board.trusted_play(move, parsed_move_str)
            elif not board.try_play_move(move, parsed_move_str):
                raise ValueError(f"Unable to play '{move_str}' in GameString.")
        return board

    @staticmethod
    def try_parse_game_string(game_str: str, trusted_play: bool = False) -> Optional[Board]:
        try:
            return Board.parse_game_string(game_str, trusted_play)
        except Exception:
            return None

    def get_game_string(self) -> str:
        parts = [f"{Enums.get_game_type_string(self.game_type)};{self._board_state_cached()};{self.current_color.name}[{self.current_player_turn}]"]
        for item in self.board_history:
            parts.append(f";{item.move_string}")
        return "".join(parts)

    def get_valid_moves(self) -> MoveSet:
        if self._cached_valid_moves is None:
            moves = MoveSet()
            if self.game_in_progress:
                start_piece = _W_QUEEN if (self._current_turn & 1) == 0 else _B_QUEEN
                end_piece = _B_QUEEN if (self._current_turn & 1) == 0 else _NUM_PIECE_NAMES
                for pn in range(start_piece, end_piece):
                    self._get_valid_moves_for_piece(pn, moves)
                if len(moves) == 0:
                    moves.fast_add(PASS_MOVE)
            self._cached_valid_moves = moves
        return self._cached_valid_moves

    def play(self, move: Move, move_string: str = "") -> None:
        if move == PASS_MOVE:
            self.pass_move()
            return
        if self.game_is_over:
            raise InvalidMoveError(move, "You can't play, the game is over.")
        valid_moves = self.get_valid_moves()
        if not valid_moves.contains(move):
            if Enums.get_color(move.piece_name) != self.current_color:
                raise InvalidMoveError(move, "It's not that player's turn.")
            if not Enums.piece_name_is_enabled_for_game_type(move.piece_name, self.game_type):
                raise InvalidMoveError(move, "That piece is not enabled in this game.")
            if move.destination == NULL_POSITION:
                raise InvalidMoveError(move, "You can't put a piece back into your hand.")
            if self.current_player_turn == 1 and Enums.get_bug_type(move.piece_name) == BugType.QueenBee:
                raise InvalidMoveError(move, "You can't play your Queen Bee on your first turn.")
            if not self.current_turn_queen_in_play:
                if self.current_player_turn == 4 and Enums.get_bug_type(move.piece_name) != BugType.QueenBee:
                    raise InvalidMoveError(move, "You must play your Queen Bee on or before your fourth turn.")
                elif self.piece_in_play(move.piece_name):
                    raise InvalidMoveError(move, "You can't move a piece in play until you've played your Queen Bee.")
            if not self._placing_piece_in_order(move.piece_name):
                raise InvalidMoveError(move, "When there are multiple pieces of the same bug type, you must play the pieces in order.")
            if self._has_piece_at(move.destination):
                raise InvalidMoveError(move, "You can't move there because a piece already exists at that position.")
            if self.piece_in_play(move.piece_name):
                if not self._piece_is_on_top(move.piece_name):
                    raise InvalidMoveError(move, "You can't move that piece because it has another piece on top of it.")
                elif not self.can_move_without_breaking_hive(move.piece_name):
                    raise InvalidMoveError(move, "You can't move that piece because it will break the hive.")
            raise InvalidMoveError(move)
        self.trusted_play(move, move_string)

    def pass_move(self) -> None:
        if self.game_is_over:
            raise InvalidMoveError(PASS_MOVE, "You can't pass, the game is over.")
        if not self.get_valid_moves().contains(PASS_MOVE):
            raise InvalidMoveError(PASS_MOVE, "You can't pass when you have valid moves.")
        self.trusted_play(PASS_MOVE, PASS_STRING)

    def try_play_move(self, move: Move, move_string: str = "") -> bool:
        valid_moves = self.get_valid_moves()
        if valid_moves.contains(move):
            self.trusted_play(move, move_string)
            return True
        return False

    def try_undo_last_move(self) -> bool:
        if self.board_history.count > 0:
            last_item = self.board_history[-1]
            if last_item.move != PASS_MOVE:
                self._set_position(last_item.move.piece_name, last_item.move.source, True)
            self.board_history.undo_last()
            self._last_piece_moved = (
                int(self.board_history.last_move.piece_name)
                if self.board_history.last_move is not None
                else _INVALID_PIECE
            )
            self._current_turn -= 1
            self._queen_surrounded_dirty = True
            self._reset_caches()
            return True
        return False

    def get_move_string(self, move: Move) -> str:
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
            for d in range(_NUM_DIRECTIONS):
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
        raise ValueError("Invalid move.")

    def try_get_move_string(self, move: Move) -> Optional[str]:
        try:
            return self.get_move_string(move)
        except Exception:
            return None

    def parse_move(self, move_string: str) -> tuple[Optional[Move], Optional[str]]:
        result = Move.try_normalize_move_string(move_string)
        if result is not None:
            is_pass, start_piece, before_separator, end_piece, after_separator = result
            result_string = Move.build_move_string(is_pass, start_piece, before_separator, end_piece, after_separator)
            if is_pass:
                return PASS_MOVE, result_string
            sp_int = int(start_piece)
            source = self._position_from_packed(self._piece_positions[sp_int])
            destination = ORIGIN_POSITION
            if end_piece != PieceName.INVALID:
                ep_int = int(end_piece)
                target_position = self._position_from_packed(self._piece_positions[ep_int])
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

    def is_noisy_move(self, move: Move) -> bool:
        if move == PASS_MOVE:
            return False
        if self._cached_enemy_queen_neighbors is None:
            self._cached_enemy_queen_neighbors = PositionSet()
            enemy_queen_pos_packed = self._piece_positions[_B_QUEEN if (self._current_turn & 1) == 0 else _W_QUEEN]
            if enemy_queen_pos_packed != PACKED_NULL:
                eq_q = unpack_q(enemy_queen_pos_packed)
                eq_r = unpack_r(enemy_queen_pos_packed)
                eq_stack = unpack_stack(enemy_queen_pos_packed)
                for d in range(_NUM_DIRECTIONS):
                    nq = eq_q + NEIGHBOR_DQ[d]
                    nr = eq_r + NEIGHBOR_DR[d]
                    self._cached_enemy_queen_neighbors.add(Position(nq, nr, eq_stack))
        if move.destination in self._cached_enemy_queen_neighbors:
            pn = int(move.piece_name)
            return self._piece_positions[pn] not in self._cached_enemy_queen_neighbors  # not quite right - needs Position
        return False

    def calculate_perft(self, depth: int) -> int:
        return asyncio.run(self.calculate_perft_async(depth))

    async def calculate_perft_async(self, depth: int) -> int:
        # The async wrapper preserves the existing API (used by tests/test_perft.py);
        # it just delegates to the synchronous recursive implementation. asyncio
        # add ~50us per await frame at recursion depths we actually use, so the
        # plain recursion below is much faster for the perft CLI.
        return self._calculate_perft_sync(depth)

    def _calculate_perft_sync(self, depth: int) -> int:
        if depth == 0:
            return 1
        valid_moves = self.get_valid_moves()
        if depth == 1:
            return len(valid_moves)
        nodes = 0
        for move in valid_moves:
            self.trusted_play(move)
            nodes += self._calculate_perft_sync(depth - 1)
            self.try_undo_last_move()
        return nodes

    def parallel_perft(self, depth: int) -> int:
        if depth == 0:
            return 1
        valid_moves = self.get_valid_moves()
        if depth == 1:
            return len(valid_moves)
        nodes = 0

        def count_for_move(move: Move) -> int:
            clone = self.clone()
            clone.trusted_play(move)
            return asyncio.run(clone.calculate_perft_async(depth - 1))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(count_for_move, move) for move in valid_moves]
            for future in futures:
                nodes += future.result()
        return nodes

    def get_board_metrics(self) -> BoardMetrics:
        board_metrics = BoardMetrics()
        board_metrics.board_state = self._board_state_cached()
        if self.game_in_progress:
            current_valid_moves = self.get_valid_moves()
            self._set_current_player_metrics(board_metrics, current_valid_moves)
            enemy_queen_neighbors = self._cached_enemy_queen_neighbors
            self._current_turn += 1
            next_valid_moves = self.get_valid_moves()
            self._set_current_player_metrics(board_metrics, next_valid_moves)
            self._current_turn -= 1
            self._cached_enemy_queen_neighbors = enemy_queen_neighbors
            self._cached_valid_moves = current_valid_moves
        return board_metrics

    def _set_current_player_metrics(self, board_metrics: BoardMetrics, move_set: MoveSet) -> None:
        start_piece = _W_QUEEN if (self._current_turn & 1) == 0 else _B_QUEEN
        end_piece = _B_QUEEN if (self._current_turn & 1) == 0 else _NUM_PIECE_NAMES
        for pn in range(start_piece, end_piece):
            piece_name = PieceName(pn)
            if Enums.piece_name_is_enabled_for_game_type(piece_name, self.game_type):
                piece_in_play = self.piece_in_play(piece_name)
                if piece_in_play:
                    board_metrics.pieces_in_play += 1
                    board_metrics[piece_name].in_play = 1
                else:
                    board_metrics.pieces_in_hand += 1
                    board_metrics[piece_name].in_play = 0
                is_pinned, noisy_count, quiet_count = self._is_pinned(piece_name, move_set)
                board_metrics[piece_name].is_pinned = 1 if is_pinned else 0
                board_metrics[piece_name].noisy_move_count = noisy_count
                board_metrics[piece_name].quiet_move_count = quiet_count
                board_metrics[piece_name].is_covered = 1 if (piece_in_play and not self._piece_is_on_top(piece_name)) else 0
                total, friendly, enemy = self._count_neighbors(piece_name)
                board_metrics[piece_name].friendly_neighbor_count = friendly
                board_metrics[piece_name].enemy_neighbor_count = enemy

    def _is_pinned(self, piece_name: PieceName, move_set: MoveSet) -> tuple[bool, int, int]:
        noisy_count = 0
        quiet_count = 0
        for move in move_set:
            if move.piece_name == piece_name:
                if self.is_noisy_move(move):
                    noisy_count += 1
                else:
                    quiet_count += 1
        return (noisy_count + quiet_count) == 0, noisy_count, quiet_count

    def clone(self) -> Board:
        board = Board(self.game_type)
        for item in self.board_history:
            board.trusted_play(item.move, item.move_string)
        return board

    def _get_valid_moves_for_piece(self, piece_name: int, move_set: MoveSet) -> None:
        if not Enums.piece_name_is_enabled_for_game_type(PieceName(piece_name), self.game_type):
            return
        if not self._placing_piece_in_order_packed(piece_name):
            return
        if self._current_turn == 0:
            if piece_name != _W_QUEEN:
                move_set.fast_add(Move(PieceName(piece_name), NULL_POSITION, ORIGIN_POSITION))
        elif self._current_turn == 1:
            if piece_name != _B_QUEEN:
                for d in range(_NUM_DIRECTIONS):
                    move_set.fast_add(Move(PieceName(piece_name), NULL_POSITION, ORIGIN_NEIGHBORS[d]))
        elif self.piece_in_hand_packed(piece_name):
            cpt = self.current_player_turn
            if cpt != 4 or (cpt == 4 and (self.current_turn_queen_in_play or (not self.current_turn_queen_in_play and Enums.get_bug_type(PieceName(piece_name)) == BugType.QueenBee))):
                self._calculate_valid_placements()
                for placement in self._cached_valid_placements:
                    move_set.fast_add(Move(PieceName(piece_name), NULL_POSITION, placement))
        elif piece_name != self._last_piece_moved and self.current_turn_queen_in_play and self._piece_is_on_top_packed(piece_name):
            if self.can_move_without_breaking_hive_packed(piece_name):
                bug_type = Enums.get_bug_type(PieceName(piece_name))
                if bug_type == BugType.QueenBee:
                    self._get_valid_slides_packed(piece_name, move_set, 1)
                elif bug_type == BugType.Spider:
                    self._get_valid_slides_packed(piece_name, move_set, 3)
                elif bug_type == BugType.Beetle:
                    self._get_valid_beetle_moves_packed(piece_name, move_set)
                elif bug_type == BugType.Grasshopper:
                    self._get_valid_grasshopper_moves_packed(piece_name, move_set)
                elif bug_type == BugType.SoldierAnt:
                    self._get_valid_slides_packed(piece_name, move_set, 0)
                elif bug_type == BugType.Mosquito:
                    self._get_valid_mosquito_moves_packed(piece_name, move_set, False)
                elif bug_type == BugType.Ladybug:
                    self._get_valid_ladybug_moves_packed(piece_name, move_set)
                elif bug_type == BugType.Pillbug:
                    new_moves = MoveSet()
                    self._get_valid_slides_packed(piece_name, new_moves, 1)
                    self._get_valid_pillbug_special_moves_packed(piece_name, new_moves)
                    for mv in new_moves:
                        move_set.add(mv)
            else:
                bug_type = Enums.get_bug_type(PieceName(piece_name))
                if bug_type == BugType.Mosquito:
                    self._get_valid_mosquito_moves_packed(piece_name, move_set, True)
                elif bug_type == BugType.Pillbug:
                    self._get_valid_pillbug_special_moves_packed(piece_name, move_set)

    def _calculate_valid_placements(self) -> None:
        if not self._cached_valid_placements_ready:
            self._cached_valid_placements.clear()
            start_piece = _W_QUEEN if (self._current_turn & 1) == 0 else _B_QUEEN
            end_piece = _B_QUEEN if (self._current_turn & 1) == 0 else _NUM_PIECE_NAMES
            current_color = self._current_turn & 1
            for pn in range(start_piece, end_piece):
                pos_packed = self._piece_positions[pn]
                if pos_packed == PACKED_NULL:
                    continue
                pos_q = unpack_q(pos_packed)
                pos_r = unpack_r(pos_packed)
                pos_stack = unpack_stack(pos_packed)
                if pos_stack != 0:
                    continue
                for d in range(_NUM_DIRECTIONS):
                    nq = pos_q + NEIGHBOR_DQ[d]
                    nr = pos_r + NEIGHBOR_DR[d]
                    neighbor_piece = self._get_piece_on_top_at_qr(nq, nr)
                    if neighbor_piece != -1:
                        if _get_color_int(neighbor_piece) != current_color:
                            d += 1
                    else:
                        original_piece_dir = (d + _HALF_DIRECTIONS) % _NUM_DIRECTIONS
                        valid_placement = True
                        for d2 in range(_NUM_DIRECTIONS):
                            if d2 != original_piece_dir:
                                sq = nq + NEIGHBOR_DQ[d2]
                                sr = nr + NEIGHBOR_DR[d2]
                                surrounding_piece = self._get_piece_on_top_at_qr(sq, sr)
                                if surrounding_piece != -1 and _get_color_int(surrounding_piece) != current_color:
                                    valid_placement = False
                                    break
                        if valid_placement:
                            self._cached_valid_placements.add(Position(nq, nr, 0))
            self._cached_valid_placements_ready = True

    def _get_valid_queen_bee_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_slides_packed(int(piece_name), move_set, 1)

    def _get_valid_spider_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_slides_packed(int(piece_name), move_set, 3)

    def _get_valid_beetle_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_beetle_moves_packed(int(piece_name), move_set)

    def _get_valid_grasshopper_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_grasshopper_moves_packed(int(piece_name), move_set)

    def _get_valid_soldier_ant_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_slides_packed(int(piece_name), move_set, 0)

    def _get_valid_mosquito_moves(self, piece_name: PieceName, move_set: MoveSet, special_ability_only: bool) -> None:
        self._get_valid_mosquito_moves_packed(int(piece_name), move_set, special_ability_only)

    def _get_valid_ladybug_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_ladybug_moves_packed(int(piece_name), move_set)

    def _get_valid_pillbug_basic_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_slides_packed(int(piece_name), move_set, 1)

    def _get_valid_pillbug_special_moves(self, piece_name: PieceName, move_set: MoveSet) -> None:
        self._get_valid_pillbug_special_moves_packed(int(piece_name), move_set)

    # --- Packed-int move generation functions ---

    def _get_valid_slides_packed(self, piece_name: int, move_set: MoveSet, fixed_range: int) -> None:
        starting_packed = self._piece_positions[piece_name]
        self._set_position_packed(piece_name, PACKED_NULL, False)
        if fixed_range > 0:
            self._get_valid_slides_fixed_packed(piece_name, move_set, starting_packed, starting_packed, starting_packed, fixed_range, fixed_range == 1)
        else:
            self._get_valid_slides_unbounded_packed(piece_name, move_set, starting_packed, starting_packed, starting_packed)
        self._set_position_packed(piece_name, starting_packed, False)

    def _get_valid_slides_unbounded_packed(self, piece_name: int, move_set: MoveSet,
                                            starting_packed: int, last_packed: int, current_packed: int) -> None:
        cur_q = unpack_q(current_packed)
        cur_r = unpack_r(current_packed)
        cur_stack = unpack_stack(current_packed)
        starting_q = unpack_q(starting_packed)
        starting_r = unpack_r(starting_packed)
        for d in range(_NUM_DIRECTIONS):
            nq = cur_q + NEIGHBOR_DQ[d]
            nr = cur_r + NEIGHBOR_DR[d]
            if nq == starting_q and nr == starting_r:
                continue
            slide_idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + cur_stack
            if self._piece_grid[slide_idx] != -1:
                continue
            # Check "must slide" rule: right and left of slide direction must differ
            right = (d + 1) % _NUM_DIRECTIONS
            left = (d + _NUM_DIRECTIONS - 1) % _NUM_DIRECTIONS
            right_q = cur_q + NEIGHBOR_DQ[right]
            right_r = cur_r + NEIGHBOR_DR[right]
            right_idx = ((right_q + HALF_BOARD) * _GRID_STRIDE_R) + ((right_r + HALF_BOARD) * BOARD_STACK_SIZE) + cur_stack
            left_q = cur_q + NEIGHBOR_DQ[left]
            left_r = cur_r + NEIGHBOR_DR[left]
            left_idx = ((left_q + HALF_BOARD) * _GRID_STRIDE_R) + ((left_r + HALF_BOARD) * BOARD_STACK_SIZE) + cur_stack
            if (self._piece_grid[right_idx] != -1) != (self._piece_grid[left_idx] != -1):
                slide_packed_full = pack_position(nq, nr, cur_stack)
                move = Move(PieceName(piece_name), self._position_from_packed(starting_packed), self._position_from_packed(slide_packed_full))
                if move_set.add(move):
                    self._get_valid_slides_unbounded_packed(piece_name, move_set, starting_packed, current_packed, slide_packed_full)

    def _get_valid_slides_fixed_packed(self, piece_name: int, move_set: MoveSet,
                                        starting_packed: int, last_packed: int, current_packed: int,
                                        remaining_slides: int, fast_add: bool) -> None:
        cur_q = unpack_q(current_packed)
        cur_r = unpack_r(current_packed)
        cur_stack = unpack_stack(current_packed)
        starting_q = unpack_q(starting_packed)
        starting_r = unpack_r(starting_packed)
        if remaining_slides == 0:
            move = Move(PieceName(piece_name), self._position_from_packed(starting_packed), self._position_from_packed(current_packed))
            if fast_add:
                move_set.fast_add(move)
            else:
                move_set.add(move)
            return
        for d in range(_NUM_DIRECTIONS):
            nq = cur_q + NEIGHBOR_DQ[d]
            nr = cur_r + NEIGHBOR_DR[d]
            if nq == starting_q and nr == starting_r:
                continue
            slide_idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + cur_stack
            if self._piece_grid[slide_idx] != -1:
                continue
            right = (d + 1) % _NUM_DIRECTIONS
            left = (d + _NUM_DIRECTIONS - 1) % _NUM_DIRECTIONS
            right_q = cur_q + NEIGHBOR_DQ[right]
            right_r = cur_r + NEIGHBOR_DR[right]
            right_idx = ((right_q + HALF_BOARD) * _GRID_STRIDE_R) + ((right_r + HALF_BOARD) * BOARD_STACK_SIZE) + cur_stack
            left_q = cur_q + NEIGHBOR_DQ[left]
            left_r = cur_r + NEIGHBOR_DR[left]
            left_idx = ((left_q + HALF_BOARD) * _GRID_STRIDE_R) + ((left_r + HALF_BOARD) * BOARD_STACK_SIZE) + cur_stack
            if (self._piece_grid[right_idx] != -1) != (self._piece_grid[left_idx] != -1):
                slide_packed_full = pack_position(nq, nr, cur_stack)
                self._get_valid_slides_fixed_packed(piece_name, move_set, starting_packed, current_packed, slide_packed_full, remaining_slides - 1, fast_add)

    def _get_valid_beetle_moves_packed(self, piece_name: int, move_set: MoveSet) -> None:
        pos_packed = self._piece_positions[piece_name]
        pos_q = unpack_q(pos_packed)
        pos_r = unpack_r(pos_packed)
        pos_stack = unpack_stack(pos_packed)
        current_height = pos_stack + 1
        for d in range(_NUM_DIRECTIONS):
            nq = pos_q + NEIGHBOR_DQ[d]
            nr = pos_r + NEIGHBOR_DR[d]
            top_neighbor = self._get_piece_on_top_at_qr(nq, nr)
            left_d = (d + _NUM_DIRECTIONS - 1) % _NUM_DIRECTIONS
            right_d = (d + 1) % _NUM_DIRECTIONS
            lq = pos_q + NEIGHBOR_DQ[left_d]
            lr = pos_r + NEIGHBOR_DR[left_d]
            top_left = self._get_piece_on_top_at_qr(lq, lr)
            rq = pos_q + NEIGHBOR_DQ[right_d]
            rr = pos_r + NEIGHBOR_DR[right_d]
            top_right = self._get_piece_on_top_at_qr(rq, rr)
            destination_height = (unpack_stack(self._piece_positions[top_neighbor]) + 1) if top_neighbor != -1 else 0
            top_left_height = (unpack_stack(self._piece_positions[top_left]) + 1) if top_left != -1 else 0
            top_right_height = (unpack_stack(self._piece_positions[top_right]) + 1) if top_right != -1 else 0
            if not (current_height == 0 and destination_height == 0 and top_left_height == 0 and top_right_height == 0):
                if not (destination_height < top_left_height and destination_height < top_right_height and current_height < top_left_height and current_height < top_right_height):
                    move = Move(
                        PieceName(piece_name),
                        self._position_from_packed(pos_packed),
                        Position(nq, nr, destination_height),
                    )
                    move_set.fast_add(move)

    def _get_valid_grasshopper_moves_packed(self, piece_name: int, move_set: MoveSet) -> None:
        starting_packed = self._piece_positions[piece_name]
        starting_q = unpack_q(starting_packed)
        starting_r = unpack_r(starting_packed)
        starting_stack = unpack_stack(starting_packed)
        for d in range(_NUM_DIRECTIONS):
            nq = starting_q + NEIGHBOR_DQ[d]
            nr = starting_r + NEIGHBOR_DR[d]
            distance = 0
            while True:
                cur_idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + starting_stack
                if self._piece_grid[cur_idx] == -1:
                    break
                nq += NEIGHBOR_DQ[d]
                nr += NEIGHBOR_DR[d]
                distance += 1
            if distance > 0:
                move = Move(
                    PieceName(piece_name),
                    self._position_from_packed(starting_packed),
                    Position(nq, nr, 0),
                )
                move_set.fast_add(move)

    def _get_valid_mosquito_moves_packed(self, piece_name: int, move_set: MoveSet, special_ability_only: bool) -> None:
        pos_packed = self._piece_positions[piece_name]
        pos_stack = unpack_stack(pos_packed)
        pos_q = unpack_q(pos_packed)
        pos_r = unpack_r(pos_packed)
        if pos_stack > 0 and not special_ability_only:
            self._get_valid_beetle_moves_packed(piece_name, move_set)
            return
        bug_types_evaluated = [False] * int(BugType.NumBugTypes)
        for d in range(_NUM_DIRECTIONS):
            nq = pos_q + NEIGHBOR_DQ[d]
            nr = pos_r + NEIGHBOR_DR[d]
            neighbor_pn = self._get_piece_on_top_at_qr(nq, nr)
            if neighbor_pn == -1:
                continue
            neighbor_bug_type = Enums.get_bug_type(PieceName(neighbor_pn))
            if bug_types_evaluated[int(neighbor_bug_type)]:
                continue
            new_moves = MoveSet()
            if special_ability_only:
                if neighbor_bug_type == BugType.Pillbug:
                    self._get_valid_pillbug_special_moves_packed(piece_name, new_moves)
            else:
                if neighbor_bug_type == BugType.QueenBee:
                    self._get_valid_slides_packed(piece_name, new_moves, 1)
                elif neighbor_bug_type == BugType.Spider:
                    self._get_valid_slides_packed(piece_name, new_moves, 3)
                elif neighbor_bug_type == BugType.Beetle:
                    self._get_valid_beetle_moves_packed(piece_name, new_moves)
                elif neighbor_bug_type == BugType.Grasshopper:
                    self._get_valid_grasshopper_moves_packed(piece_name, new_moves)
                elif neighbor_bug_type == BugType.SoldierAnt:
                    self._get_valid_slides_packed(piece_name, new_moves, 0)
                elif neighbor_bug_type == BugType.Ladybug:
                    self._get_valid_ladybug_moves_packed(piece_name, new_moves)
                elif neighbor_bug_type == BugType.Pillbug:
                    self._get_valid_slides_packed(piece_name, new_moves, 1)
                    self._get_valid_pillbug_special_moves_packed(piece_name, new_moves)
            for mv in new_moves:
                move_set.add(mv)
            bug_types_evaluated[int(neighbor_bug_type)] = True

    def _get_valid_ladybug_moves_packed(self, piece_name: int, move_set: MoveSet) -> None:
        starting_packed = self._piece_positions[piece_name]
        starting_pos = self._position_from_packed(starting_packed)
        first_moves = MoveSet()
        self._get_valid_beetle_moves_packed(piece_name, first_moves)
        for first_move in first_moves:
            first_dest = first_move.destination
            if first_dest.stack > 0:
                first_dest_packed = pack_position(first_dest.q, first_dest.r, first_dest.stack)
                self._set_position_packed(piece_name, first_dest_packed, False)
                second_moves = MoveSet()
                self._get_valid_beetle_moves_packed(piece_name, second_moves)
                for second_move in second_moves:
                    second_dest = second_move.destination
                    if second_dest.stack > 0:
                        second_dest_packed = pack_position(second_dest.q, second_dest.r, second_dest.stack)
                        self._set_position_packed(piece_name, second_dest_packed, False)
                        third_moves = MoveSet()
                        self._get_valid_beetle_moves_packed(piece_name, third_moves)
                        for third_move in third_moves:
                            if third_move.destination.stack == 0 and third_move.destination != starting_pos:
                                final_move = Move(PieceName(piece_name), starting_pos, third_move.destination)
                                move_set.add(final_move)
                        self._set_position_packed(piece_name, first_dest_packed, False)
                self._set_position_packed(piece_name, starting_packed, False)

    def _get_valid_pillbug_special_moves_packed(self, piece_name: int, move_set: MoveSet) -> None:
        pos_packed = self._piece_positions[piece_name]
        pos_q = unpack_q(pos_packed)
        pos_r = unpack_r(pos_packed)
        pos_stack = unpack_stack(pos_packed)
        above_target_packed = pack_position(pos_q, pos_r, pos_stack + 1)
        for d in range(_NUM_DIRECTIONS):
            nq = pos_q + NEIGHBOR_DQ[d]
            nr = pos_r + NEIGHBOR_DR[d]
            neighbor_idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + pos_stack
            neighbor_pn = self._piece_grid[neighbor_idx]
            if neighbor_pn == -1 or neighbor_pn == self._last_piece_moved:
                continue
            # Check no piece above
            above_idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + (pos_stack + 1)
            if self._piece_grid[above_idx] != -1:
                continue
            if not self.can_move_without_breaking_hive_packed(neighbor_pn):
                continue
            neighbor_packed = pack_position(nq, nr, pos_stack)
            first_move = Move(PieceName(neighbor_pn), self._position_from_packed(neighbor_packed), self._position_from_packed(above_target_packed))
            first_moves = MoveSet()
            self._get_valid_beetle_moves_packed(neighbor_pn, first_moves)
            if not first_moves.contains(first_move):
                continue
            self._set_position_packed(neighbor_pn, above_target_packed, False)
            second_moves = MoveSet()
            self._get_valid_beetle_moves_packed(neighbor_pn, second_moves)
            for second_move in second_moves:
                if second_move.destination.stack == 0 and second_move.destination != self._position_from_packed(neighbor_packed):
                    final_move = Move(PieceName(neighbor_pn), self._position_from_packed(neighbor_packed), second_move.destination)
                    move_set.add(final_move)
            self._set_position_packed(neighbor_pn, neighbor_packed, False)

    def trusted_play(self, move: Move, move_str: str = "") -> None:
        self.board_history.add(move, move_str)
        if move != PASS_MOVE:
            self._set_position(move.piece_name, move.destination, True)
        self.current_turn += 1
        self._last_piece_moved = int(move.piece_name)

    def _placing_piece_in_order(self, piece_name: PieceName) -> bool:
        return self._placing_piece_in_order_packed(int(piece_name))

    def _placing_piece_in_order_packed(self, piece_name: int) -> bool:
        if self._piece_positions[piece_name] == PACKED_NULL:
            if piece_name in _ORDERED_PIECE_NAMES:
                return self._piece_positions[piece_name - 1] != PACKED_NULL
        return True

    def get_position(self, piece_name: PieceName) -> Position:
        return self._position_from_packed(self._piece_positions[int(piece_name)])

    def _position_from_packed(self, p: int) -> Position:
        return Position(unpack_q(p), unpack_r(p), unpack_stack(p))

    def _set_position(self, piece_name: PieceName, position: Position, update_zobrist: bool) -> None:
        if position == NULL_POSITION:
            self._set_position_packed(int(piece_name), PACKED_NULL, update_zobrist)
        else:
            self._set_position_packed(int(piece_name), pack_position(position.q, position.r, position.stack), update_zobrist)

    def _set_position_packed(self, piece_name: int, packed: int, update_zobrist: bool) -> None:
        old = self._piece_positions[piece_name]
        if old != PACKED_NULL:
            if update_zobrist:
                self._zobrist_hash.toggle_piece(PieceName(piece_name), self._position_from_packed(old))
            old_q = unpack_q(old)
            old_r = unpack_r(old)
            old_stack = unpack_stack(old)
            self._piece_grid[((old_q + HALF_BOARD) * _GRID_STRIDE_R) + ((old_r + HALF_BOARD) * BOARD_STACK_SIZE) + old_stack] = -1
        self._piece_positions[piece_name] = packed
        if packed != PACKED_NULL:
            if update_zobrist:
                self._zobrist_hash.toggle_piece(PieceName(piece_name), self._position_from_packed(packed))
            new_q = unpack_q(packed)
            new_r = unpack_r(packed)
            new_stack = unpack_stack(packed)
            self._piece_grid[((new_q + HALF_BOARD) * _GRID_STRIDE_R) + ((new_r + HALF_BOARD) * BOARD_STACK_SIZE) + new_stack] = piece_name

    def _get_piece_at(self, position: Position) -> PieceName:
        if position == NULL_POSITION:
            return PieceName.INVALID
        return PieceName(self._piece_grid[((position.q + HALF_BOARD) * _GRID_STRIDE_R) + ((position.r + HALF_BOARD) * BOARD_STACK_SIZE) + position.stack])

    def _get_piece_at_dir(self, position: Position, direction: Direction) -> PieceName:
        idx = ((position.q + HALF_BOARD + NEIGHBOR_DQ[direction.value]) * _GRID_STRIDE_R) + ((position.r + HALF_BOARD + NEIGHBOR_DR[direction.value]) * BOARD_STACK_SIZE) + (position.stack + NEIGHBOR_DSTACK[direction.value])
        return PieceName(self._piece_grid[idx])

    def get_piece_on_top_at(self, position: Position) -> PieceName:
        if position == NULL_POSITION:
            return PieceName.INVALID
        return PieceName(self._get_piece_on_top_at_qr(position.q, position.r))

    def _get_piece_on_top_at_qr(self, q: int, r: int) -> int:
        col_idx = ((q + HALF_BOARD) * _GRID_STRIDE_R) + ((r + HALF_BOARD) * BOARD_STACK_SIZE)
        top = -1
        for stack in range(BOARD_STACK_SIZE):
            p = self._piece_grid[col_idx + stack]
            if p == -1:
                break
            top = p
        return top

    def _has_piece_at(self, position: Position) -> bool:
        if position == NULL_POSITION:
            return False
        return self._piece_grid[((position.q + HALF_BOARD) * _GRID_STRIDE_R) + ((position.r + HALF_BOARD) * BOARD_STACK_SIZE) + position.stack] != -1

    def _has_piece_at_dir(self, position: Position, direction: Direction) -> bool:
        idx = ((position.q + HALF_BOARD + NEIGHBOR_DQ[direction.value]) * _GRID_STRIDE_R) + ((position.r + HALF_BOARD + NEIGHBOR_DR[direction.value]) * BOARD_STACK_SIZE) + (position.stack + NEIGHBOR_DSTACK[direction.value])
        return self._piece_grid[idx] != -1

    def piece_in_hand(self, piece_name: PieceName) -> bool:
        return self._piece_positions[int(piece_name)] == PACKED_NULL

    def piece_in_hand_packed(self, piece_name: int) -> bool:
        return self._piece_positions[piece_name] == PACKED_NULL

    def piece_in_play(self, piece_name: PieceName) -> bool:
        return self._piece_positions[int(piece_name)] != PACKED_NULL

    def _piece_is_on_top(self, piece_name: PieceName) -> bool:
        return self._piece_is_on_top_packed(int(piece_name))

    def _piece_is_on_top_packed(self, piece_name: int) -> bool:
        pos_packed = self._piece_positions[piece_name]
        if pos_packed == PACKED_NULL:
            return False
        q = unpack_q(pos_packed)
        r = unpack_r(pos_packed)
        stack = unpack_stack(pos_packed)
        return self._piece_grid[((q + HALF_BOARD) * _GRID_STRIDE_R) + ((r + HALF_BOARD) * BOARD_STACK_SIZE) + (stack + 1)] == -1

    def can_move_without_breaking_hive(self, piece_name: PieceName) -> bool:
        return self.can_move_without_breaking_hive_packed(int(piece_name))

    def can_move_without_breaking_hive_packed(self, piece_name: int) -> bool:
        pos_packed = self._piece_positions[piece_name]
        if pos_packed == PACKED_NULL:
            return True
        pos_stack = unpack_stack(pos_packed)
        if pos_stack == 0:
            pos_q = unpack_q(pos_packed)
            pos_r = unpack_r(pos_packed)
            gaps = 0
            last_has_piece = None
            for d in range(_NUM_DIRECTIONS):
                nq = pos_q + NEIGHBOR_DQ[d]
                nr = pos_r + NEIGHBOR_DR[d]
                idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + pos_stack
                has_piece = self._piece_grid[idx] != -1
                if last_has_piece is not None and last_has_piece != has_piece:
                    gaps += 1
                    if gaps > 2:
                        break
                last_has_piece = has_piece
            if gaps <= 2:
                return True
            self._set_position_packed(piece_name, PACKED_NULL, False)
            is_one_hive = self.is_one_hive()
            self._set_position_packed(piece_name, pos_packed, False)
            return is_one_hive
        return True

    def is_one_hive(self) -> bool:
        pieces_visited = 0
        starting_piece = -1
        part_of_hive = self._part_of_hive
        for pn in range(_NUM_PIECE_NAMES):
            if self._piece_positions[pn] == PACKED_NULL:
                part_of_hive[pn] = True
                pieces_visited += 1
            else:
                part_of_hive[pn] = False
                pos_packed = self._piece_positions[pn]
                if starting_piece == -1 and unpack_stack(pos_packed) == 0:
                    starting_piece = pn
                    part_of_hive[pn] = True
                    pieces_visited += 1
        if starting_piece != -1 and pieces_visited < _NUM_PIECE_NAMES:
            self._pieces_to_look_at.clear()
            self._pieces_to_look_at.append(starting_piece)
            while self._pieces_to_look_at:
                current_piece = self._pieces_to_look_at.popleft()
                cur_packed = self._piece_positions[current_piece]
                cur_q = unpack_q(cur_packed)
                cur_r = unpack_r(cur_packed)
                cur_stack = unpack_stack(cur_packed)
                for d in range(_NUM_DIRECTIONS):
                    nq = cur_q + NEIGHBOR_DQ[d]
                    nr = cur_r + NEIGHBOR_DR[d]
                    idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + cur_stack
                    neighbor_piece = self._piece_grid[idx]
                    if neighbor_piece != -1 and not part_of_hive[neighbor_piece]:
                        self._pieces_to_look_at.append(neighbor_piece)
                        part_of_hive[neighbor_piece] = True
                        pieces_visited += 1
                cur_col_idx = ((cur_q + HALF_BOARD) * _GRID_STRIDE_R) + ((cur_r + HALF_BOARD) * BOARD_STACK_SIZE)
                stack = cur_stack + 1
                while stack < BOARD_STACK_SIZE:
                    piece_above = self._piece_grid[cur_col_idx + stack]
                    if piece_above == -1:
                        break
                    part_of_hive[piece_above] = True
                    pieces_visited += 1
                    stack += 1
        return pieces_visited == _NUM_PIECE_NAMES

    def _count_neighbors(self, piece_name: PieceName) -> tuple[int, int, int]:
        return self._count_neighbors_full(int(piece_name))

    def _count_neighbors_packed(self, piece_name: int) -> int:
        total, _, _ = self._count_neighbors_full(piece_name)
        return total

    def _count_neighbors_full(self, piece_name: int) -> tuple[int, int, int]:
        pos_packed = self._piece_positions[piece_name]
        total = 0
        friendly = 0
        enemy = 0
        if pos_packed != PACKED_NULL:
            piece_color = _get_color_int(piece_name)
            pos_q = unpack_q(pos_packed)
            pos_r = unpack_r(pos_packed)
            pos_stack = unpack_stack(pos_packed)
            for d in range(_NUM_DIRECTIONS):
                nq = pos_q + NEIGHBOR_DQ[d]
                nr = pos_r + NEIGHBOR_DR[d]
                idx = ((nq + HALF_BOARD) * _GRID_STRIDE_R) + ((nr + HALF_BOARD) * BOARD_STACK_SIZE) + pos_stack
                neighbor = self._piece_grid[idx]
                if neighbor != -1:
                    total += 1
                    if piece_color == _get_color_int(neighbor):
                        friendly += 1
                    else:
                        enemy += 1
        return total, friendly, enemy

    def _reset_state(self) -> None:
        self._queen_surrounded_dirty = True

    def _reset_caches(self) -> None:
        self._cached_valid_placements_ready = False
        self._cached_valid_moves = None
        self._cached_enemy_queen_neighbors = None
