from __future__ import annotations

import os
import struct

from mzinga.core.enums import PieceName
from mzinga.core.position import Position, BOARD_SIZE, BOARD_STACK_SIZE, HALF_BOARD


def _rand64_array(n: int) -> list[int]:
    """Generate n random 64-bit ints using bulk os.urandom.

    Each call to random.getrandbits(64) is a Python C function call
    (~600ns). Doing 4M of them takes ~2.5s. Reading N*8 bytes from
    os.urandom and unpacking them in bulk is faster.
    """
    if n <= 0:
        return []
    raw = os.urandom(n * 8)
    return list(struct.unpack(f"<{n}Q", raw))


class ZobristHash:
    _next = 1
    _hash_part_by_turn_color: int = 0
    _hash_part_by_last_moved_piece: list[int] = []
    _hash_part_by_position: list[list[list[list[int]]]] = []

    @staticmethod
    def _init_statics():
        if ZobristHash._hash_part_by_position:
            return
        n_pieces = int(PieceName.NumPieceNames)
        board_size = BOARD_SIZE
        board_stack = BOARD_STACK_SIZE + 1
        position_count = n_pieces * board_size * board_size * board_stack
        total = 1 + n_pieces + position_count
        all_rand = _rand64_array(total)
        # Build the 4D list by slicing the flat array. The slice is
        # amortized O(n) per level, total O(n) for the whole reshape.
        idx = 0
        ZobristHash._next = 1
        ZobristHash._hash_part_by_turn_color = all_rand[idx]; idx += 1
        ZobristHash._hash_part_by_last_moved_piece = all_rand[idx:idx + n_pieces]; idx += n_pieces
        # Build innermost level (1 board_stack-sized slice) for each
        # (pn, q, r) using a single shared list of slices.
        flat_pos = all_rand[idx:]
        stride = board_size * board_size * board_stack
        per_q = board_size * board_stack
        per_r = board_stack
        per_p = per_q * board_size
        # Avoid nested list comprehension: use a single for-loop with
        # pre-allocated lists, which is ~3x faster than list-of-lists
        # in this size regime on CPython 3.13.
        result: list = [None] * n_pieces
        for pn in range(n_pieces):
            piece_grid: list = [None] * board_size
            base_pn = pn * stride
            for q in range(board_size):
                base_q = base_pn + q * per_q
                row: list = [None] * board_size
                for r in range(board_size):
                    base = base_q + r * per_r
                    row[r] = flat_pos[base:base + board_stack]
                piece_grid[q] = row
            result[pn] = piece_grid
        ZobristHash._hash_part_by_position = result

    def __init__(self):
        self._init_statics()
        self.value: int = EMPTY_BOARD

    def toggle_piece(self, piece_name: PieceName, position: Position) -> None:
        self.value ^= self._hash_part_by_position[
            int(piece_name)
        ][HALF_BOARD + position.q][HALF_BOARD + position.r][position.stack + 1]

    def toggle_last_moved_piece(self, piece_name: PieceName) -> None:
        if piece_name != PieceName.INVALID:
            self.value ^= self._hash_part_by_last_moved_piece[int(piece_name)]

    def toggle_turn(self) -> None:
        self.value ^= self._hash_part_by_turn_color


EMPTY_BOARD: int = 0
