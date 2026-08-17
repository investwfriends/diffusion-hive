"""Module-level helpers for fast packed-int position arithmetic.

The public `Position` class is unchanged. Internally, `Board` stores piece
positions as packed ints and indexes a flat grid for O(1) lookups.
"""
from typing import NamedTuple

from mzinga.core.enums import Direction


# Encoding: 21 bits total.
# bits  0-7:   q + 128
# bits  8-15:  r + 128
# bits 16-20: stack + 2  (so stack=-1 -> 1, stack=0 -> 2, stack=7 -> 9)
_STACK_OFFSET = 2
_STACK_BITS = 5
_STACK_MASK = (1 << _STACK_BITS) - 1
_COORD_BITS = 8
_COORD_MASK = (1 << _COORD_BITS) - 1
_COORD_OFFSET = 128
_STACK_SHIFT = 2 * _COORD_BITS

_NULL_STACK_KEY = 1  # stack=-1 maps to key 1, never matches a real piece
_EMPTY_STACK_KEY = 2  # stack=0 maps to key 2


def pack_position(q: int, r: int, stack: int) -> int:
    return (
        ((q + _COORD_OFFSET) & _COORD_MASK)
        | (((r + _COORD_OFFSET) & _COORD_MASK) << _COORD_BITS)
        | (((stack + _STACK_OFFSET) & _STACK_MASK) << _STACK_SHIFT)
    )


def unpack_q(p: int) -> int:
    return (p & _COORD_MASK) - _COORD_OFFSET


def unpack_r(p: int) -> int:
    return ((p >> _COORD_BITS) & _COORD_MASK) - _COORD_OFFSET


def unpack_stack(p: int) -> int:
    return ((p >> _STACK_SHIFT) & _STACK_MASK) - _STACK_OFFSET


def unpack_position(p: int) -> tuple[int, int, int]:
    return unpack_q(p), unpack_r(p), unpack_stack(p)


# Pre-pack origin and null sentinel positions to avoid the call overhead in hot code.
PACKED_ORIGIN = pack_position(0, 0, 0)
PACKED_NULL = pack_position(0, 0, -1)


# Pre-computed (q, r, stack) deltas indexed by Direction.value, in unpacked form
# for direct use in array math. Note Above is 6 (Direction.Above) with dq=0, dr=0, dstack=1.
NEIGHBOR_DQ = (0, 1, 1, 0, -1, -1, 0)
NEIGHBOR_DR = (-1, -1, 0, 1, 1, 0, 0)
NEIGHBOR_DSTACK = (0, 0, 0, 0, 0, 0, 1)
NUM_NEIGHBOR_DIRECTIONS = 6  # Direction.NumDirections


def left_of_idx(d: int) -> int:
    return (d + NUM_NEIGHBOR_DIRECTIONS - 1) % NUM_NEIGHBOR_DIRECTIONS


def right_of_idx(d: int) -> int:
    return (d + 1) % NUM_NEIGHBOR_DIRECTIONS


class Position(NamedTuple):
    q: int
    r: int
    stack: int

    def get_neighbor_at(self, direction: Direction) -> "Position":
        return Position(
            self.q + NEIGHBOR_DQ[direction.value],
            self.r + NEIGHBOR_DR[direction.value],
            self.stack + NEIGHBOR_DSTACK[direction.value],
        )

    def get_above(self) -> "Position":
        return Position(self.q, self.r, self.stack + 1)

    def get_below(self) -> "Position":
        return Position(self.q, self.r, self.stack - 1)

    def get_bottom(self) -> "Position":
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
