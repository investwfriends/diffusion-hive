from __future__ import annotations
from collections.abc import Iterator
from typing import TYPE_CHECKING, Generic, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from mzinga.core.move import Move


class FastSet(Generic[T]):
    """List-backed set optimized for small collections (typically <30 items)."""

    def __init__(self, items: list[T] | None = None):
        self._items: list[T] = list(items) if items else []

    @property
    def count(self) -> int:
        return len(self._items)

    def contains(self, item: T) -> bool:
        for i in range(len(self._items) - 1, -1, -1):
            if self._items[i] == item:
                return True
        return False

    def add(self, item: T) -> bool:
        if self.contains(item):
            return False
        self.fast_add(item)
        return True

    def fast_add(self, item: T) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    def validate(self) -> None:
        if len(set(self._items)) != len(self._items):
            raise ValueError("FastSet contains duplicates.")

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __repr__(self) -> str:
        return f"FastSet({self._items})"


class MoveSet(FastSet["Move"]):
    def contains_piece_name(self, piece_name) -> bool:
        for item in self._items:
            if item.piece_name == piece_name:
                return True
        return False

    @staticmethod
    def parse_move_list(board, move_list: str, separator: str = ";") -> MoveSet:
        from mzinga.core.board import Board

        moves = MoveSet()
        for input_move_str in move_list.split(separator):
            move, _ = board.parse_move(input_move_str)
            if move is None:
                raise ValueError(f"Unable to parse '{input_move_str}'.")
            moves.add(move)
        return moves
