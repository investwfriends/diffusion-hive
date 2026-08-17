from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Optional

from mzinga.core.move import Move


@dataclass(frozen=True)
class BoardHistoryItem:
    move: Move
    move_string: str
    
    def __str__(self) -> str:
        return self.move_string


class BoardHistory:
    def __init__(self):
        self._items: list[BoardHistoryItem] = []
    
    @property
    def count(self) -> int:
        return len(self._items)
    
    @property
    def last_move(self) -> Optional[Move]:
        if self._items:
            return self._items[-1].move
        return None
    
    def add(self, move: Move, move_str: str) -> None:
        self._items.append(BoardHistoryItem(move, move_str))
    
    def undo_last(self) -> None:
        if self._items:
            self._items.pop()
    
    def __getitem__(self, index: int) -> BoardHistoryItem:
        return self._items[index]
    
    def __iter__(self) -> Iterator[BoardHistoryItem]:
        return iter(self._items)
    
    def __len__(self) -> int:
        return len(self._items)
