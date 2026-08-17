from __future__ import annotations
import threading
from collections import OrderedDict
from collections.abc import Callable


class FixedCache[K, V]:
    """Thread-safe LRU cache with fixed capacity and optional replace predicate."""

    def __init__(self, capacity: int = 1024, replace_predicate: Callable[[V, V], bool] | None = None):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self._capacity = capacity
        self._replace_predicate = replace_predicate
        self._dict: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._dict)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def usage(self) -> float:
        with self._lock:
            return len(self._dict) / self._capacity if self._capacity > 0 else 0.0

    def store(self, key: K, new_entry: V) -> None:
        with self._lock:
            if key in self._dict:
                existing = self._dict[key]
                if self._replace_predicate is None or self._replace_predicate(existing, new_entry):
                    self._dict[key] = new_entry
                    self._dict.move_to_end(key)
            else:
                if len(self._dict) >= self._capacity:
                    self._dict.popitem(last=False)
                self._dict[key] = new_entry

    def try_lookup(self, key: K) -> V | None:
        with self._lock:
            if key in self._dict:
                return self._dict[key]
            return None

    def clear(self) -> None:
        with self._lock:
            self._dict.clear()

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._dict

    def __repr__(self) -> str:
        with self._lock:
            return f"FixedCache({len(self._dict)}/{self._capacity})"
