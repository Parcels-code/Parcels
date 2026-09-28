from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable

import numpy as np


class ByteBoundedLRUCache:
    """LRU cache bounded by total stored bytes.

    Keys are hashable (typically chunk coordinate tuples).
    Values are numpy arrays whose .nbytes drives eviction.
    """

    def __init__(self, max_bytes: int) -> None:
        assert max_bytes > 0
        self._max_bytes = max_bytes
        self._cache: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._current_bytes = 0

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    def get(self, key: Hashable) -> np.ndarray | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Hashable, value: np.ndarray) -> None:
        nbytes = value.nbytes
        if nbytes > self._max_bytes:
            return
        # Remove existing entry if present (will be re-inserted at end)
        if key in self._cache:
            self._current_bytes -= self._cache.pop(key).nbytes
        # Evict LRU entries until there's room
        while self._current_bytes + nbytes > self._max_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= evicted.nbytes
        self._cache[key] = value
        self._current_bytes += nbytes

    def clear(self) -> None:
        self._cache.clear()
        self._current_bytes = 0
