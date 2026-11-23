"""A thread-safe LRU bounded by bytes, shared by everything the sidecar caches.

One budget for all of it, because the entries compete for the same scarce memory: rendered
responses, ranked hits and encoded query vectors. Each entry carries a tag (the store generations
it was computed against), and a lookup with a different tag is a miss that drops the entry, so a
new graph generation invalidates what was cached before it without a sweep.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable


class ByteLRU:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.hits = self.misses = 0
        self._entries: OrderedDict[Hashable, tuple[Hashable, object, int]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable, tag: Hashable = None):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] != tag:
                if entry is not None:  # computed against an older generation
                    self._drop(key)
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[1]

    def put(self, key: Hashable, value, nbytes: int, tag: Hashable = None) -> None:
        if nbytes > self.capacity:
            return
        with self._lock:
            if key in self._entries:
                self._drop(key)
            self._entries[key] = (tag, value, nbytes)
            self.size += nbytes
            while self.size > self.capacity:
                self._drop(next(iter(self._entries)))

    def _drop(self, key: Hashable) -> None:
        self.size -= self._entries.pop(key)[2]

    def describe(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "bytes": self.size, "capacity": self.capacity,
                    "hits": self.hits, "misses": self.misses}
