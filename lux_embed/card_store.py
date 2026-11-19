"""Built cards, stored so that embedding needs nothing but them and a GPU.

``cards --out cards.lmdb`` builds every card of a scope on CPUs (records + labels + Stage 1 facets,
in a pool of worker processes) and writes them here; ``embed --cards cards.lmdb`` then reads them
back in the same order, so the GPU job needs no record LMDB, label store or facets, and never waits
for card building. The cards are model-independent (each encoder's document format is applied at
encoding time), so one build serves every model.

Layout of ``cards.lmdb`` (an LMDB directory):

* sub-DB ``cards:<scope>`` - key = 16-byte UUID (the label store's order, so a scope is written and
  read in key order), value = zlib JSON ``[iri, title, text]``.
* sub-DB ``meta`` - key = scope, value = JSON ``{"count", "cursor", "complete", "limit"}``; ``cursor``
  is the UUID of the last committed card, which is where a resumed build starts.

One writer per file at a time (LMDB's rule): give each concurrently running scope its own file.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterator

import lmdb

from .records import uuid_bytes, uuid_str

MAP_SIZE = 1 << 40


def pack(iri: str, title: str, text: str) -> tuple[bytes, bytes]:
    """The ``(key, value)`` row of one card."""
    value = zlib.compress(json.dumps([iri, title, text], ensure_ascii=False).encode(), 6)
    return uuid_bytes(iri), value


def unpack(value: bytes) -> tuple[str, str, str]:
    iri, title, text = json.loads(zlib.decompress(value))
    return iri, title, text


class CardWriter:
    """Appends one scope's cards in key order, a batch per transaction, resumably."""

    def __init__(self, path: str, scope: str, restart: bool = False):
        self.scope = scope
        self.env = lmdb.open(path, map_size=MAP_SIZE, max_dbs=8, subdir=True)
        self.db, self.meta_db = self.env.open_db(f"cards:{scope}".encode()), self.env.open_db(b"meta")
        with self.env.begin(write=True) as txn:
            raw = txn.get(scope.encode(), db=self.meta_db)
            if raw is not None and restart:
                txn.drop(self.db, delete=False)
                raw = None
        self.meta = json.loads(raw) if raw else {"scope": scope, "count": 0, "cursor": None, "complete": False}

    @property
    def cursor(self) -> str | None:
        return self.meta["cursor"]

    @property
    def complete(self) -> bool:
        return self.meta["complete"]

    def append(self, rows: list[tuple[bytes, bytes]]) -> None:
        if not rows:
            return
        with self.env.begin(write=True) as txn:
            # append=True: keys must come after everything stored, which is also the order guard
            consumed, added = txn.cursor(db=self.db).putmulti(rows, append=True)
            if added != len(rows):
                raise RuntimeError(f"{self.scope}: cards arrived out of key order")
            self.meta["count"] += len(rows)
            self.meta["cursor"] = uuid_str(rows[-1][0])
            txn.put(self.scope.encode(), json.dumps(self.meta).encode(), db=self.meta_db)

    def finish(self, limit: int = 0) -> None:
        self.meta["complete"] = True
        if limit:
            self.meta["limit"] = limit
        with self.env.begin(write=True) as txn:
            txn.put(self.scope.encode(), json.dumps(self.meta).encode(), db=self.meta_db)
        self.env.sync(True)

    def close(self) -> None:
        self.env.close()


class CardStore:
    """Read side: one scope's cards in key order."""

    def __init__(self, path: str, scope: str):
        self.scope = scope
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=True, max_dbs=8, subdir=True)
        try:
            meta_db = self.env.open_db(b"meta", create=False)
            with self.env.begin(db=meta_db) as txn:
                raw = txn.get(scope.encode())
            self.db = self.env.open_db(f"cards:{scope}".encode(), create=False) if raw else None
        except lmdb.NotFoundError:
            raw, self.db = None, None
        self.meta = json.loads(raw) if raw else None

    def batches(self, after: str | None, size: int, lower: bytes | None = None,
                bound: bytes | None = None) -> Iterator[list[tuple[str, str, str]]]:
        """``(iri, title, text)`` lists of ``size`` cards after the UUID ``after``, within ``[lower, bound)``."""
        with self.env.begin(db=self.db, buffers=True) as txn:
            cursor = txn.cursor()
            skip = uuid_bytes(after) if after else None
            start = max(filter(None, (skip, lower)), default=None)
            if not (cursor.set_range(start) if start else cursor.first()):
                return
            batch = []
            for key, value in cursor:
                key = bytes(key)
                if bound is not None and key >= bound:
                    break
                if key == skip:
                    continue
                batch.append(unpack(value))
                if len(batch) == size:
                    yield batch
                    batch = []
            if batch:
                yield batch
