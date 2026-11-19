"""Pass 1: a compact UUID -> (class, primary name) store over every record.

References in LUX records carry no usable labels, so every reference a card mentions, and every
facet value Stage 1 returns, is resolved here. The class column doubles as an index, so one record
class can be enumerated without decompressing the whole record store.

Layout: LMDB sub-DB ``labels``, key = 16-byte UUID (byte order == hex string order), value =
``"<Class>\\t<primary name>"`` in UTF-8.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Iterator
from functools import lru_cache

import lmdb

from .linkedart import primary_name
from .records import SCOPE_OF_CLASS, RecordStore, available_cpus, uuid_bytes, uuid_str

MAX_LABEL_CHARS = 200

_worker_store: RecordStore | None = None


def _init_worker(path: str) -> None:
    global _worker_store
    _worker_store = RecordStore(path, readahead=True)  # a sequential scan


def _label_partition(bounds: tuple[bytes | None, bytes | None]) -> list[tuple[bytes, bytes]]:
    out = []
    for uuid, record in _worker_store.scan(*bounds):
        cls = record.get("type", "")
        name = (primary_name(record) or "")[:MAX_LABEL_CHARS].replace("\t", " ")
        out.append((uuid_bytes(uuid), f"{cls}\t{name}".encode()))
    out.sort()
    return out


def build_labels(records_path: str, out_path: str, workers: int = 0, partitions: int = 4096,
                 report_every: float = 30.0) -> int:
    """Scans every record in parallel and writes the label store; returns the records labelled.

    Reports progress after every 1/16 of the partitions or ``report_every`` seconds, whichever
    comes first. Partitions finish in key order, so nothing is reported until the first is done.
    """
    store = RecordStore(records_path)
    ranges, expected = store.partitions(partitions), len(store)
    workers = workers or max(1, available_cpus() - 2)
    print(f"labels: scanning {expected:,} records in {len(ranges):,} partitions with {workers} workers",
          flush=True)
    env = lmdb.open(out_path, map_size=1 << 40, max_dbs=2, subdir=True, sync=False, writemap=False)
    db = env.open_db(b"labels")
    total, started, reported = 0, time.time(), time.time()
    step = max(1, len(ranges) // 16)
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(records_path,)) as pool:
        # `imap` keeps partition order, and partitions are key-ordered, so every put appends.
        for done, rows in enumerate(pool.imap(_label_partition, ranges, chunksize=4), 1):
            with env.begin(write=True, db=db) as txn:
                cursor = txn.cursor()
                cursor.putmulti(rows, append=True)
            total += len(rows)
            now = time.time()
            if done % step == 0 or done == len(ranges) or now - reported >= report_every:
                reported = now
                rate = total / max(now - started, 1e-9)
                left = (expected - total) / rate if rate else 0
                print(f"labels: {total:,}/{expected:,} records, {done:,}/{len(ranges):,} partitions "
                      f"({rate:,.0f}/s, ~{left / 60:,.1f} min left)", flush=True)
    env.sync(True)
    env.close()
    return total


class LabelStore:
    """Read side of the label store."""

    def __init__(self, path: str):
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=False, max_dbs=2, subdir=True)
        self.db = self.env.open_db(b"labels", create=False)
        self.txn = self.env.begin(db=self.db, buffers=False)  # long-lived: the store is immutable

    @lru_cache(maxsize=1 << 18)
    def get(self, iri_or_uuid: str) -> tuple[str, str] | None:
        """``(class, label)`` of a record, or None if absent."""
        try:
            raw = self.txn.get(uuid_bytes(iri_or_uuid))
        except ValueError:  # not a UUID-keyed LUX IRI
            return None
        if raw is None:
            return None
        cls, _, label = raw.decode().partition("\t")
        return cls, label

    def label(self, iri_or_uuid: str) -> str | None:
        found = self.get(iri_or_uuid)
        return found[1] if found and found[1] else None

    def labels(self, iris, limit: int | None = None) -> list[str]:
        """Distinct labels of ``iris``, in order, unresolvable ones dropped."""
        out, seen = [], set()
        for iri in iris:
            label = self.label(iri)
            if label and label.casefold() not in seen:
                seen.add(label.casefold())
                out.append(label)
                if limit and len(out) == limit:
                    break
        return out

    def __len__(self) -> int:
        return self.txn.stat(self.db)["entries"]

    def scope_members(self, scope: str, after: str | None = None) -> Iterator[tuple[str, str]]:
        """Yields ``(uuid, class)`` of every record in ``scope``, in key order, after ``after``."""
        classes = {cls for cls, s in SCOPE_OF_CLASS.items() if s == scope}
        cursor = self.txn.cursor()
        if not (cursor.set_range(uuid_bytes(after)) if after else cursor.first()):
            return
        for key, value in cursor:
            uuid = uuid_str(key)
            if after and uuid == after:
                continue
            cls = value[: value.index(b"\t")].decode()
            if cls in classes:
                yield uuid, cls
