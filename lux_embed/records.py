"""Read-only access to a LUX record LMDB, and the record-class vocabulary.

Two layouts are recognised automatically:

* the middle tier's datacache (e.g. ``nrich_lux.lmdb``): sub-DB ``data``, key = the 36-character
  UUID string, value = zlib JSON ``{"json": <Linked Art record>, "triples": ..., ...}``;
* the data-pipeline distribution store: sub-DB ``data``, key = the 16-byte UUID, value = zlib JSON
  of the bare Linked Art record.

Either way :meth:`RecordStore.get` answers the Linked Art record itself.
"""

from __future__ import annotations

import random
import uuid as uuidlib
import zlib
from collections.abc import Iterator

import lmdb

try:  # orjson parses records ~3x faster; json is the fallback
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover
    import json

    _loads = json.loads

DATA_PREFIX = "https://lux.collections.yale.edu/data/"

# Linked Art class -> LUX core type ("scope"), as qleverlux's CLASS_TO_SCOPE folds them.
SCOPE_OF_CLASS = {
    "HumanMadeObject": "item",
    "DigitalObject": "item",
    "VisualItem": "work",
    "LinguisticObject": "work",
    "Set": "set",
    "Person": "agent",
    "Group": "agent",
    "Place": "place",
    "Type": "concept",
    "Material": "concept",
    "Language": "concept",
    "Currency": "concept",
    "MeasurementUnit": "concept",
    "Activity": "event",
    "Event": "event",
    "Period": "event",
}
SCOPES = ("item", "work", "set", "agent", "place", "concept", "event")


def uuid_of(iri_or_uuid: str) -> str:
    """The UUID string at the end of a LUX IRI (or the argument itself, if already a UUID)."""
    return iri_or_uuid.rstrip("/").rsplit("/", 1)[-1]


def uuid_bytes(iri_or_uuid: str) -> bytes:
    return uuidlib.UUID(uuid_of(iri_or_uuid)).bytes


def uuid_str(raw: bytes) -> str:
    return str(uuidlib.UUID(bytes=raw))


class RecordStore:
    """Read-only Linked Art records keyed by UUID.

    ``lock=False`` is safe here: the record cache is written once and never while served.
    ``readahead`` is off by default because ``embed`` looks records up one by one; a full ``scan``
    should turn it on, or every page is its own read (on cluster storage, workers sit at 2% CPU).
    """

    def __init__(self, path: str, db: bytes = b"data", readahead: bool = False):
        self.path = path
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=readahead, max_dbs=8, subdir=True)
        self.db = self.env.open_db(db, create=False)
        with self.env.begin(db=self.db) as txn:
            first = next(iter(txn.cursor().iternext(values=False)), b"")
        self.binary_keys = len(first) == 16

    def __len__(self) -> int:
        with self.env.begin(db=self.db) as txn:
            return txn.stat(self.db)["entries"]

    def _key(self, iri_or_uuid: str) -> bytes:
        return uuid_bytes(iri_or_uuid) if self.binary_keys else uuid_of(iri_or_uuid).encode()

    @staticmethod
    def _decode(value: bytes) -> dict:
        doc = _loads(zlib.decompress(value))
        return doc["json"] if isinstance(doc, dict) and "json" in doc and "type" not in doc else doc

    def get(self, iri_or_uuid: str, txn: lmdb.Transaction | None = None) -> dict | None:
        if txn is None:
            with self.env.begin(db=self.db) as own:
                raw = own.get(self._key(iri_or_uuid))
        else:
            raw = txn.get(self._key(iri_or_uuid), db=self.db)
        return None if raw is None else self._decode(raw)

    def scan(self, first: bytes | None = None, bound: bytes | None = None) -> Iterator[tuple[str, dict]]:
        """Yields ``(uuid, record)`` for keys in ``[first, bound)``, in key order."""
        with self.env.begin(db=self.db, buffers=True) as txn:
            cursor = txn.cursor()
            if not (cursor.set_range(first) if first else cursor.first()):
                return
            for key, value in cursor:
                key = bytes(key)
                if bound is not None and key >= bound:
                    return
                yield (uuid_str(key) if self.binary_keys else key.decode()), self._decode(value)

    def sample(self, count: int, seed: int = 0) -> Iterator[dict]:
        """Yields about ``count`` records spread evenly over the store, without reading the rest.

        Each one is the first record at or after a random UUID, so the cost is ``count`` seeks
        rather than a walk over every record (minutes on full LUX). UUID keys are uniform, so this
        samples evenly; neighbouring seeks can land on the same record, which is only dropped.
        Asking for the whole store or more just scans it.
        """
        if count >= len(self):
            yield from (record for _, record in self.scan())
            return
        rng = random.Random(seed)
        points = sorted(uuidlib.UUID(int=rng.getrandbits(128)) for _ in range(count))
        last = None
        with self.env.begin(db=self.db, buffers=True) as txn:
            cursor = txn.cursor()
            for point in points:  # in key order, so the seeks move forward through the file
                if not cursor.set_range(point.bytes if self.binary_keys else str(point).encode()):
                    break  # past the last key; every later point is too
                key = bytes(cursor.key())
                if key != last:
                    last = key
                    yield self._decode(cursor.value())

    def partitions(self, count: int) -> list[tuple[bytes | None, bytes | None]]:
        """Splits the key space into about ``count`` ranges of similar size, for parallel scans."""
        if self.binary_keys:  # split on the first two bytes, so up to 65,536 ranges
            edges = [round(i * 65536 / count).to_bytes(2, "big") for i in range(1, count)
                     if round(i * 65536 / count) < 65536]
        else:  # lower-case hex strings: split on the first three hex digits
            edges = [f"{round(i * 4096 / count):03x}".encode() for i in range(1, count) if round(i * 4096 / count) < 4096]
        edges = sorted(set(edges))
        bounds = [None, *edges]
        return list(zip(bounds, [*edges, None]))
