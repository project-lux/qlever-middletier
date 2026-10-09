"""LMDB record cache: zlib-compressed JSON keyed by identifier.

An alternative to the PostgreSQL document cache for single-process or
read-mostly deployments. Written by ``files/load-json-to-postgres.py``, or for
Wikidata by the data pipeline's ``make_wikidata_lmdb.py``.

How an identifier becomes a key is ``QLMT_LMDB_KEY_FORMAT``:

* ``uuid`` - the 16 raw bytes of the UUID (the LUX stores)
* ``text`` - the identifier as UTF-8
* ``qid`` - a Wikidata Q number as a 4 byte big-endian int, then one character
  for the type the record was built as. The same Q-id is cached once per type
  (Q90 is a place, and a concept), so the type is part of the key; it comes
  from the record class in the URL, and the character from the store's own
  ``types`` table, so the writer stays the only owner of the codes.
"""

from __future__ import annotations

import re
import zlib
from uuid import UUID

import lmdb
import ujson as json

from qleverlux.enums import CLASS_TO_TYPE

QID = re.compile(r"Q[1-9][0-9]*$")
UINT32_MAX = 2**32 - 1


def normalise_identifier(key_format, identifier):
    """The identifier in canonical form, or ValueError if it cannot be one."""
    if key_format == "uuid":
        return str(UUID(identifier))
    if key_format == "qid" and not QID.match(identifier):
        raise ValueError(f"Not a Wikidata Q-id: {identifier}")
    return identifier


class LmdbCache:
    """Read-only handle on the LMDB document cache."""

    def __init__(self, settings):
        self.settings = settings
        self.env = None
        self.db = None
        #: Linked Art type -> key character, from the store (qid keys only).
        self.type_codes = {}

    @property
    def enabled(self) -> bool:
        return self.settings.use_lmdb_data_cache and self.settings.lmdb_path != ""

    def connect(self):
        print(f"Connecting to LMDB: {self.settings.lmdb_path}")
        if self.env is None:
            self.env = lmdb.open(
                self.settings.lmdb_path, max_dbs=3, readonly=True, lock=False
            )
        if self.db is None:
            self.db = self.env.open_db(b"data", dupsort=False)
        if self.settings.lmdb_key_format == "qid" and not self.type_codes:
            types_db = self.env.open_db(b"types", dupsort=False)
            with self.env.begin() as txn:
                self.type_codes = {
                    k.decode("utf-8"): v for k, v in txn.cursor(db=types_db)
                }
        # And make a new txn for each request

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.db = None

    def key(self, identifier, record_class=None) -> bytes:
        """The store's key for a record, or ValueError if it cannot have one."""
        fmt = self.settings.lmdb_key_format
        if fmt == "uuid":
            return UUID(identifier).bytes
        if fmt == "text":
            return identifier.encode("utf-8")
        if not QID.match(identifier):
            raise ValueError(f"Not a Wikidata Q-id: {identifier}")
        n = int(identifier[1:])
        if n > UINT32_MAX:
            raise ValueError(f"Q-id out of range: {identifier}")
        code = self.type_codes.get(CLASS_TO_TYPE.get(record_class))
        if code is None:
            raise ValueError(f"No key character for record class {record_class}")
        return n.to_bytes(4, "big") + code

    def get(self, identifier, record_class=None):
        """Return the decoded record, or None if absent."""
        if self.env is None:
            return None
        try:
            key = self.key(identifier, record_class)
        except ValueError:
            return None
        with self.env.begin(buffers=True) as txn:
            value = txn.get(key=key, db=self.db)
            if not value:
                return None
            js = json.loads(zlib.decompress(value).decode())
            if self.settings.lmdb_json_path:
                js = js[self.settings.lmdb_json_path]
            return js
