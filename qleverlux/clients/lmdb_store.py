"""LMDB record cache: zlib-compressed JSON keyed by raw UUID bytes.

An alternative to the PostgreSQL document cache for single-process or
read-mostly deployments. Written by ``files/load-json-to-postgres.py``.
"""

from __future__ import annotations

import zlib
from uuid import UUID

import lmdb
import ujson as json


class LmdbCache:
    """Read-only handle on the LMDB document cache."""

    def __init__(self, settings):
        self.settings = settings
        self.env = None
        self.db = None

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
        # And make a new txn for each request

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.db = None

    def get(self, identifier):
        """Return the decoded record, or None if absent."""
        if self.env is None:
            return None
        with self.env.begin(buffers=True) as txn:
            if self.settings.lmdb_binary_uuid_keys:
                key = UUID(identifier).bytes
            else:
                key = identifier.encode("utf-8")
            value = txn.get(key=key, db=self.db)
            if not value:
                return None
            js = json.loads(zlib.decompress(value).decode())
            if self.settings.lmdb_json_path:
                js = js[self.settings.lmdb_json_path]
            return js
