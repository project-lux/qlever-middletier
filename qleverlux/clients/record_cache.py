"""Record JSON and HAL links, from whichever caches are configured.

Record JSON lives in PostgreSQL and/or LMDB; HAL links live in PostgreSQL or on
disk. When both PostgreSQL tables are on, one joined query fetches the record
and its links together.
"""

from __future__ import annotations

import os

import ujson as json

from qleverlux.clients.lmdb_store import LmdbCache
from qleverlux.clients.postgres import PostgresCache


class RecordCache:
    """The document cache: PostgreSQL, LMDB, or both."""

    def __init__(self, settings, postgres=None, lmdb_cache=None):
        self.settings = settings
        self.postgres = postgres if postgres is not None else PostgresCache(settings)
        self.lmdb = lmdb_cache if lmdb_cache is not None else LmdbCache(settings)

    async def connect(self):
        if self.postgres.enabled:
            await self.postgres.connect()
        if self.lmdb.enabled:
            self.lmdb.connect()

    async def aclose(self):
        await self.postgres.aclose()
        self.lmdb.close()

    async def fetch(self, identifier, record_class=None):
        """Return ``[record, hal_links]``, or None if the record is not cached.

        Either element may be None: the caches are independently configurable.
        ``record_class`` is the class from the URL; only an LMDB keyed by type
        reads it.
        """
        js = None
        hal = None

        row, kind = await self.postgres.fetch_record(identifier)
        if kind is not None:
            if row:
                if kind == "both":
                    return row
                elif kind == "data":
                    js = row[0]
                elif kind == "hal":
                    hal = row[0]
            else:
                return None

        from_lmdb = self.lmdb.get(identifier, record_class)
        if from_lmdb is not None:
            js = from_lmdb

        return [js, hal]


class HalCache:
    """Precomputed HAL ``_links`` blocks, on disk or in PostgreSQL.

    Computing a block means one query per candidate relation, so a miss is a
    slow request and the cache matters more than it looks.
    """

    def __init__(self, settings, postgres):
        self.settings = settings
        self.postgres = postgres
        if settings.use_disk_hal_cache:
            os.makedirs(settings.hal_cache_path, exist_ok=True)

    def disk_path(self, identifier):
        return os.path.join(self.settings.hal_cache_path, f"{identifier}.json")

    def get(self, identifier):
        """Only the disk cache is read here - the PostgreSQL copy arrives with
        the record itself, via the joined query in ``RecordCache.fetch``."""
        if not self.settings.use_disk_hal_cache:
            return None
        fn = self.disk_path(identifier)
        if os.path.exists(fn):
            with open(fn, "r") as f:
                return json.load(f)
        return None

    async def put(self, identifier, links):
        if self.settings.use_disk_hal_cache:
            with open(self.disk_path(identifier), "w") as f:
                json.dump(links, f)
        elif self.settings.use_pg_hal_cache:
            await self.postgres.store_hal(identifier, json.dumps(links))
