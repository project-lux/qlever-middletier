"""PostgreSQL record and HAL caches.

Record JSON is not in QLever - QLever answers "which URIs match?", this answers
"what does this URI look like?". The HAL table holds precomputed ``_links``
blocks, because building one means a query per candidate relation.

Both tables are optional and independently switchable, so a fetch may ask for
one, the other, or both in a single join.
"""

from __future__ import annotations

from psycopg import AsyncConnection
from psycopg.rows import dict_row


class PostgresCache:
    """A lazily (re)connected async connection to the document cache."""

    def __init__(self, settings):
        self.settings = settings
        self.conn = None

    @property
    def enabled(self) -> bool:
        return self.settings.use_pg_data_cache or self.settings.use_pg_hal_cache

    def connect_args(self) -> dict:
        """Keyword arguments for the connection, which psycopg quotes itself.

        They were once pasted into a conninfo string, so a password with a
        space or a quote in it failed, or added parameters of its own
        (``x host=elsewhere``). And QLMT_PGSSLMODE, "require" by default, was
        never passed at all: libpq's own default, "prefer", falls back to an
        unencrypted connection, and can be made to by anyone on the path.
        Over TCP (QLMT_PGHOST set) it is applied now; empty leaves libpq's
        default. A Unix socket takes no SSL, so it is left out there.
        """
        settings = self.settings
        args = {"user": settings.pguser, "dbname": settings.pgdb}
        if settings.pghost:
            args.update(host=settings.pghost, port=settings.pgport)
            if settings.pgpass:
                args["password"] = settings.pgpass
            if settings.pgsslmode:
                args["sslmode"] = settings.pgsslmode
        return args

    async def connect(self):
        settings = self.settings
        print(
            f"Connecting to PostgreSQL: {settings.pghost}:{settings.pgport}"
            f"/{settings.pgdb}"
        )
        try:
            self.conn = await AsyncConnection.connect(**self.connect_args())
        except Exception as e:
            print(f"Error connecting to database: {e}")

    async def aclose(self):
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    def _record_query(self):
        """The SELECT to run, and which columns come back, for this config.

        Returns (sql, kind) where kind is "both", "data", "hal" or None.
        """
        settings = self.settings
        if settings.use_pg_data_cache and settings.use_pg_hal_cache:
            return (
                f"SELECT doc.data, hal.data FROM {settings.pgtable} AS doc "
                f"LEFT JOIN {settings.pgtable_hal} AS hal "
                "ON doc.identifier = hal.identifier WHERE doc.identifier = %s",
                "both",
            )
        elif settings.use_pg_data_cache:
            return f"SELECT data FROM {settings.pgtable} WHERE identifier = %s", "data"
        elif settings.use_pg_hal_cache:
            return (
                f"SELECT data FROM {settings.pgtable_hal} WHERE identifier = %s",
                "hal",
            )
        return None, None

    async def fetch_record(self, identifier):
        """Return (row, kind). ``row`` is None when the record is not cached."""
        qry, kind = self._record_query()
        if qry is None:
            return None, None
        params = (identifier,)
        try:
            # this will fail at least the very first attempt before the
            # connection is created
            if self.conn is None:
                await self.connect()
            async with self.conn.cursor() as cursor:
                await cursor.execute(qry, params)
                row = await cursor.fetchone()
        except Exception as e:
            print("(re)connecting...")
            print(e)
            await self.connect()
            async with self.conn.cursor() as cursor:
                await cursor.execute(qry, params)
                row = await cursor.fetchone()
        return row, kind

    async def store_hal(self, identifier, links):
        print(f"Storing HAL cache for {identifier}")
        async with self.conn.cursor(row_factory=dict_row) as cursor:
            qry = (
                f"INSERT INTO {self.settings.pgtable_hal} (identifier, data) "
                "VALUES (%s, %s)"
            )
            params = (identifier, links)
            try:
                await cursor.execute(qry, params)
                row = await cursor.fetchall()
                print(f"Stored: {row}")
            except Exception as e:
                # try to reconnect
                print(e)
                print("(re)connecting to pg")
                await self.connect()
                print("reconnected")
                cursor2 = self.conn.cursor(row_factory=dict_row)
                await cursor2.execute(qry, params)
                print("Stored")
