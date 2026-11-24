"""LMDB + USearch v3 hybrid vector store.

Moved here from ``../vectors/lmdb_vector_store.py``, which is now a shim importing this module.

One LMDB environment holds one or more *scopes* (the LUX core types: item, work, set, agent,
place, concept, event), each in its own sub-DBs, so the IRI maps, exact vectors and HNSW graph of
a scope always describe the same snapshot:

    {scope}.meta        b"meta"           -> JSON {dimensions, count, chunk_rows, generation, ...}
    {scope}.iri_to_id   IRI (utf-8)       -> u64 big-endian dense id (0..N-1)
    {scope}.id_to_iri   u64 big-endian id -> IRI (utf-8)
    {scope}.vectors     u32 big-endian chunk number -> PAD + (chunk_rows x D) matrix (f32 or f16)
    {scope}.graph       b"safegraphs"     -> (page - 16) zero bytes + USearch SafeGraphs monolith

``scope=None`` uses the unprefixed names of the original single-scope layout.

The graph may be built on fewer dimensions than the vectors hold (``graph_dimensions`` in the
meta, absent when it is the full width): the first N of each vector, re-normalized, as the
Matryoshka slicing of queries does. Walks then use the same slice of the query, and the exact
rerank the full stored vectors, so the graph only has to put the true neighbours in the pool.
On 327k concepts a Harrier graph on 512 of 1024 dimensions kept full-width recall at 600 bytes a
vector instead of 1112; at 256 text recall@100 fell to 0.79 (files/graph_quantization.py).
It needs a bigger rerank pool for that, which :data:`NARROW_RERANK_FACTOR` gives it by default.

Large LMDB values live in contiguous overflow pages starting 16 bytes past a page boundary (the
page header), so a 48-byte PAD makes a vector chunk 64-byte aligned, and a (page - 16)-byte pad
puts the graph exactly on a page boundary. Reading a chunk is a zero-copy NumPy view: fetching
vector `i` is pointer arithmetic, not a B-tree descent per vector.

The HNSW graph is served in place from LMDB's mmap with `DenseIndex.view_ram_buffer` (added on the
`view-ram-buffer` branch of the USearch checkout): its vectors, norms and neighbor lists are never
copied, so opening a 1M x 384 bf16 graph costs ~20 ms and ~60 MB (keys + key map) instead of
~70 ms and ~940 MB. The view reads LMDB pages for as long as it lives, so each graph generation
keeps its own read transaction open (pinning that MVCC snapshot) until the view is released. On a
USearch build without `view_ram_buffer`, the graph is copied out with `load_ram_buffer` instead.

Writing happens in two steps, so building can stream and resume:
* :class:`VectorWriter` appends vectors and IRIs a whole chunk per transaction;
* :func:`build_index` reads a scope's vectors back, builds the HNSW graph and publishes it as a
  new generation, which readers pick up with :meth:`LmdbVectorStore.refresh`.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import sys
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

import lmdb
import numpy as np
import usearch
from usearch import DenseIndex, DenseMetric, Scalar
from usearch.keys import keys_to_integers

PAD = 48  # 16-byte LMDB page header + 48 = 64-byte alignment for overflow values
LMDB_PAGE_HEADER = 16  # an overflow value starts this far past its first page's boundary
LMDB_NODE_HEADER = 8  # mn_lo, mn_hi, mn_flags, mn_ksize (u16 each); the key follows
LMDB_F_BIGDATA = 0x01  # a leaf node whose data is the page number of an overflow value
LMDB_P_OVERFLOW = 0x04
WHOLE_CHUNK_SHARE = 8  # read a whole chunk once more than 1/8 of its rows are wanted
PARALLEL_READS = 32  # more separate reads than this go to the read pool
_READ_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="vector-rows")  # idle threads cost nothing
CAN_VIEW = hasattr(DenseIndex, "view_ram_buffer")
CHUNK_ROWS = 65_536
MAX_DBS = 64
#: Candidates reranked per result wanted, by default: a graph on fewer dimensions orders its pool
#: less well, so it needs a bigger one. On 327k Harrier concepts with a 512-of-1024 graph, text
#: recall@100 was 0.961 at 3, 0.980 at 5 and 0.990 at 10 (full width: 0.992 at 3).
RERANK_FACTOR, NARROW_RERANK_FACTOR = 3, 10
TABLES = ("meta", "iri_to_id", "id_to_iri", "vectors", "graph")

PACK_U64 = struct.Struct(">Q").pack  # big-endian, so LMDB order == numeric order (append=True works)
PACK_U32 = struct.Struct(">I").pack
UNPACK_U64 = struct.Struct(">Q").unpack


def _matched_ids(matches) -> list[np.ndarray]:
    """Each query row's hit ids, as u64.

    Only the first ``counts[row]`` slots of a row are results; the rest of the
    buffer is uninitialized and may hold anything, including bytes that
    ``keys_to_integers`` refuses as a 128-bit identity. So slice before
    converting, never after.
    """
    counts = np.array(matches.counts)
    keys = np.array(matches.keys)  # copy out before the arena is reused
    return [keys_to_integers(keys[row][:count]) for row, count in enumerate(counts.tolist())]


def db_names(scope: str | None) -> dict[str, bytes]:
    return {table: (table if scope is None else f"{scope}.{table}").encode() for table in TABLES}


def _open_dbs(env: lmdb.Environment, scope: str | None, create: bool) -> dict[str, object]:
    return {table: env.open_db(name, create=create) for table, name in db_names(scope).items()}


def scopes_in(path_or_env: str | lmdb.Environment) -> list[str]:
    """The scopes a store holds; pass the open environment if this process already has one."""
    env = path_or_env if isinstance(path_or_env, lmdb.Environment) else lmdb.open(
        path_or_env, readonly=True, max_dbs=MAX_DBS, subdir=True)
    with env.begin() as txn:
        names = [bytes(key).decode() for key, _ in txn.cursor()]
    if env is not path_or_env:
        env.close()
    return sorted({name[: -len(".meta")] for name in names if name.endswith(".meta")})


def _write_graph(txn: lmdb.Transaction, env: lmdb.Environment, dbs: dict, monolith) -> int:
    """Stores a SafeGraphs monolith on a page boundary; returns the pad in front of it."""
    graph_pad = env.stat()["psize"] - LMDB_PAGE_HEADER
    value = bytearray(graph_pad + len(monolith))
    value[graph_pad:] = monolith
    txn.put(b"safegraphs", value, db=dbs["graph"])
    return graph_pad


#: Graph quantizations the store can build: it hands USearch float vectors, which the integer and
#: bit kinds do not convert - I8 from float input truncated every value and built a graph with
#: seeded recall@10 0.814 against 0.99 (they want input already scaled, or one 0/1 byte per bit).
FLOAT_QUANTIZATIONS = ("F64", "F32", "F16", "BF16", "E5M2", "E4M3", "E3M2", "E2M3")
#: What graphs are built as unless asked otherwise. E4M3, one byte a dimension, had the recall of
#: BF16 at half the size: on dev's 586k Harrier concepts text recall@10/@100 0.997/0.980 against
#: 0.997/0.985, 1112 bytes a vector against 2136 (files/graph_quantization.py).
GRAPH_QUANTIZATION = Scalar.E4M3


def _graph_rows(vectors: np.ndarray, graph_dims: int) -> np.ndarray:
    """``vectors`` as the graph sees them: the first ``graph_dims`` dimensions, re-normalized when
    that is fewer than they have. A zero row stays zero rather than becoming NaN."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if graph_dims >= vectors.shape[1]:
        return vectors
    sliced = vectors[:, :graph_dims]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    return sliced / np.where(norms > 0, norms, 1)


def _check_graph_dims(graph_dims: int, dims: int) -> int:
    """``graph_dims`` as stored: 0 means the vectors' full width."""
    if graph_dims == 0:
        return dims
    if not 0 < graph_dims <= dims:
        raise ValueError(f"a graph on {graph_dims} dimensions does not fit {dims}d vectors")
    return graph_dims


def _build_graph(chunks: Iterable[tuple[int, np.ndarray]], count: int, dims: int, quantization: Scalar,
                 connectivity: int, refine: bool) -> memoryview:
    """An HNSW graph over ``chunks``, each already cut to the graph's ``dims``."""
    if quantization.name not in FLOAT_QUANTIZATIONS:
        raise ValueError(f"cannot build a {quantization.name} graph: the store gives USearch float vectors, "
                         f"which that kind does not convert. Use one of {', '.join(FLOAT_QUANTIZATIONS)} "
                         f"(E4M3 for one byte a dimension).")
    pool = usearch.Executor.for_ram()
    index = DenseIndex.init_ram(dims, DenseMetric.ANGULAR, quantization, connectivity=connectivity)
    index.reserve(count)
    for first, block in chunks:
        index.on(pool).insert(np.arange(first, first + len(block), dtype=np.uint64), np.asarray(block, np.float32))
    if refine:  # cheap insertion + one refinement beats one expensive insertion pass
        index.on(pool).refine()
    monolith = memoryview(index.on(pool).save_buffer(usearch.Arena(pool)))
    pool.drain()  # an undrained pool keeps every worker spinning
    return monolith


def build_store(
    path: str,
    iris: Sequence[str],
    vectors: np.ndarray,
    *,
    scope: str | None = None,
    quantization: Scalar = GRAPH_QUANTIZATION,
    connectivity: int = 16,
    refine: bool = True,
    chunk_rows: int = CHUNK_ROWS,
    generation: int = 1,
    graph_dimensions: int = 0,
) -> None:
    """Builds a scope in one go from in-memory arrays: graph, vectors and IRI maps in one transaction.
    ``graph_dimensions`` builds the graph on that many leading dimensions (0: all of them)."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    if len(iris) != n:
        raise ValueError(f"{len(iris)} IRIs for {n} vectors")
    graph_dims = _check_graph_dims(graph_dimensions, dim)
    monolith = _build_graph([(0, _graph_rows(vectors, graph_dims))], n, graph_dims, quantization, connectivity, refine)

    iri_bytes = sum(len(s.encode()) for s in iris)
    payload = vectors.nbytes + len(monolith) + 2 * (iri_bytes + 32 * n)
    # 2x headroom: rewriting a large value needs a fresh contiguous run while readers pin the old one.
    env = lmdb.open(path, map_size=max(2 * payload + (256 << 20), 1 << 30), max_dbs=MAX_DBS, subdir=True)
    dbs = _open_dbs(env, scope, create=True)
    with env.begin(write=True) as txn:
        for doc_id, iri in enumerate(iris):
            key, value = PACK_U64(doc_id), iri.encode("utf-8")
            txn.put(value, key, db=dbs["iri_to_id"])
            txn.put(key, value, db=dbs["id_to_iri"], append=True)
        for chunk, start in enumerate(range(0, n, chunk_rows)):
            block = vectors[start : start + chunk_rows]
            txn.put(PACK_U32(chunk), b"\0" * PAD + block.tobytes(), db=dbs["vectors"], append=True)
        graph_pad = _write_graph(txn, env, dbs, monolith)
        meta = {
            "scope": scope,
            "dimensions": dim,
            "count": n,
            "chunk_rows": chunk_rows,
            "vector_dtype": "float32",
            "metric": "angular",
            "quantization": quantization.name,
            "connectivity": connectivity,
            "generation": generation,
            "graph_pad": graph_pad,
            "complete": True,
        }
        if graph_dims < dim:
            meta["graph_dimensions"] = graph_dims
        txn.put(b"meta", json.dumps(meta).encode(), db=dbs["meta"])
    env.close()


class VectorWriter:
    """Streams one scope's vectors into a store, resumably.

    Rows are committed a whole chunk per write transaction, together with their IRIs and the meta
    record, so a crash loses at most one uncommitted chunk and :attr:`cursor` - the last committed
    IRI - says where to resume. :meth:`finish` commits the final partial chunk.
    """

    def __init__(self, path: str, scope: str, dims: int, *, chunk_rows: int = 4096,
                 vector_dtype: str = "float32", model: str | None = None, restart: bool = False,
                 map_size: int = 1 << 40):
        if vector_dtype not in ("float32", "float16"):
            raise ValueError("vectors are stored as float32 or float16")
        self.env = lmdb.open(path, map_size=map_size, max_dbs=MAX_DBS, subdir=True)
        self.dbs = _open_dbs(self.env, scope, create=True)
        with self.env.begin(write=True) as txn:
            raw = txn.get(b"meta", db=self.dbs["meta"])
            if raw is not None and restart:
                for table in ("iri_to_id", "id_to_iri", "vectors", "graph", "meta"):
                    txn.drop(self.dbs[table], delete=False)
                raw = None
        if raw is None:
            self.meta = {"scope": scope, "dimensions": dims, "count": 0, "chunk_rows": chunk_rows,
                         "vector_dtype": vector_dtype, "metric": "angular", "model": model,
                         "cursor": None, "complete": False, "generation": 0}
        else:
            self.meta = json.loads(raw)
            if (self.meta["dimensions"], self.meta["chunk_rows"], self.meta["vector_dtype"]) != (dims, chunk_rows, vector_dtype):
                raise ValueError(f"scope {scope!r} was started with {self.meta['dimensions']}d, "
                                 f"{self.meta['chunk_rows']}-row {self.meta['vector_dtype']} chunks; pass restart=True to rebuild")
        self.dtype = np.dtype(vector_dtype)
        self._iris: list[str] = []
        self._rows: list[np.ndarray] = []
        self._pending = 0

    @property
    def count(self) -> int:
        return self.meta["count"]

    @property
    def cursor(self) -> str | None:
        return self.meta["cursor"]

    @property
    def complete(self) -> bool:
        return self.meta["complete"]

    def append(self, iris: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(iris), self.meta["dimensions"]):
            raise ValueError(f"expected {len(iris)} x {self.meta['dimensions']} vectors, got {vectors.shape}")
        self._iris += list(iris)
        self._rows.append(vectors)
        self._pending += len(iris)
        while self._pending >= self.meta["chunk_rows"]:
            self._commit(self.meta["chunk_rows"])

    def finish(self) -> None:
        if self._pending:
            self._commit(self._pending)
        self.meta["complete"] = True
        with self.env.begin(write=True) as txn:
            txn.put(b"meta", json.dumps(self.meta).encode(), db=self.dbs["meta"])

    def _commit(self, rows: int) -> None:
        block = np.concatenate(self._rows) if len(self._rows) > 1 else self._rows[0]
        chunk, rest = block[:rows], block[rows:]
        iris, self._iris = self._iris[:rows], self._iris[rows:]
        self._rows, self._pending = ([rest] if len(rest) else []), len(rest)
        first = self.meta["count"]
        if first % self.meta["chunk_rows"]:
            raise RuntimeError("a partial chunk was already committed; the scope is finished")
        with self.env.begin(write=True) as txn:
            for offset, iri in enumerate(iris):
                key, value = PACK_U64(first + offset), iri.encode("utf-8")
                txn.put(value, key, db=self.dbs["iri_to_id"])
                txn.put(key, value, db=self.dbs["id_to_iri"], append=True)
            txn.put(PACK_U32(first // self.meta["chunk_rows"]), b"\0" * PAD + chunk.astype(self.dtype).tobytes(),
                    db=self.dbs["vectors"], append=True)
            self.meta["count"] = first + rows
            self.meta["cursor"] = iris[-1]
            txn.put(b"meta", json.dumps(self.meta).encode(), db=self.dbs["meta"])

    def close(self) -> None:
        self.env.close()


def build_index(path: str, scope: str | None, *, quantization: Scalar = GRAPH_QUANTIZATION, connectivity: int = 16,
                refine: bool = True, map_size: int = 1 << 40, graph_dimensions: int = 0) -> dict:
    """Builds a scope's HNSW graph from its stored vectors and publishes it as the next generation.

    ``graph_dimensions`` builds it on that many leading dimensions of each vector (0: all of them).
    Like the quantization it is a property of this build, not the store: rebuilding without it
    goes back to the full width."""
    env = lmdb.open(path, map_size=map_size, max_dbs=MAX_DBS, subdir=True)
    dbs = _open_dbs(env, scope, create=False)
    with env.begin() as txn:
        meta = json.loads(txn.get(b"meta", db=dbs["meta"]))
    dims, rows, dtype = meta["dimensions"], meta["chunk_rows"], np.dtype(meta.get("vector_dtype", "float32"))
    try:
        graph_dims = _check_graph_dims(graph_dimensions, dims)
    except ValueError:
        env.close()
        raise

    def chunks():
        with env.begin(buffers=True) as txn:
            for key, value in txn.cursor(db=dbs["vectors"]):
                block = np.frombuffer(value, dtype=dtype, offset=PAD).reshape(-1, dims)
                yield struct.unpack(">I", bytes(key))[0] * rows, _graph_rows(block, graph_dims)

    monolith = _build_graph(chunks(), meta["count"], graph_dims, quantization, connectivity, refine)
    with env.begin(write=True) as txn:
        meta.update(graph_pad=_write_graph(txn, env, dbs, monolith), quantization=quantization.name,
                    connectivity=connectivity, generation=meta.get("generation", 0) + 1)
        meta.pop("graph_dimensions", None)
        if graph_dims < dims:
            meta["graph_dimensions"] = graph_dims
        txn.put(b"meta", json.dumps(meta).encode(), db=dbs["meta"])
    env.close()
    return meta


class _GraphSnapshot:
    """One graph generation: its index and, for an in-place view, the read transaction whose MVCC
    snapshot the view reads. Holding this object keeps both alive; releasing it frees the view
    first and only then ends the transaction, so no search ever reads recycled pages."""

    def __init__(self, env: lmdb.Environment, dbs: dict, threads: int = 0):
        self.txn: lmdb.Transaction | None = env.begin(buffers=True)
        self.index: DenseIndex | None = None
        try:
            self.meta = json.loads(bytes(self.txn.get(b"meta", db=dbs["meta"])))
            graph = self.txn.get(b"safegraphs", db=dbs["graph"])
            if graph is None:
                raise LookupError(f"scope {self.meta.get('scope')!r} has no graph yet; run build_index")
            blob = memoryview(graph)[self.meta.get("graph_pad", PAD):]
            if CAN_VIEW:
                # Zero-copy: vectors, norms and neighbors are read straight out of LMDB's mmap.
                self.index = DenseIndex.view_ram_buffer(blob)
            else:
                # Stock USearch 3.0: copy the graph out; the snapshot can end right away.
                # The pool exists only for this load: an undrained pool keeps every worker
                # spinning, about two cores each, even with no work queued.
                pool = usearch.Executor.for_ram(threads=threads)
                self.index = DenseIndex.load_ram_buffer(blob, executor=pool)
                pool.drain()
                self.close_txn()
        except BaseException:
            self.close()
            raise

    @property
    def in_place(self) -> bool:
        return self.txn is not None

    def close_txn(self) -> None:
        if self.txn is not None:
            self.txn.abort()
            self.txn = None

    def close(self) -> None:
        self.index = None  # the view goes first: it reads the transaction's pages until freed
        self.close_txn()

    def __del__(self) -> None:
        if hasattr(self, "txn"):
            self.close()


class _RowReader:
    """Reads vector rows straight from the data file, so scoring N hits reads about N rows.

    py-lmdb touches every page of a value it returns (``txn.get`` of a 13 MB chunk faulted in all
    769 of its pages, where positioning a cursor on its key faulted 1), so reading rows through it
    pulls in each whole chunk a hit sits in: harmless while the store is in memory, seconds of
    disk reads when it is not. Instead a chunk is located by its key - LMDB keeps the page number
    of a large value in the leaf node beside the key - and its rows are read with ``os.pread``,
    which maps nothing. Chunks are written once and never rewritten, and the reads happen inside
    the caller's read transaction, which keeps those pages from being reused.

    The node layout is LMDB 0.9's on a little-endian 64-bit machine. ``verify`` checks it against
    the page header in the file before it is trusted; a store it does not fit is read as before.
    """

    def __init__(self, env: lmdb.Environment):
        path = env.path()
        self.path = os.path.join(path, "data.mdb") if os.path.isdir(path) else path
        self.psize = env.stat()["psize"]
        self.fd = os.open(self.path, os.O_RDONLY)
        if hasattr(os, "posix_fadvise"):  # Linux: no readahead for scattered row reads
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def locate(self, cursor: lmdb.Cursor, key: bytes) -> tuple[int, int] | None:
        """(file offset, size) of the value stored under ``key``, read from its leaf node without
        touching the value; None when the value is stored inline (small) or the key is absent."""
        if not cursor.set_key(key):
            return None
        found = cursor.key()
        if not isinstance(found, memoryview):  # a copy (buffers=False): nothing beside it to read
            return None
        address = np.frombuffer(found, dtype=np.uint8).ctypes.data
        lo, hi, flags, ksize = struct.unpack("<HHHH", ctypes.string_at(address - LMDB_NODE_HEADER, LMDB_NODE_HEADER))
        if not flags & LMDB_F_BIGDATA or ksize != len(found):
            return None
        (pgno,) = struct.unpack("<Q", ctypes.string_at(address + ksize, 8))
        return pgno * self.psize + LMDB_PAGE_HEADER, lo | (hi << 16)

    def verify(self, cursor: lmdb.Cursor, key: bytes) -> bool:
        """Whether ``locate`` finds the value under ``key`` where the file says it is: the page
        header there names the same page, marks it as an overflow page long enough for the value,
        and the value starts with PAD zero bytes, as every chunk does."""
        if sys.byteorder != "little" or struct.calcsize("P") != 8:
            return False
        located = self.locate(cursor, key)
        if located is None:
            return False
        offset, size = located
        head = os.pread(self.fd, LMDB_PAGE_HEADER + PAD, offset - LMDB_PAGE_HEADER)
        if len(head) < LMDB_PAGE_HEADER + PAD:
            return False
        pgno, _pad, flags, pages = struct.unpack("<QHHI", head[:LMDB_PAGE_HEADER])
        return (pgno * self.psize == offset - LMDB_PAGE_HEADER and flags & LMDB_P_OVERFLOW
                and pages * self.psize >= size + LMDB_PAGE_HEADER and head[LMDB_PAGE_HEADER:] == bytes(PAD))

    def read(self, spans: list[tuple[int, int]]) -> list[bytes]:
        """``os.pread`` each (offset, length); many at once on the read pool, which keeps several
        reads in flight when the pages are on disk."""
        if len(spans) > PARALLEL_READS:
            return list(_READ_POOL.map(lambda span: os.pread(self.fd, span[1], span[0]), spans))
        return [os.pread(self.fd, length, offset) for offset, length in spans]


class LmdbVectorStore:
    """Read side of one scope: HNSW served in place from LMDB's mmap (or copied, on stock USearch),
    exact vectors read zero-copy from the chunks.

    Searches run on the calling thread, so a server's request threads search in parallel with no
    executor to coordinate - and no idle pool spinning its workers between requests."""

    def __init__(self, path: str, scope: str | None = None, *, threads: int = 0, map_size: int = 1 << 40,
                 env: lmdb.Environment | None = None):
        # Keep lock=True (the default): it registers each read txn in LMDB's reader table, so a
        # concurrent writer never recycles pages a zero-copy buffer still points at.
        # A view pins a read txn, and LMDB cannot remap a grown file while one is open
        # (MDB_MAP_RESIZED), so the reader reserves address space up front - none of it is
        # allocated, and pages past the file's end are never touched. Several scopes of one file
        # share one environment (py-lmdb opens a path once per process).
        self._owns_env = env is None
        self.env = env or lmdb.open(path, readonly=True, max_dbs=MAX_DBS, readahead=False, subdir=True,
                                    map_size=map_size)
        self.scope = scope
        self.dbs = _open_dbs(self.env, scope, create=False)
        self.threads = threads  # only the copying fallback uses threads, for loading the graph
        self._local = threading.local()
        self._graph: _GraphSnapshot | None = None
        self.refresh()
        self._rows = self._row_reader()

    def _row_reader(self) -> _RowReader | None:
        """A reader for rows straight from the file, if this store's layout passes its check."""
        try:
            reader = _RowReader(self.env)
        except OSError:
            return None
        with self.env.begin(buffers=True) as txn, txn.cursor(db=self.dbs["vectors"]) as cursor:
            first = cursor.first() and bytes(cursor.key())
            if first and reader.verify(cursor, first):
                return reader
        # A store whose first chunk is small enough to sit inline has nothing to check against;
        # its chunks are small, so reading them whole costs little anyway.
        reader.close()
        return None

    # -- lifecycle -----------------------------------------------------------------------------

    @property
    def meta(self) -> dict:
        return self._graph.meta

    @property
    def index(self) -> DenseIndex:
        return self._graph.index

    def refresh(self) -> bool:
        """Opens the graph if a writer published a new generation; True when it swapped.

        The previous generation stays readable for searches already holding it, and its read
        transaction ends once the last of them lets go. Until then LMDB keeps its pages, so a
        writer's file grows by about one graph per generation still pinned."""
        if self._graph is not None:
            with self.env.begin(buffers=True) as txn:
                generation = json.loads(bytes(txn.get(b"meta", db=self.dbs["meta"])))["generation"]
            if generation == self._graph.meta["generation"]:
                return False
        self._graph = _GraphSnapshot(self.env, self.dbs, self.threads)  # atomic swap
        return True

    def close(self) -> None:
        if getattr(self, "_rows", None) is not None:
            self._rows.close()
            self._rows = None
        if self._graph is not None:
            self._graph.close()
            self._graph = None
        if self._owns_env:
            self.env.close()

    def describe(self) -> dict:
        """Which way the graph is served and how much memory the index itself owns."""
        graph = self._graph
        return {
            "scope": self.scope,
            "in_place": graph.in_place,
            "entries": len(graph.index),
            "dimensions": graph.meta["dimensions"],
            "graph_dimensions": graph.meta.get("graph_dimensions", graph.meta["dimensions"]),
            "owned_bytes": graph.index.describe().memory_usage,
            "generation": graph.meta["generation"],
        }

    @property
    def _arena(self) -> usearch.Arena:
        # An arena serves one call at a time, so each thread keeps its own.
        arena = getattr(self._local, "arena", None)
        if arena is None:
            arena = self._local.arena = usearch.Arena()
        return arena

    # -- zero-copy reads -------------------------------------------------------------------------

    def _vectors(self, txn: lmdb.Transaction, ids: np.ndarray) -> np.ndarray:
        """Rows for `ids` as f32. Read straight from the file where the store allows it (see
        :class:`_RowReader`): consecutive rows in one read, a chunk with many wanted rows whole,
        the rest one row at a time. Otherwise through py-lmdb, which reads each whole chunk."""
        dim, rows_per_chunk = self.meta["dimensions"], self.meta["chunk_rows"]
        dtype = np.dtype(self.meta.get("vector_dtype", "float32"))
        ids = np.asarray(ids, dtype=np.uint64)
        out = np.empty((len(ids), dim), dtype=np.float32)
        chunks, rows = np.divmod(ids, rows_per_chunk)
        row_bytes = dim * dtype.itemsize
        spans: list[tuple[int, int]] = []  # (offset, length) to read
        fills: list[tuple[np.ndarray, int, np.ndarray]] = []  # (out positions, span index, rows within span)
        with txn.cursor(db=self.dbs["vectors"]) as cursor:
            for chunk in np.unique(chunks):
                mask = np.nonzero(chunks == chunk)[0]
                key = PACK_U32(int(chunk))
                located = self._rows.locate(cursor, key) if self._rows is not None else None
                if located is None:
                    matrix = np.frombuffer(txn.get(key, db=self.dbs["vectors"]), dtype=dtype, offset=PAD).reshape(-1, dim)
                    out[mask] = matrix[rows[mask]]
                    continue
                start, size = located
                start += PAD
                wanted = rows[mask].astype(np.int64)
                if len(np.unique(wanted)) * WHOLE_CHUNK_SHARE > rows_per_chunk:
                    fills.append((mask, len(spans), wanted))
                    spans.append((start, size - PAD))
                    continue
                # Runs of consecutive rows, each one read.
                order = np.argsort(wanted, kind="stable")
                run_first = None
                for position in order:
                    row = int(wanted[position])
                    if run_first is None or row > run_last + 1:
                        if run_first is not None:
                            spans.append((start + run_first * row_bytes, (run_last - run_first + 1) * row_bytes))
                        run_first, run_last, run_members = row, row, []
                        fills.append((run_members, len(spans), run_first))
                    run_last = row
                    run_members.append(int(mask[position]))
                spans.append((start + run_first * row_bytes, (run_last - run_first + 1) * row_bytes))
            data = self._rows.read(spans) if spans else []
        for positions, span, rows_in in fills:
            matrix = np.frombuffer(data[span], dtype=dtype).reshape(-1, dim)
            if isinstance(rows_in, np.ndarray):  # a whole chunk: rows_in are the wanted rows
                out[positions] = matrix[rows_in]
            else:  # a run of rows starting at row rows_in
                positions = np.asarray(positions)
                out[positions] = matrix[rows[positions].astype(np.int64) - rows_in]
        return out

    def _ids_for(self, txn: lmdb.Transaction, iris: Iterable[str]) -> tuple[list[str], np.ndarray]:
        found_iris, found_ids = [], []
        db = self.dbs["iri_to_id"]
        for iri in iris:
            raw = txn.get(iri.encode("utf-8"), db=db)
            if raw is not None:
                found_iris.append(iri)
                found_ids.append(UNPACK_U64(raw)[0])
        return found_iris, np.asarray(found_ids, dtype=np.uint64)

    def _iris_for(self, txn: lmdb.Transaction, ids: Iterable[int]) -> list[str]:
        db = self.dbs["id_to_iri"]
        return [bytes(txn.get(PACK_U64(int(i)), db=db)).decode("utf-8") for i in ids]

    @staticmethod
    def _cosine(matrix: np.ndarray, queries: np.ndarray) -> np.ndarray:
        """Each row's cosine with its best-matching query (``queries`` is one vector or a matrix)."""
        queries = np.atleast_2d(queries)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) * np.linalg.norm(queries, axis=1)
        return ((matrix @ queries.T) / np.where(norms == 0, 1, norms)).max(axis=1)

    @staticmethod
    def _top(scores: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(scores))
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part])]

    def _queries(self, query: np.ndarray) -> np.ndarray:
        """One query vector or a matrix of them, as f32 rows, sliced Matryoshka-style and
        re-normalized when wider than this scope's vectors."""
        q = np.atleast_2d(np.asarray(query, dtype=np.float32))
        dim = self.meta["dimensions"]
        if q.shape[1] < dim:
            raise ValueError(f"a {q.shape[1]}d query cannot search {dim}d vectors")
        if q.shape[1] > dim:
            q = q[:, :dim]
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
        return q

    # -- public API -------------------------------------------------------------------------------

    def get_vector(self, iri: str) -> np.ndarray | None:
        with self.env.begin(buffers=True) as txn:
            _, ids = self._ids_for(txn, [iri])
            return self._vectors(txn, ids)[0] if len(ids) else None

    @staticmethod
    def _for_graph(graph: _GraphSnapshot, queries: np.ndarray) -> np.ndarray:
        """Query rows of the vectors' width cut to the width ``graph`` was built on."""
        return _graph_rows(queries, graph.meta.get("graph_dimensions", graph.meta["dimensions"]))

    def _walk(self, graph: _GraphSnapshot, queries: np.ndarray, wanted: int, expansion_factor: int) -> dict[int, float]:
        """One HNSW walk per query row; each id found with its best quantized cosine (over the
        graph's dimensions, which may be fewer than the vectors')."""
        # The walk returns at most expansion_factor hits, so widen it to cover what is wanted;
        # otherwise a top_k above expansion_factor / rerank_factor silently comes back short.
        matches = graph.index.search(self._for_graph(graph, queries), wanted, self._arena,
                                     expansion_factor=max(expansion_factor, wanted))
        distances = np.array(matches.distances)
        best: dict[int, float] = {}
        for row, ids in enumerate(_matched_ids(matches)):
            for key, distance in zip(ids.tolist(), distances[row][: len(ids)].tolist()):
                best[key] = max(best.get(key, -2.0), 1.0 - distance)
        return best

    def search_global(
        self, query: np.ndarray, top_k: int = 20, *, rerank_factor: int | None = None, expansion_factor: int = 128,
        rewalk: int = 0, rewalk_expansion_factor: int | None = None,
    ) -> list[tuple[str, float]]:
        """Vector-first: quantized HNSW walk, then exact cosine rerank from LMDB.

        ``query`` may be a matrix of several query vectors: each walks the graph, the hits are
        pooled, and every hit is scored by its best-matching query ("similar to any of these").

        ``rewalk`` then walks again from the stored vectors of the best ``rewalk`` hits and
        reranks the pool. A text query sits away from every record (its best cosine is about
        0.6), and a walk from there misses records the graph does link: each one missed on dev
        was found first when its own vector was the query. A walk from a record starts where the
        graph is reliable, so the second walks find some of what the first one passed by.
        ``rewalk_expansion_factor`` is their width (``expansion_factor`` when None). On dev narrow
        ones (300) were cheap and recovered a missed top hit; wide ones (10,000) recovered more on
        place queries but cost as much as the first walk again.

        ``rerank_factor`` is how many candidates are reranked per result wanted; None picks
        :data:`RERANK_FACTOR`, or :data:`NARROW_RERANK_FACTOR` for a graph on fewer dimensions."""
        q = self._queries(query)
        graph = self._graph  # held for the whole call: keeps this generation's pages pinned
        if rerank_factor is None:
            rerank_factor = NARROW_RERANK_FACTOR if "graph_dimensions" in graph.meta else RERANK_FACTOR
        wanted = top_k * max(rerank_factor, 1)
        best = self._walk(graph, q, wanted, expansion_factor)
        ids = np.fromiter(best, dtype=np.uint64, count=len(best))
        if not len(ids):
            return []
        # A graph on fewer dimensions scores over those alone, so its cosines are always replaced.
        exact = rerank_factor > 1 or rewalk or "graph_dimensions" in graph.meta
        with self.env.begin(buffers=True) as txn:
            if exact:
                scores = self._cosine(self._vectors(txn, ids), q)
            else:
                scores = np.fromiter(best.values(), dtype=np.float64, count=len(best))
            if rewalk:
                seeds = self._vectors(txn, ids[self._top(scores, rewalk)])
        if rewalk:
            more = self._walk(graph, seeds, wanted, rewalk_expansion_factor or expansion_factor)
            ids = np.union1d(ids, np.fromiter(more, dtype=np.uint64, count=len(more)))
        with self.env.begin(buffers=True) as txn:
            if rewalk:
                scores = self._cosine(self._vectors(txn, ids), q)
            order = self._top(scores, top_k)
            return list(zip(self._iris_for(txn, ids[order]), scores[order].tolist()))

    def search_exact(self, query: np.ndarray, top_k: int = 20, *, block_rows: int = 16_384) -> list[tuple[str, float]]:
        """Every vector of the scope scored exactly, no graph: perfect recall, at a cost linear in
        the scope's size. Chunks are read zero-copy and converted a block at a time, so memory
        stays at one block whatever the scope."""
        q = self._queries(query)
        dim, rows_per_chunk, count = self.meta["dimensions"], self.meta["chunk_rows"], self.meta["count"]
        dtype = np.dtype(self.meta.get("vector_dtype", "float32"))
        top_ids = np.empty(0, dtype=np.uint64)
        top_scores = np.empty(0, dtype=np.float64)
        with self.env.begin(buffers=True) as txn:
            with txn.cursor(db=self.dbs["vectors"]) as cursor:
                for key, buf in cursor:
                    first = struct.unpack(">I", bytes(key))[0] * rows_per_chunk
                    matrix = np.frombuffer(buf, dtype=dtype, offset=PAD).reshape(-1, dim)
                    valid = max(0, min(len(matrix), count - first))
                    for start in range(0, valid, block_rows):
                        block = matrix[start : min(start + block_rows, valid)].astype(np.float32)
                        scores = self._cosine(block, q).astype(np.float64)
                        keep = self._top(scores, top_k)
                        top_ids = np.concatenate([top_ids, (first + start + keep).astype(np.uint64)])
                        top_scores = np.concatenate([top_scores, scores[keep]])
                        if len(top_ids) > 4 * top_k:
                            survivors = self._top(top_scores, top_k)
                            top_ids, top_scores = top_ids[survivors], top_scores[survivors]
            if not len(top_ids):
                return []
            order = self._top(top_scores, top_k)
            return list(zip(self._iris_for(txn, top_ids[order]), top_scores[order].tolist()))

    def search_candidates(
        self,
        query: np.ndarray,
        candidate_iris: Iterable[str],
        top_k: int = 20,
        *,
        exact_threshold: int = 50_000,
        expansion_factor: int = 256,
    ) -> list[tuple[str, float]]:
        """Graph-first (VALUES pushdown): rank IRIs the SPARQL side already selected.

        Small candidate sets are scored exactly from the zero-copy chunks. Large ones use a
        filtered HNSW walk, where rejected nodes still route the traversal, then the exact rerank.
        """
        q = self._queries(query)
        with self.env.begin(buffers=True) as txn:
            iris, ids = self._ids_for(txn, candidate_iris)
            if not len(ids):
                return []
            if len(ids) <= exact_threshold:
                scores = self._cosine(self._vectors(txn, ids), q)
                order = self._top(scores, top_k)
                return [(iris[i], float(scores[i])) for i in order]

        allowed = set(ids.tolist())  # Python predicate holds the GIL; a C++ middle tier would use a bitset
        graph = self._graph  # held for the whole call: keeps this generation's pages pinned
        wanted = top_k * 3
        matches = graph.index.search(
            self._for_graph(graph, q), wanted, self._arena, expansion_factor=max(expansion_factor, wanted),
            predicate=lambda key: int(key) in allowed,
        )
        hit_ids = np.unique(np.concatenate(_matched_ids(matches)))
        if not len(hit_ids):
            return []
        with self.env.begin(buffers=True) as txn:
            scores = self._cosine(self._vectors(txn, hit_ids), q)
            order = self._top(scores, top_k)
            return list(zip(self._iris_for(txn, hit_ids[order]), scores[order].tolist()))


if __name__ == "__main__":
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else "demo.lmdb"
    n, dim, clusters = 100_000, 384, 1_000
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)
    data = centers[rng.integers(0, clusters, n)] + 0.35 * rng.standard_normal((n, dim)).astype(np.float32)
    iris = [f"https://lux.collections.yale.edu/data/concept/{i:08x}-0000-4000-8000-000000000000" for i in range(n)]

    t = time.perf_counter()
    build_store(path, iris, data)
    print(f"build: {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    store = LmdbVectorStore(path)
    print(f"open: {(time.perf_counter() - t) * 1e3:.1f} ms -> {store.describe()}")

    queries = data[rng.choice(n, 200, replace=False)] + 0.05 * rng.standard_normal((200, dim)).astype(np.float32)
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    truth = [set(np.argsort(-(unit @ (q / np.linalg.norm(q))))[:10].tolist()) for q in queries]
    iri_index = {iri: i for i, iri in enumerate(iris)}
    for rerank in (1, 3):
        t = time.perf_counter()
        results = [store.search_global(q, 10, rerank_factor=rerank) for q in queries]
        elapsed = (time.perf_counter() - t) / len(queries) * 1e6
        recall = np.mean([len(truth[i] & {iri_index[iri] for iri, _ in r}) / 10 for i, r in enumerate(results)])
        print(f"search_global rerank_factor={rerank}: recall@10 {recall:.3f}, {elapsed:.0f} us/query")

    candidates = [iris[i] for i in rng.choice(n, 5_000, replace=False)]
    t = time.perf_counter()
    hits = store.search_candidates(queries[0], candidates, 10)
    print(f"search_candidates exact over 5,000: {(time.perf_counter() - t) * 1e3:.1f} ms -> {hits[0]}")
    many = [iris[i] for i in rng.choice(n, 60_000, replace=False)]
    t = time.perf_counter()
    hits = store.search_candidates(queries[0], many, 10)
    print(f"search_candidates filtered HNSW over 60,000: {(time.perf_counter() - t) * 1e3:.1f} ms -> {hits[0]}")
    print(f"get_vector ok: {np.allclose(store.get_vector(iris[123]), data[123])}")
