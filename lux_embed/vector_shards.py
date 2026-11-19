"""Vectors written to plain files by GPU jobs, then loaded into the LMDB vector store by one CPU job.

``embed --cards C.lmdb --out DIR [--shard I/N]`` writes a shard directory instead of an LMDB store,
so GPU jobs never contend for LMDB's single writer, and a big class can be split by key range over
N GPUs at once. ``load --store V.lmdb --scope S DIR...`` then writes every shard of the scope into
the store, in key order, exactly as ``embed --store`` would have (same chunks, same dtype, same
model in the meta), and builds the graph.

A shard directory holds:

* ``vectors.bin`` - rows of ``dims`` floats (``vector_dtype``), row-major, appended per batch;
* ``iris.txt`` - the row's IRI, one per line, in the same order;
* ``meta.json`` - ``{scope, model, dims, vector_dtype, shard: [i, n], count, iris_bytes, cursor,
  complete, limit?}``, replaced atomically after both files are flushed. On reopening, both files
  are cut back to what ``meta.json`` counts, so a crash mid-batch loses that batch and no more.
"""

from __future__ import annotations

import json
import os

import numpy as np

UUID_SPACE = 1 << 128


def shard_bounds(shard: tuple[int, int]) -> tuple[bytes | None, bytes | None]:
    """The 16-byte key range ``[lower, bound)`` of shard ``i`` of ``n`` (``None`` = open end)."""
    i, n = shard
    lower = (i * UUID_SPACE // n).to_bytes(16, "big") if i else None
    bound = ((i + 1) * UUID_SPACE // n).to_bytes(16, "big") if i + 1 < n else None
    return lower, bound


def parse_shard(text: str | None) -> tuple[int, int]:
    if not text:
        return 0, 1
    i, _, n = text.partition("/")
    shard = int(i), int(n)
    if not 0 <= shard[0] < shard[1]:
        raise SystemExit(f"--shard {text}: expected I/N with 0 <= I < N")
    return shard


class ShardWriter:
    """The file counterpart of ``VectorWriter``: same ``append`` / ``finish`` / ``cursor`` surface."""

    def __init__(self, out: str, scope: str, dims: int, *, vector_dtype: str = "float32", model: str | None = None,
                 shard: tuple[int, int] = (0, 1), restart: bool = False):
        os.makedirs(out, exist_ok=True)
        self.out, self.dtype = out, np.dtype(vector_dtype)
        self.meta_path = os.path.join(out, "meta.json")
        fresh = {"scope": scope, "model": model, "dims": dims, "vector_dtype": vector_dtype, "shard": list(shard),
                 "count": 0, "iris_bytes": 0, "cursor": None, "complete": False}
        if os.path.exists(self.meta_path) and not restart:
            with open(self.meta_path) as f:
                self.meta = json.load(f)
            got = {k: self.meta[k] for k in ("scope", "model", "dims", "vector_dtype", "shard")}
            want = {k: fresh[k] for k in got}
            if got != want:
                raise SystemExit(f"{out} was started as {got}, not {want}; use another --out, or --restart")
        else:
            self.meta = fresh
        self._pending = 0  # VectorWriter's surface; nothing is ever held back here
        self.vectors = self._open("vectors.bin", self.meta["count"] * dims * self.dtype.itemsize)
        self.iris = self._open("iris.txt", self.meta["iris_bytes"])
        self._save()

    def _open(self, name: str, committed: int):
        path = os.path.join(self.out, name)
        handle = open(path, "r+b" if os.path.exists(path) else "w+b")
        handle.truncate(committed)  # drop whatever a crash left past the last commit
        handle.seek(committed)
        return handle

    @property
    def count(self) -> int:
        return self.meta["count"]

    @property
    def cursor(self) -> str | None:
        return self.meta["cursor"]

    @property
    def complete(self) -> bool:
        return self.meta["complete"]

    def append(self, iris: list[str], vectors: np.ndarray) -> None:
        if not iris:
            return
        rows = np.ascontiguousarray(vectors, dtype=self.dtype)
        if rows.shape != (len(iris), self.meta["dims"]):
            raise ValueError(f"expected {len(iris)} x {self.meta['dims']} vectors, got {rows.shape}")
        text = "".join(iri + "\n" for iri in iris).encode()
        self.vectors.write(rows.tobytes())
        self.iris.write(text)
        for handle in (self.vectors, self.iris):
            handle.flush()
            os.fsync(handle.fileno())
        self.meta["count"] += len(iris)
        self.meta["iris_bytes"] += len(text)
        self.meta["cursor"] = iris[-1]
        self._save()

    def finish(self) -> None:
        self.meta["complete"] = True
        self._save()

    def close(self) -> None:
        self.vectors.close()
        self.iris.close()

    def _save(self) -> None:
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.meta_path)


def read_shard_meta(path: str) -> dict:
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def load_shards(store: str, scope: str, dirs: list[str], *, restart: bool = False, chunk_rows: int = 4096,
                rows: int = 65536) -> int:
    """Writes every shard of ``scope`` into ``store`` in key order; returns the vectors written."""
    from qleverlux.vectors.lmdb_vector_store import VectorWriter

    metas = sorted(((read_shard_meta(d), d) for d in dirs), key=lambda pair: pair[0]["shard"][0])
    metas = [(m, d) for m, d in metas if m["scope"] == scope]
    if not metas:
        raise SystemExit(f"none of the shard directories given is for {scope}")
    first = metas[0][0]
    for meta, path in metas:
        if not meta["complete"]:
            raise SystemExit(f"{path} is not finished ({meta['count']:,} vectors so far)")
        for key in ("model", "dims", "vector_dtype"):
            if meta[key] != first[key]:
                raise SystemExit(f"{path} has {key} {meta[key]!r}, but {metas[0][1]} has {first[key]!r}")
        if meta["shard"][1] != first["shard"][1]:
            raise SystemExit(f"{path} is a shard of {meta['shard'][1]}, {metas[0][1]} of {first['shard'][1]}")
    found = [meta["shard"][0] for meta, _ in metas]
    total = first["shard"][1]
    if found != list(range(total)):
        missing = sorted(set(range(total)) - set(found))
        raise SystemExit(f"{scope}: need shards 0..{total - 1} once each; missing {missing}, got {found}")
    if any(meta.get("limit") for meta, _ in metas):
        print(f"{scope}: loading --limit trial shards, not the whole scope", flush=True)

    writer = VectorWriter(store, scope, first["dims"], chunk_rows=chunk_rows, vector_dtype=first["vector_dtype"],
                          model=first["model"], restart=restart)
    if writer.count or writer.complete:
        writer.close()
        raise SystemExit(f"{store} already has {writer.count:,} {scope} vectors; --restart to load them again")
    dtype = np.dtype(first["vector_dtype"])
    for meta, path in metas:
        count, dims = meta["count"], meta["dims"]
        if not count:
            continue
        matrix = np.memmap(os.path.join(path, "vectors.bin"), dtype=dtype, mode="r", shape=(count, dims))
        with open(os.path.join(path, "iris.txt"), "rb") as f:
            iris = f.read(meta["iris_bytes"]).decode().splitlines()
        if len(iris) != count:
            raise SystemExit(f"{path}: {len(iris):,} IRIs for {count:,} vectors")
        for start in range(0, count, rows):
            writer.append(iris[start : start + rows], matrix[start : start + rows])
        print(f"{scope}: shard {meta['shard'][0]}/{total} loaded, {writer.count + writer._pending:,} vectors",
              flush=True)
        del matrix
    if any(meta.get("limit") for meta, _ in metas):
        writer.meta["limit"] = sum(meta.get("limit") or meta["count"] for meta, _ in metas)
    writer.finish()
    loaded = writer.count
    writer.close()
    return loaded
