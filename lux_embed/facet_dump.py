"""Stage 1 offline: extract the incoming facets once, where QLever is, and read them back elsewhere.

``embed --qlever`` asks QLever for each batch's facets as it goes, so the embedding machine must
reach the endpoint. When it cannot (a GPU cluster, with the record LMDB copied over), run
``python -m lux_embed facets --qlever URL --out facets.lmdb`` beside QLever, copy the directory,
and give ``embed`` / ``cards`` ``--facets facets.lmdb`` instead of ``--qlever``. The cards come out
the same: what is stored is exactly what :meth:`Facets.fetch` returns, already ranked and capped,
so the document frequencies (``facet_df.json``) are only needed on the extracting side.

Layout of ``facets.lmdb`` (an LMDB directory):

* sub-DB ``facets:<scope>`` - key = 16-byte UUID of the target, value = zlib JSON
  ``{kind: [value UUID, ...], span: [start, end]}``, a span being one of ``DATE_KINDS`` (``dates``,
  and for agents ``published``). Values are stored as UUIDs because that is all the label store
  needs to resolve them. Targets with no facets have no entry.
* sub-DB ``meta`` - key = scope, value = JSON ``{"done", "total", "complete", "kinds", "dates"}``, ``dates``
  listing the spans extracted.
* ``targets-<scope>.txt`` - every target IRI, one per line, fetched once so a resumed run walks the
  same list; ``done`` counts a committed prefix of it.
* ``facet_df.json`` - the document-frequency cache :class:`Facets` keeps.

Needs only ``lmdb`` and the standard library, so it runs in any Python 3.10+ with ``lmdb``
installed - the extracting machine does not need torch or the record LMDB.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import lmdb

from .facets import DATE_KINDS, FACET_BATCH, PREFIX, SCOPE_FACETS, Facets, QLever
from .records import uuid_bytes, uuid_of

FACET_SCOPES = tuple(SCOPE_FACETS)  # work, agent, set, place
MAP_SIZE = 1 << 40


def _open_env(path: str, readonly: bool = False) -> lmdb.Environment:
    if readonly:
        return lmdb.open(path, readonly=True, lock=False, readahead=False, max_dbs=8, subdir=True)
    return lmdb.open(path, map_size=MAP_SIZE, max_dbs=8, subdir=True)


def _encode(facets: dict) -> bytes:
    out = {}
    for kind, values in facets.items():
        out[kind] = list(values) if kind in DATE_KINDS else [uuid_of(v) for v in values]
    return zlib.compress(json.dumps(out, separators=(",", ":")).encode(), 6)


def _decode(raw: bytes) -> dict:
    return json.loads(zlib.decompress(raw))


# -- extraction ------------------------------------------------------------------------------------


def _download_targets(qlever: QLever, facets: Facets, path: str) -> int:
    """Writes every target that has at least one incoming join to ``path``; returns how many.

    Streamed straight to disk: a full LUX scope is millions of IRIs, too many to hold as a TSV
    string and a list at once. Targets without any incoming join could only ever get an empty
    entry, so they are not enumerated at all.
    """
    joins = " UNION ".join(f"{{ {join} }}" for join, _ in facets.plans)
    query = f"SELECT DISTINCT ?target WHERE {{ {joins} }}"
    body = urllib.parse.urlencode({"query": PREFIX + query}).encode()
    request = urllib.request.Request(qlever.endpoint, data=body, headers={"Accept": "text/tab-separated-values"})
    tmp = path + ".tmp"
    print("listing targets (one query; QLever computes it before sending anything)...", flush=True)
    with urllib.request.urlopen(request, timeout=max(qlever.timeout, 600)) as response, open(tmp, "wb") as raw:
        received, reported = 0, time.time()
        while chunk := response.read(1 << 20):
            raw.write(chunk)
            received += len(chunk)
            if time.time() - reported > 10:
                reported = time.time()
                print(f"listing targets: {received / (1 << 20):,.0f} MB received", flush=True)
    count = 0
    with open(tmp) as tsv, open(path + ".iris", "w") as out:
        next(tsv, None)  # header
        for line in tsv:
            cell = line.rstrip("\n")
            if cell.startswith("<") and cell.endswith(">"):
                out.write(cell[1:-1] + "\n")
                count += 1
    os.replace(path + ".iris", path)
    os.remove(tmp)
    return count


def _batches(path: str, skip: int, size: int):
    with open(path) as lines:
        for _ in range(skip):
            next(lines)
        batch = []
        for line in lines:
            batch.append(line.rstrip("\n"))
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


def dump_scope(qlever: QLever, scope: str, out: str, batch: int = FACET_BATCH, workers: int = 4,
               restart: bool = False) -> None:
    """Extracts ``scope``'s facets into ``out``, resuming after the last committed batch."""
    os.makedirs(out, exist_ok=True)
    facets = Facets(qlever, scope, cache_dir=out)
    env = _open_env(out)
    db, meta_db = env.open_db(f"facets:{scope}".encode()), env.open_db(b"meta")
    with env.begin(db=meta_db) as txn:
        raw = txn.get(scope.encode())
    meta = json.loads(raw) if raw and not restart else None
    targets_path = os.path.join(out, f"targets-{scope}.txt")

    if not facets.enabled:
        print(f"{scope}: the endpoint has none of the {scope} facet predicates; nothing to extract", flush=True)
        env.close()
        return
    if meta and meta.get("complete"):
        print(f"{scope}: already complete ({meta['done']:,} targets); --restart to redo it", flush=True)
        env.close()
        return
    if meta is None or not os.path.exists(targets_path):
        if meta is not None:
            print(f"{scope}: {targets_path} is missing; starting again", flush=True)
        with env.begin(write=True) as txn:
            txn.drop(db, delete=False)
        started = time.time()
        total = _download_targets(qlever, facets, targets_path)
        print(f"{scope}: {total:,} targets in {time.time() - started:,.0f}s -> {targets_path}", flush=True)
        meta = {"done": 0, "total": total, "complete": False,
                "kinds": sorted({kind for _, usable in facets.plans for kind in usable}),
                "dates": sorted(facets.date_spans)}
        with env.begin(write=True, db=meta_db) as txn:
            txn.put(scope.encode(), json.dumps(meta).encode())
    else:
        print(f"{scope}: resuming after {meta['done']:,} of {meta['total']:,} targets", flush=True)

    def commit(size: int, found: dict) -> None:
        meta["done"] += size
        with env.begin(write=True) as txn:
            for target, values in found.items():
                if values:
                    txn.put(uuid_bytes(target), _encode(values), db=db)
            txn.put(scope.encode(), json.dumps(meta).encode(), db=meta_db)

    started, done_at_start, last_report = time.time(), meta["done"], 0.0
    # Batches run concurrently but commit in order, so "done" is always a prefix of the target file.
    with ThreadPoolExecutor(workers) as pool:
        pending: deque = deque()
        for targets in _batches(targets_path, meta["done"], batch):
            pending.append((len(targets), pool.submit(facets.fetch, targets)))
            while len(pending) > 2 * workers or (pending and pending[0][1].done()):
                size, future = pending.popleft()
                commit(size, future.result())
            now = time.time()
            if now - last_report > 30:
                last_report = now
                rate = (meta["done"] - done_at_start) / max(now - started, 1e-9)
                left = (meta["total"] - meta["done"]) / rate if rate else 0
                print(f"{scope}: {meta['done']:,}/{meta['total']:,} targets ({rate:,.0f}/s, "
                      f"~{left / 3600:,.1f}h left)", flush=True)
        while pending:
            size, future = pending.popleft()
            commit(size, future.result())

    meta["complete"] = True
    with env.begin(write=True) as txn:
        txn.put(scope.encode(), json.dumps(meta).encode(), db=meta_db)
        stored = txn.stat(db)["entries"]
    env.sync(True)
    env.close()
    print(f"{scope}: complete, {stored:,} of {meta['total']:,} targets have facets "
          f"({time.time() - started:,.0f}s)", flush=True)


# -- reading ---------------------------------------------------------------------------------------


class FacetDump:
    """Read side: the same ``fetch`` as :class:`Facets`, answered from the extracted LMDB."""

    def __init__(self, path: str, scope: str):
        self.scope = scope
        self.env = _open_env(path, readonly=True)
        try:
            meta_db = self.env.open_db(b"meta", create=False)
            with self.env.begin(db=meta_db) as txn:
                raw = txn.get(scope.encode())
        except lmdb.NotFoundError:
            raw = None
        self.meta = json.loads(raw) if raw else None
        self.db = self.env.open_db(f"facets:{scope}".encode(), create=False) if self.meta else None
        if self.meta and not self.meta.get("complete"):
            print(f"facets: {path} has only {self.meta['done']:,} of {self.meta['total']:,} {scope} "
                  f"targets extracted; the rest of the cards go without", flush=True)

    @property
    def enabled(self) -> bool:
        return self.db is not None

    def fetch(self, targets: list[str]) -> dict[str, dict]:
        out = {}
        with self.env.begin(db=self.db) as txn:
            for target in targets:
                try:
                    raw = txn.get(uuid_bytes(target))
                except ValueError:  # not a UUID-keyed IRI
                    continue
                if raw is not None:
                    out[target] = _decode(raw)
        return out
