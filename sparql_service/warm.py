"""Reads what searches touch into the page cache: ``python -m sparql_service.warm <store> --budget <bytes>``.

The sidecar runs this as a child process at startup, for two reasons. The sidecar opens the store
with readahead off, because an HNSW walk faults pages in random order and readahead would fetch
128 KB for every 4 KB it uses. And py-lmdb touches every page of a value it returns. Together
they mean that warming through the sidecar's own environment fetches a 100 MB vector chunk one
page at a time. LMDB also forbids opening one environment twice in a process. So a child opens it
with readahead on, where the same touches stream from disk. The page cache it fills is the
sidecar's too, and the sidecar loads the encoder meanwhile.

Only live data is read, in the order a search needs it: every scope's graph (the HNSW walk), then
the IRI maps (one lookup per pushed-down candidate), then the vector chunks (exact scoring). Free
pages, such as old graph generations, are never read. Reading stops when the budget runs out, so
on a small machine it is the vectors that go unwarmed.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import mmap
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import lmdb
import numpy as np

from qleverlux.vectors.lmdb_vector_store import MAX_DBS, _open_dbs, scopes_in

log = logging.getLogger("sparql_service.warm")


def warm(path: str, budget: int, threads: int = 8) -> dict[str, int]:
    """Warms up to ``budget`` bytes; returns the bytes warmed per phase."""
    env = lmdb.open(path, readonly=True, max_dbs=MAX_DBS, subdir=True, map_size=1 << 40)  # readahead on
    psize = env.stat()["psize"]

    def size(txn, db) -> int:
        stat = txn.stat(db)
        return (stat["branch_pages"] + stat["leaf_pages"] + stat["overflow_pages"]) * psize

    def touch(db, key: bytes) -> None:
        with env.begin(buffers=True) as txn:
            value = txn.get(key, db=db)
            if value is not None:  # py-lmdb prefaults values already; this makes sure of it
                np.frombuffer(value, dtype=np.uint8)[:: mmap.PAGESIZE].sum()

    def walk(db) -> None:
        # One thread per map: py-lmdb drops the GIL on every cursor step, so threads walking
        # cached pages fight over it (8 threads were 27x slower than one). A walk faulting cold
        # pages still gets readahead.
        with env.begin(buffers=True) as txn, txn.cursor(db=db) as cursor:
            cursor.first()
            deque(cursor.iternext(keys=True, values=False), maxlen=0)

    graphs, maps, vectors = [], [], []
    # Handles opened after a transaction began are not valid in it, so open them all first.
    handles = {scope: _open_dbs(env, scope, create=False) for scope in scopes_in(env)}
    with env.begin(buffers=True) as txn:
        for scope, dbs in handles.items():
            if not txn.stat(dbs["graph"])["entries"]:
                log.warning("vector store: %s has no graph; not warmed", scope)
                continue
            graphs.append((size(txn, dbs["graph"]), [lambda db=dbs["graph"]: touch(db, b"safegraphs")]))
            for name in ("id_to_iri", "iri_to_id"):
                maps.append((size(txn, dbs[name]), [lambda db=dbs[name]: walk(db)]))

            with txn.cursor(db=dbs["vectors"]) as cursor:
                chunks = [bytes(key) for key in cursor.iternext(keys=True, values=False)]
            if chunks:
                each = size(txn, dbs["vectors"]) // len(chunks)
                vectors.append([(each, [lambda db=dbs["vectors"], key=key: touch(db, key)]) for key in chunks])

    # Small scopes' vectors first, so a short budget leaves whole scopes cold, not every scope partly.
    vectors = list(itertools.chain.from_iterable(sorted(vectors, key=len)))

    spent, warmed = 0, {}
    with ThreadPoolExecutor(threads) as pool:
        for phase, items in (("graphs", graphs), ("IRI maps", maps), ("vectors", vectors)):
            started, jobs, done, skipped = time.perf_counter(), [], 0, 0
            for nbytes, item_jobs in items:
                if spent + nbytes > budget:
                    skipped += nbytes
                    continue
                spent += nbytes
                done += nbytes
                jobs += item_jobs
            if phase == "IRI maps":
                for job in jobs:
                    job()
            else:
                list(pool.map(lambda job: job(), jobs))
            warmed[phase] = done
            log.info("vector store: %s, %.2f GB read into the page cache in %.1fs",
                     phase, done / 1e9, time.perf_counter() - started)
            if skipped:
                log.warning("vector store: %s, %.2f GB not warmed: over the %.1f GB budget",
                            phase, skipped / 1e9, budget / 1e9)
    env.close()
    return warmed


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sparql_service.warm")
    parser.add_argument("store")
    parser.add_argument("--budget", type=int, required=True, help="bytes")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s:     %(message)s", level=logging.INFO, stream=sys.stderr)
    warm(args.store, args.budget, args.threads)


if __name__ == "__main__":
    main()
