"""How much recall a smaller HNSW graph costs, measured on a real scope.

Reads a scope's vectors from a vector store (read-only: nothing is written), builds a graph for
each quantization in memory, and compares searches against exact cosine over every vector:

    ../vectors/.venv/bin/python files/graph_quantization.py --store ../vectors/data/mds/vectors.lmdb \\
        --scope concept --quantizations bf16,e4m3,i8,b1 --efs 300,3000,10000 --pools 300,1000,3000

Each search is what the sidecar does: a graph walk for a pool of candidates, then an exact rerank of
the pool with the full vectors. So a coarse graph only has to put the true neighbours somewhere in
the pool, and a bigger pool (more rows read from disk) buys recall back. Recall@K is the share of
the exact top K that the reranked top K contains.

Two kinds of query, because they behave differently:
- text: the encoder's query vectors for ``--texts`` (the store's own model), which sit away from
  every record - the case HNSW finds hardest;
- seeded: the stored vectors of ``--seeds`` random records ("more like this"), the record itself
  left out of both the truth and the results.

Quantizations: bf16 (2 bytes a dimension, what is built today), e4m3 and i8 (1 byte), b1 (1 bit:
the sign of each dimension, compared by Hamming distance). Sizes are the graph's bytes per vector,
neighbour lists included. Needs numpy, lmdb, usearch, and sentence-transformers for text queries.

Fewer dimensions: ``--reductions`` builds the graph on smaller vectors - ``pca256`` projects onto
the top 256 principal components of a sample of the scope, ``trunc256`` keeps the first 256
dimensions (Matryoshka-trained models put the most in the leading ones), ``none`` the full vectors.
``--reranks`` says what the pool is reranked with: ``full`` or ``reduced`` vectors as float32, or
either stored smaller: ``-i8`` (a byte a dimension, scaled per row), ``-sq8`` (a byte, scaled per
dimension) or ``-f16``.
The truth is always exact cosine over the full vectors, so every loss shows up.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

# The vectors venv does not install this repo; run from anywhere, it imports from the checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmdb
import numpy as np
import usearch
from usearch import DenseIndex, DenseMetric, Scalar

from qleverlux.vectors.lmdb_vector_store import MAX_DBS, PAD, _matched_ids, _open_dbs

DEFAULT_TEXTS = [
    "quantum entanglement", "quantum physics", "photosynthesis", "baroque architecture", "string quartets",
    "roman coins", "children's books", "medieval manuscripts", "impressionist painting", "jazz",
    "folk music", "botanical illustration", "science fiction novels", "medieval armour", "textile manufacturing",
    "japanese woodblock prints", "civil rights movement", "volcanoes", "shipbuilding", "astronomy",
    "ceramics glazes", "funeral customs", "printing presses", "beekeeping", "railways",
    "tropical diseases", "silver jewellery", "landscape photography", "opera", "cartography",
]


def load_vectors(path: str, scope: str, limit: int = 0) -> tuple[np.ndarray, dict]:
    env = lmdb.open(path, readonly=True, max_dbs=MAX_DBS, subdir=True, lock=False, map_size=1 << 40)
    dbs = _open_dbs(env, scope, create=False)
    with env.begin(buffers=True) as txn:
        meta = json.loads(bytes(txn.get(b"meta", db=dbs["meta"])))
        dims, rows = meta["dimensions"], meta["chunk_rows"]
        dtype = np.dtype(meta.get("vector_dtype", "float32"))
        count = meta["count"] if not limit else min(limit, meta["count"])
        out = np.empty((count, dims), dtype=np.float32)
        for key, value in txn.cursor(db=dbs["vectors"]):
            first = struct.unpack(">I", bytes(key))[0] * rows
            if first >= count:
                break
            block = np.frombuffer(value, dtype=dtype, offset=PAD).reshape(-1, dims)
            take = min(len(block), count - first)
            out[first:first + take] = block[:take]
    env.close()
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out, meta


def as_stored(name: str, vectors: np.ndarray) -> tuple[DenseMetric, Scalar, np.ndarray]:
    """What USearch is given for each quantization (checked on random data first: i8 wants int8
    input already scaled, b1 one byte of 0/1 per dimension, which it packs)."""
    if name == "b1":
        return DenseMetric.HAMMING, Scalar.U1X8, (vectors > 0).astype(np.uint8)
    if name == "i8":
        scaled = vectors / np.abs(vectors).max(axis=1, keepdims=True) * 127
        return DenseMetric.ANGULAR, Scalar.I8, np.round(scaled).astype(np.int8)
    scalar = {"bf16": Scalar.BF16, "e4m3": Scalar.E4M3, "f16": Scalar.F16}[name]
    return DenseMetric.ANGULAR, scalar, vectors


def reducer(spec: str, vectors: np.ndarray, sample: int = 50_000):
    """A function from full vectors to the reduced, renormalised ones that ``spec`` names."""
    if spec == "none":
        return lambda v: v
    dims = int(spec.removeprefix("pca").removeprefix("trunc"))
    if spec.startswith("trunc"):
        return lambda v: v[:, :dims] / np.linalg.norm(v[:, :dims], axis=1, keepdims=True)
    rows = np.random.default_rng(1).choice(len(vectors), min(sample, len(vectors)), replace=False)
    mean = vectors[rows].mean(axis=0)
    basis = np.linalg.svd(vectors[rows] - mean, full_matrices=False)[2][:dims].T.astype(np.float32)

    def project(v):
        out = (v - mean) @ basis
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    return project


def as_stored_rerank(kind: str, vectors: np.ndarray) -> np.ndarray:
    """``vectors`` as a smaller vector store would give them back, dequantised: ``i8`` one byte a
    dimension scaled per row, ``sq8`` one byte scaled per dimension (each over its own range in
    the data), ``f16`` half floats."""
    if kind == "f16":
        return vectors.astype(np.float16).astype(np.float32)
    if kind == "sq8":
        low, high = vectors.min(axis=0), vectors.max(axis=0)
        step = (high - low) / 255
        return np.round((vectors - low) / step).astype(np.uint8).astype(np.float32) * step + low
    scale = np.abs(vectors).max(axis=1, keepdims=True) / 127
    return np.round(vectors / scale).astype(np.int8).astype(np.float32) * scale


def build(name: str, vectors: np.ndarray, connectivity: int) -> tuple[DenseIndex, int, float]:
    metric, scalar, data = as_stored(name, vectors)
    started = time.perf_counter()
    pool = usearch.Executor.for_ram()
    index = DenseIndex.init_ram(vectors.shape[1], metric, scalar, connectivity=connectivity)
    index.reserve(len(vectors))
    index.on(pool).insert(np.arange(len(vectors), dtype=np.uint64), data)
    index.on(pool).refine()  # as build_index does
    size = len(memoryview(index.on(pool).save_buffer(usearch.Arena(pool))))
    pool.drain()  # an undrained pool keeps every worker spinning
    return index, size, time.perf_counter() - started


def exact_top(vectors: np.ndarray, queries: np.ndarray, k: int, exclude: np.ndarray | None) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 64):
        scores = queries[start:start + 64] @ vectors.T
        if exclude is not None:
            scores[np.arange(len(scores)), exclude[start:start + 64]] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        out[start:start + 64] = np.take_along_axis(top, order, axis=1)
    return out


def evaluate(name, index, graph_queries, rerank_vectors, rerank_queries, truth, ks, pool_size, ef, exclude):
    """Walk the graph with ``graph_queries``, rerank the pool with ``rerank_*``, score against ``truth``."""
    queries = rerank_queries
    _, _, data = as_stored(name, graph_queries)
    arena = usearch.Arena()
    started = time.perf_counter()
    found = []
    for row in range(len(queries)):  # one at a time, on this thread, as the sidecar searches
        match = index.search(data[row:row + 1], pool_size + 1, arena, expansion_factor=max(ef, pool_size + 1))
        ids = _matched_ids(match)[0].astype(np.int64)
        if exclude is not None:
            ids = ids[ids != exclude[row]]
        ranked = ids[np.argsort(-(rerank_vectors[ids] @ queries[row]))]  # the exact rerank
        found.append(ranked)
    per_query = (time.perf_counter() - started) / len(queries) * 1e3
    recalls = {k: np.mean([len(set(found[r][:k]) & set(truth[r][:k])) / k for r in range(len(queries))]) for k in ks}
    return recalls, per_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True)
    parser.add_argument("--scope", default="concept")
    parser.add_argument("--quantizations", default="bf16,e4m3,i8,b1")
    parser.add_argument("--connectivity", type=int, default=16)
    parser.add_argument("--efs", default="300,3000,10000")
    parser.add_argument("--pools", default="300,1000,3000", help="candidates walked for, then reranked exactly")
    parser.add_argument("--ks", default="10,100", help="recall@K reported")
    parser.add_argument("--texts", default="", help="file of text queries, one a line (default: a built-in 30)")
    parser.add_argument("--seeds", type=int, default=200, help="random records as more-like-this queries")
    parser.add_argument("--reductions", default="none", help="none, pcaN, truncN: what the graph is built on")
    parser.add_argument("--reranks", default="full", help="full or reduced, optionally -i8, -sq8 or -f16: what the pool is reranked with")
    parser.add_argument("--limit", type=int, default=0, help="only the first N vectors, for a quick run")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vectors, meta = load_vectors(args.store, args.scope, args.limit)
    ks = [int(k) for k in args.ks.split(",")]
    print(f"{args.scope}: {len(vectors):,} vectors, {vectors.shape[1]}d, model {meta.get('model')}")

    query_sets = []
    texts = [line.strip() for line in open(args.texts) if line.strip()] if args.texts else DEFAULT_TEXTS
    if texts:
        from lux_embed.encoder import Encoder

        encoder = Encoder(device=args.device, model=meta.get("model") or "google/embeddinggemma-2")
        text_queries = np.vstack([encoder.encode_query(t, vectors.shape[1]) for t in texts]).astype(np.float32)
        query_sets.append(("text", text_queries, None))
    if args.seeds:
        seeds = np.random.default_rng(0).choice(len(vectors), args.seeds, replace=False)
        query_sets.append(("seeded", vectors[seeds], seeds))
    truths = {label: exact_top(vectors, q, max(ks), exclude) for label, q, exclude in query_sets}

    for spec in args.reductions.split(","):
        reduce = reducer(spec, vectors)
        reduced = reduce(vectors)
        reduced_queries = {label: reduce(q) for label, q, _ in query_sets}
        if spec != "none":
            print(f"\n=== {spec}: graph on {reduced.shape[1]} dimensions")
        reranks = {}
        for how in args.reranks.split(","):
            base = "full" if how.startswith("full") or spec == "none" else "reduced"
            kind = how.partition("-")[2]
            key = base + ("-" + kind if kind else "")
            if key not in reranks:  # with no reduction, "reduced" is "full"
                source = vectors if base == "full" else reduced
                reranks[key] = (as_stored_rerank(kind, source) if kind else source, base)
        for name in args.quantizations.split(","):
            index, size, seconds = build(name, reduced, args.connectivity)
            print(f"\n{name}: {size / len(vectors):,.0f} bytes a vector ({size / 1e9:.2f} GB), built in {seconds:,.0f} s")
            for label, queries, exclude in query_sets:
                for key, (rerank_vectors, base) in reranks.items():
                    rerank_queries = queries if base == "full" else reduced_queries[label]
                    for pool_size in (int(p) for p in args.pools.split(",")):
                        for ef in (int(e) for e in args.efs.split(",")):
                            recalls, ms = evaluate(name, index, reduced_queries[label], rerank_vectors, rerank_queries,
                                                   truths[label], ks, pool_size, ef, exclude)
                            shown = "  ".join(f"recall@{k} {recalls[k]:.3f}" for k in ks)
                            print(f"   {label:6} rerank {key:10} pool {pool_size:>5}  ef {ef:>6}: {shown}   {ms:6.1f} ms/query")
            del index

if __name__ == "__main__":
    main()
