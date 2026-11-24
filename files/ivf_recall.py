"""How well an IVF index (clusters on disk, centres in memory) finds a scope's neighbours.

The companion to graph_quantization.py, with the same vectors, queries and exact answers, for the
question that one cannot answer: what if the index does not fit in memory? An IVF index keeps only
the cluster centres in memory; a query picks the ``nprobe`` nearest centres, reads those clusters'
stored codes (contiguous, so sequential reads), scores them, and reranks the best ``pool`` exactly
with the full vectors - as the sidecar does after a graph walk.

    python files/ivf_recall.py --store ../vectors/data/mds/vectors.lmdb --scope concept \\
        --codes sq8,pq64,b1 --nprobes 8,16,32,64,128 --pools 300,1000

Per setting it reports recall@K, the vectors scanned and the bytes of codes read per query (what
the disk would serve), and time per query. Codes: fp16 (2 bytes a dimension), sq8 (1 byte),
pq64 / pq128 (product quantization, that many bytes a vector), b1 (1 bit a dimension, Hamming).

Needs faiss-cpu. Text queries need sentence-transformers to encode; where FAISS and the encoder
live in different environments, encode once with ``--save-text-vectors q.npy`` in the one with the
encoder (it exits after saving) and pass ``--text-vectors q.npy`` in the one with FAISS.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from graph_quantization import DEFAULT_TEXTS, exact_top, load_vectors  # noqa: E402


def encode_texts(texts, meta, dims, device):
    from lux_embed.encoder import Encoder

    encoder = Encoder(device=device, model=meta.get("model") or "google/embeddinggemma-2")
    return np.vstack([encoder.encode_query(t, dims) for t in texts]).astype(np.float32)


def build(codes: str, vectors: np.ndarray, nlist: int, train_size: int):
    import faiss

    d = len(vectors[0])
    rng = np.random.default_rng(0)
    train = vectors[rng.choice(len(vectors), min(train_size, len(vectors)), replace=False)]
    started = time.perf_counter()
    if codes == "b1":
        bits = np.packbits(vectors > 0, axis=1)
        index = faiss.IndexBinaryIVF(faiss.IndexBinaryFlat(d), d, nlist)
        index.train(np.packbits(train > 0, axis=1))
        index.add(bits)
    else:
        quantizer = faiss.IndexFlatIP(d)
        if codes in ("fp16", "sq8"):
            kind = faiss.ScalarQuantizer.QT_fp16 if codes == "fp16" else faiss.ScalarQuantizer.QT_8bit
            index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, kind, faiss.METRIC_INNER_PRODUCT)
        elif codes.startswith("pq"):
            m = int(codes[2:])
            if d % m:
                raise ValueError(f"{codes}: {d} dimensions do not split into {m} parts")
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"unknown codes {codes!r}")
        index.train(train)
        index.add(vectors)
    seconds = time.perf_counter() - started
    sizes = np.array([index.invlists.list_size(i) for i in range(nlist)])
    return index, seconds, sizes


def evaluate(codes, index, vectors, queries, truth, ks, nprobe, pool, exclude):
    import faiss

    index.nprobe = nprobe
    data = np.packbits(queries > 0, axis=1) if codes == "b1" else queries
    stats = faiss.cvar.indexIVF_stats
    stats.reset()
    started = time.perf_counter()
    found = []
    for row in range(len(queries)):  # one at a time, on one thread, as the sidecar would
        _, ids = index.search(data[row:row + 1], pool + 1)
        ids = ids[0][ids[0] >= 0].astype(np.int64)
        if exclude is not None:
            ids = ids[ids != exclude[row]]
        found.append(ids[np.argsort(-(vectors[ids] @ queries[row]))])  # the exact rerank
    per_query = (time.perf_counter() - started) / len(queries) * 1e3
    scanned = stats.ndis / len(queries)
    recalls = {k: np.mean([len(set(found[r][:k]) & set(truth[r][:k])) / k for r in range(len(queries))]) for k in ks}
    return recalls, scanned, scanned * index.code_size, per_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True)
    parser.add_argument("--scope", default="concept")
    parser.add_argument("--codes", default="sq8,pq64,b1")
    parser.add_argument("--nlist", type=int, default=0, help="clusters (default: 4 x sqrt(N), a power of two)")
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--nprobes", default="8,16,32,64,128")
    parser.add_argument("--pools", default="300,1000", help="candidates kept from the codes, then reranked exactly")
    parser.add_argument("--ks", default="10,100")
    parser.add_argument("--texts", default="", help="file of text queries, one a line (default: graph_quantization's 30)")
    parser.add_argument("--text-vectors", default="", help="precomputed text query vectors (.npy), same order")
    parser.add_argument("--save-text-vectors", default="", help="encode the texts, save them here, and exit")
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vectors, meta = load_vectors(args.store, args.scope, args.limit)
    texts = [line.strip() for line in open(args.texts) if line.strip()] if args.texts else DEFAULT_TEXTS
    if args.save_text_vectors:
        np.save(args.save_text_vectors, encode_texts(texts, meta, vectors.shape[1], args.device))
        print(f"saved {len(texts)} text query vectors to {args.save_text_vectors}")
        return

    import faiss

    faiss.omp_set_num_threads(os.cpu_count() or 1)  # build on every core; searches below are one at a time
    nlist = args.nlist or 2 ** round(math.log2(4 * math.sqrt(len(vectors))))
    ks = [int(k) for k in args.ks.split(",")]
    print(f"{args.scope}: {len(vectors):,} vectors, {vectors.shape[1]}d, model {meta.get('model')}, {nlist} clusters")

    query_sets = []
    if texts:
        text_queries = (np.load(args.text_vectors) if args.text_vectors
                        else encode_texts(texts, meta, vectors.shape[1], args.device)).astype(np.float32)
        query_sets.append(("text", text_queries, None))
    if args.seeds:
        seeds = np.random.default_rng(0).choice(len(vectors), args.seeds, replace=False)
        query_sets.append(("seeded", vectors[seeds], seeds))
    truths = {label: exact_top(vectors, q, max(ks), exclude) for label, q, exclude in query_sets}

    for codes in args.codes.split(","):
        index, seconds, sizes = build(codes, vectors, nlist, args.train_size)
        centres = nlist * vectors.shape[1] * (1 / 8 if codes == "b1" else 4)
        print(f"\n{codes}: {index.code_size} bytes a vector, built in {seconds:,.0f} s; in memory: centres "
              f"{centres / 1e6:,.1f} MB; clusters hold {sizes.mean():,.0f} vectors on average, {sizes.max():,} at most")
        faiss.omp_set_num_threads(1)
        for label, queries, exclude in query_sets:
            for pool in (int(p) for p in args.pools.split(",")):
                for nprobe in (int(n) for n in args.nprobes.split(",")):
                    recalls, scanned, read, ms = evaluate(codes, index, vectors, queries, truths[label], ks,
                                                          nprobe, pool, exclude)
                    shown = "  ".join(f"recall@{k} {recalls[k]:.3f}" for k in ks)
                    print(f"   {label:6} pool {pool:>5}  nprobe {nprobe:>4}: {shown}   scanned {scanned:>8,.0f} "
                          f"({read / 1e6:6.2f} MB)   {ms:6.1f} ms/query")
        faiss.omp_set_num_threads(os.cpu_count() or 1)
        del index


if __name__ == "__main__":
    main()
