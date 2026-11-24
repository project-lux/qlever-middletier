"""Run the vector SERVICE: ``python -m sparql_service --store data/mds/vectors.lmdb``.

Single process on purpose: index pages are shared through the OS page cache anyway, the text
encoder (when used) is loaded once, and searches already run in parallel on worker threads.
"""

import argparse

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m sparql_service")
    parser.add_argument("--store", required=True, help="vector store LMDB built by lux_embed")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", default="auto", help="encoder device for text queries: auto | cuda | mps | cpu")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request deadline, seconds")
    parser.add_argument("--exact-threshold", type=int, default=200_000,
                        help="pushed-down candidates up to this many are scored exactly")
    parser.add_argument("--expansion-factor", type=int, default=10_000,
                        help="HNSW candidates kept in play during a walk (at least top-K x 3 is used); "
                             "higher escapes dense clusters of near-duplicates at a roughly linear cost. "
                             "vec:ef overrides it per query")
    parser.add_argument("--rewalk", type=int, default=10,
                        help="walk the graph again from the best N hits' own vectors and rerank the pool; "
                             "recovers records a text query's walk passes by. vec:rewalk overrides it")
    parser.add_argument("--rewalk-ef", type=int, default=300,
                        help="expansion factor of the second walks (at least top-K x 3 is used); narrow ones "
                             "are cheap, wide ones recover more but cost as much as the first walk again. "
                             "vec:rewalkEf overrides it")
    parser.add_argument("--rerank-factor", type=int, default=None,
                        help="candidates reranked exactly per result wanted (default: 3, or 10 in a scope whose "
                             "graph was built on fewer dimensions than its vectors)")
    parser.add_argument("--exact-below", type=int, default=0,
                        help="score every vector, no graph, in scopes of at most this many records "
                             "(perfect recall, linear cost). vec:exact overrides it")
    parser.add_argument("--no-warm", dest="warm", action="store_false",
                        help="skip loading the encoder and the store before serving (the first queries pay for it)")
    parser.add_argument("--warm-gb", type=float, default=None,
                        help="most of the store to read into the page cache at startup (default: half of RAM)")
    parser.add_argument("--cache-mb", type=float, default=256,
                        help="memory for cached responses, hits and encoded query text")
    args = parser.parse_args()
    app = create_app(args.store, device=args.device, timeout=args.timeout, exact_threshold=args.exact_threshold,
                     cache_bytes=int(args.cache_mb * (1 << 20)), expansion_factor=args.expansion_factor,
                     rewalk=args.rewalk, rewalk_ef=args.rewalk_ef,
                     exact_below=args.exact_below, rerank_factor=args.rerank_factor,
                     warm=args.warm, warm_bytes=None if args.warm_gb is None else int(args.warm_gb * 1e9))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
