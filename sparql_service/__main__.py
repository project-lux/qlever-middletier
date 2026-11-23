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
    parser.add_argument("--no-warm", dest="warm", action="store_false",
                        help="skip loading the encoder and the store before serving (the first queries pay for it)")
    parser.add_argument("--warm-gb", type=float, default=None,
                        help="most of the store to read into the page cache at startup (default: half of RAM)")
    args = parser.parse_args()
    app = create_app(args.store, device=args.device, timeout=args.timeout, exact_threshold=args.exact_threshold,
                     warm=args.warm, warm_bytes=None if args.warm_gb is None else int(args.warm_gb * 1e9))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
