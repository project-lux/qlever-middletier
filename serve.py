#!/usr/bin/env python
"""Production server: hypercorn, HTTPS/2, single process."""

from qleverlux.server import serve_https

if __name__ == "__main__":
    serve_https()
