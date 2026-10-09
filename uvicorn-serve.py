#!/usr/bin/env python
"""Development server: plain HTTP on port 5001, single process."""

from qleverlux.server import serve_dev

if __name__ == "__main__":
    serve_dev()
