#!/usr/bin/env python
"""Development server: plain HTTP on QLMT_MTPORT, single process."""

from qleverlux.server import serve_dev

if __name__ == "__main__":
    serve_dev()
