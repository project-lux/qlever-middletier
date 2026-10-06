#!/usr/bin/env python
"""Production server: hypercorn with several worker processes."""

from qleverlux.server import run_workers

if __name__ == "__main__":
    run_workers()
