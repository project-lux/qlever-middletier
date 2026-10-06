"""qleverlux: the LUX middle tier on a QLever SPARQL endpoint.

Layout:

* ``settings``    - every configurable knob, from env then CLI
* ``query``       - LUX JSON queries -> SPARQL (translator, vocabulary, catalogue)
* ``sparql``      - the SPARQL object model the translator emits
* ``clients``     - QLever, PostgreSQL, LMDB, the AI backend
* ``services``    - what each endpoint actually does
* ``api``         - the routes
* ``presentation``- URI rewriting and ActivityStreams envelopes
* ``app``         - the FastAPI application and the object graph behind it
"""

__version__ = "0.5.0"
