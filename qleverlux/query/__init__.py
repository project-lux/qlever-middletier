"""Query translation: LUX JSON queries in, SPARQL out.

* ``predicates`` - the QLever predicate vocabulary, as data.
* ``text`` - free-text clauses (materialized views, ``ql:has-word``, phrases).
* ``translator`` - walks a parsed luxql tree and emits a query object.
* ``related`` - builds the related-list count queries and their HAL probes.
* ``catalogue`` - loads config/ and queries/ and precompiles everything once.
"""
