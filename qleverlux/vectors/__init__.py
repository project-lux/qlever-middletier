"""The LMDB + USearch vector store behind the ``similar`` search term.

The middle tier does not import this: it reaches vectors through QLever's
``SERVICE`` call to the ``../vectors/sparql_service`` sidecar. The store lives
here so that it is versioned; the sidecar and ``lux_embed`` import it through
the ``lmdb_vector_store`` shim in that checkout. It needs ``numpy``, ``lmdb``
and the patched USearch build (``view_ram_buffer``), none of which the middle
tier itself requires.
"""
