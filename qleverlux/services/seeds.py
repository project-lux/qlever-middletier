"""Seeds for ``similarRecords`` given as a query rather than as ids.

``{"similarRecords": {"name": "quantum physics"}}`` means "records similar to
the records that query finds". The vector SERVICE call needs those records as
constant IRIs, so the query has to be run before the outer query can be
written - which the translator, having no QLever client, cannot do. This runs
each such subquery as an ordinary search in the relationship's scope, takes the
top ``MAX_SEED_RECORDS`` by relevance, and leaves their URIs on the node as
``seed_records`` for the translator.

Ids (``{"id": ...}`` or an ``OR`` of them) need no lookup and are left alone.
A subquery may itself use ``similarRecords``; it is resolved first, to a
bounded depth.
"""

from __future__ import annotations

from luxql import LuxBoolean, LuxRelationship

from qleverlux.errors import run_query
from qleverlux.query.translator import MAX_SEED_RECORDS, SIMILAR_RECORDS_FIELD, seed_ids

#: How deeply similarRecords may nest inside a similarRecords subquery.
MAX_DEPTH = 3


async def resolve_similar_records(parsed, catalogue, qlever, depth=0):
    """Give every query-valued similarRecords in ``parsed`` its seed URIs.

    Raises ValueError for a subquery luxql rejects or nesting past MAX_DEPTH
    (the caller's mistake), and lets a QLeverError from the seed query through.
    """
    for node in _unresolved(parsed):
        if depth >= MAX_DEPTH:
            raise ValueError(f"similarRecords nested more than {MAX_DEPTH} deep")
        scope = node.provides_scope
        sub = catalogue.json_reader.read(node.children[0].to_json(), scope)
        await resolve_similar_records(sub, catalogue, qlever, depth + 1)
        spq = catalogue.translator.translate_search(
            sub, scope=scope, limit=MAX_SEED_RECORDS, sort="relevance", order="DESC"
        )
        res = await run_query(qlever, spq.get_text())
        node.seed_records = [row[0] for row in res["results"][:MAX_SEED_RECORDS]]


def _unresolved(node):
    """similarRecords nodes whose records are a query still to be run.

    Does not descend into them: their subquery is read and resolved on its own.
    """
    if isinstance(node, LuxRelationship) and node.field == SIMILAR_RECORDS_FIELD:
        if getattr(node, "seed_records", None) is None and seed_ids(
            node.children[0] if node.children else None
        ) is None:
            yield node
        return
    if isinstance(node, (LuxBoolean, LuxRelationship)):
        for child in node.children:
            yield from _unresolved(child)
