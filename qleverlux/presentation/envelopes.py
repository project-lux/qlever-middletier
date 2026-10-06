"""ActivityStreams envelopes for search, facet and related-list responses.

The LUX API returns ``OrderedCollection`` / ``OrderedCollectionPage`` wrappers
around what are really result URIs and counts. These are the primitives; each
service fills in the ids and items.
"""

from __future__ import annotations

SEARCH_CONTEXT = "https://linked.art/ns/v1/search.json"


def collection(uri, total=0, label=None, summary=None, first=None, value=None):
    """An ``OrderedCollection``: a result set someone could page through."""
    js = {"id": uri, "type": "OrderedCollection"}
    if label is not None:
        js["label"] = {"en": [label]}
    if summary is not None:
        js["summary"] = {"en": [summary]}
    if first is not None:
        js["first"] = {"id": first, "type": "OrderedCollectionPage"}
    if value is not None:
        js["value"] = value
    js["totalItems"] = total
    return js


def collection_page(uri, part_of=None, items=None, context=True):
    """An ``OrderedCollectionPage``: one page of results."""
    js = {}
    if context:
        js["@context"] = SEARCH_CONTEXT
    js["id"] = uri
    js["type"] = "OrderedCollectionPage"
    if part_of is not None:
        js["partOf"] = part_of
    js["orderedItems"] = items if items is not None else []
    return js


def page_link(uri):
    return {"id": uri, "type": "OrderedCollectionPage"}
