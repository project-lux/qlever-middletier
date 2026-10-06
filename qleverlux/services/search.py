"""Search and search-estimate.

A query arrives either as a LUX JSON query or as a simple string; both end up
as a luxql tree, which the translator turns into SPARQL.

Pagination is deliberately coarse. SPARQL is issued at offsets rounded down to
a multiple of 60 so that nearby pages share an ``alru_cache`` entry, and the
exact page is sliced out of the results in Python.
"""

from __future__ import annotations

import urllib.parse

import ujson as json
from fastapi.responses import JSONResponse
from luxql.string_parser import QueryParser

from qleverlux.enums import MULTI_SCOPE
from qleverlux.errors import BadQuery, run_query
from qleverlux.presentation import envelopes
from qleverlux.query.catalogue import MultiScopeError, read_multi_branches

#: SPARQL offsets are rounded down to a multiple of this, to share cache entries.
OFFSET_GRANULARITY = 60


class SearchService:
    def __init__(self, settings, catalogue, qlever, uris):
        self.settings = settings
        self.catalogue = catalogue
        self.qlever = qlever
        self.uris = uris
        self.query_parser = QueryParser()

    # -- translation -----------------------------------------------------

    def parse_query(self, q, scope):
        """A JSON query, or a simple string query parsed into one."""
        try:
            jq = json.loads(q)
            assert type(jq) is dict
        except Exception:
            # fall back to trying to parse simple text query
            qp = self.query_parser.parse(q)
            qjs = qp.to_json()
            k = list(qjs.keys())[0]
            jq = {"_scope": scope}
            jq[k] = qjs[k]
        return jq

    def resolve_sort(self, scope, sort, q):
        """The predicate to sort by, or "relevance" when there is no usable one.

        A multi search has no sorts table of its own, and cannot: a sort only
        works across a union if every branch's scope maps the key to the *same*
        predicate. ``archiveSortId`` does, because ``lux:sortIdentifier`` is
        unscoped. Anything the branches disagree on - or that only some of them
        have - falls back to relevance rather than erroring, so a multi search
        can simply be unsorted.
        """
        if scope != MULTI_SCOPE:
            return self.catalogue.sorts.get(scope, {}).get(sort, "relevance")

        try:
            branches = json.loads(q).get("OR", [])
            scopes = [
                b.get("_scope")
                for b in branches
                if isinstance(b, dict) and b.get("_scope")
            ]
        except Exception:
            scopes = []
        preds = [self.catalogue.sorts.get(s, {}).get(sort) for s in scopes]
        if scopes and all(preds):
            # De-duplicated: one predicate if every branch agrees (as
            # archiveSortId does), otherwise one per branch, which the
            # translator turns into a UNION so each scope sorts by its own.
            unique = list(dict.fromkeys(preds))
            return unique[0] if len(unique) == 1 else unique
        if sort and sort != "relevance":
            print(
                f"multi search: {sort!r} is not defined for every scope in "
                f"{sorted(set(scopes))}, using relevance"
            )
        return "relevance"

    def is_multi(self, scope, jq):
        """A search is multi if the path says so or the query declares it."""
        return scope == MULTI_SCOPE or jq.get("_scope") == MULTI_SCOPE

    def make_sparql_query(
        self, scope, q, page=1, pageLength=0, sort="relevance", order="DESC"
    ):
        if pageLength < 1:
            pageLength = self.settings.page_length
        offset = (page - 1) * pageLength
        soffset = (offset // OFFSET_GRANULARITY) * OFFSET_GRANULARITY
        q = self.uris.inbound(q)
        jq = self.parse_query(q, scope)
        # A query luxql rejects, or one whose text is entirely stopwords, is the
        # caller's mistake: raise so it surfaces as a 400 rather than being
        # swallowed into a None query and reported as a QLever timeout.
        translator = self.catalogue.translator
        try:
            if self.is_multi(scope, jq):
                branches = read_multi_branches(self.catalogue.json_reader, jq)
            else:
                parsed = self.catalogue.json_reader.read(jq, scope)
        except (MultiScopeError, ValueError) as e:
            raise BadQuery(str(e)) from e
        try:
            if self.is_multi(scope, jq):
                spq = translator.translate_multi_search(
                    branches, offset=soffset, sort=sort, order=order
                )
            else:
                spq = translator.translate_search(
                    parsed, scope=scope, offset=soffset, sort=sort, order=order
                )
        except ValueError as e:
            raise BadQuery(str(e)) from e
        except Exception as e:
            print(f"Error translating search: {e}")
            return None
        return spq.get_text()

    def translate_string_query(self, scope, q):
        """Simple string query -> the equivalent LUX JSON query."""
        js = {"_scope": scope}
        try:
            qp = self.query_parser.parse(q)
            # now translate AST into JSON query
            qjs = qp.to_json()
            k = list(qjs.keys())[0]
            js[k] = qjs[k]
        except Exception:
            js["AND"] = [{"text": q}]
        return js

    # -- endpoints -------------------------------------------------------

    async def search(self, scope, q, page=1, pageLength=0, sort="relevance:desc"):
        scope = scope.value
        page = int(page)
        pageLength = int(pageLength)
        if pageLength < 1:
            pageLength = self.settings.page_length
        # must default pageLength first: offset has to match the one
        # make_sparql_query computes, or the slice below is taken from the
        # wrong place in the result set
        offset = (page - 1) * pageLength
        sort = sort.strip()
        if sort:
            try:
                sort, ascdesc = sort.split(":")
                ascdesc = ascdesc.upper().strip()
                sort = sort.strip()
            except Exception:
                ascdesc = "ASC"
        else:
            sort = "relevance"
            ascdesc = "DESC"
        pred = self.resolve_sort(scope, sort, q)
        uq = urllib.parse.quote(q)

        qt = self.make_sparql_query(scope, q, page, pageLength, pred, ascdesc)

        # print("---query---")
        # print(qt)
        res = await run_query(self.qlever, qt)
        # print(res["time"])

        mt = self.settings.mt_uri
        js = envelopes.collection_page(
            f"{mt}api/search/{scope}?q={uq}&page=1",
            part_of=envelopes.collection(
                f"{mt}api/search-estimate/{scope}?q={uq}",
                total=res["total"],
                label="Search Results",
                summary="Description of Search Results",
            ),
        )
        js["_timing"] = res["time"]
        # FIXME: do next and prev

        start = offset % OFFSET_GRANULARITY
        for r in res["results"][start : start + pageLength]:
            js["orderedItems"].append(
                {"id": self.uris.outbound_record(r[0]), "type": "Object"}
            )
        return JSONResponse(content=js)

    async def estimate(self, scope, q={}, page=1):
        uq = urllib.parse.quote(q)
        mt = self.settings.mt_uri
        js = {"@context": envelopes.SEARCH_CONTEXT}
        js.update(
            envelopes.collection(
                f"{mt}api/search/{scope}?q={uq}",
                total=0,
                label="Search Results",
                summary="Description of Search Results",
            )
        )
        qt = self.make_sparql_query(scope, q)
        print(qt)

        res = await run_query(self.qlever, qt)
        js["totalItems"] = res["total"]
        js["_timing"] = res["time"]
        return JSONResponse(content=js)
