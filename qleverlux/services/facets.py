"""Facet values for a query.

A facet name resolves to a predicate - usually through ``config/facets.json``
and the translator's vocabulary, but record type and the two "responsible"
facets are special-cased onto fixed property paths. Each value comes back with
a count and a link to the search that would narrow to it.
"""

from __future__ import annotations

import asyncio
import urllib.parse

import ujson as json
from fastapi.responses import JSONResponse

from qleverlux.errors import run_query
from qleverlux.presentation import envelopes
from qleverlux.services.search import OFFSET_GRANULARITY

#: Facets that do not come from config/facets.json.
FIXED_FACET_PREDICATES = {
    # the collections an item belongs to. Upstream also requires the set to be
    # classified as a collection (aat:300025976), but every set in the index
    # is, so the filter would change nothing - add it if that stops being true
    "responsibleCollections": "lux:itemMemberOfSet",
    "responsibleUnits": "lux:itemMemberOfSet/lux:setCuratedBy/lux:agentMemberOfGroup",
}

#: The query that selects one value of each fixed facet. These have no single
#: search term, so they copy what the front end builds for them (buildQuery in
#: lux-frontend's src/config/facets.ts) and our links match what it runs.
FIXED_FACET_QUERIES = {
    "responsibleCollections": lambda v: {"memberOf": {"id": v}},
    "responsibleUnits": lambda v: {
        "memberOf": {"curatedBy": {"OR": [{"memberOf": {"id": v}}, {"id": v}]}}
    },
}

#: The search term a record type facet value selects with.
RECORD_TYPE_TERM = "recordType"

#: Namespaces stripped from facet values before they are echoed back.
LUX_NS = "https://lux.collections.yale.edu/ns/"
LA_NS = "https://linked.art/ns/terms/"
_STRIP_NAMESPACES = (LUX_NS, LA_NS)


class FacetService:
    def __init__(self, settings, catalogue, qlever, uris):
        self.settings = settings
        self.catalogue = catalogue
        self.qlever = qlever
        self.uris = uris

    def resolve_predicate(self, name, scope):
        """Facet name -> (predicate, search term name)."""
        if name.endswith("RecordType"):
            return "a", RECORD_TYPE_TERM
        if name in FIXED_FACET_PREDICATES:
            return FIXED_FACET_PREDICATES[name], None

        pname = self.catalogue.facets.get(name, None)
        if not pname:
            print(f" *** request for unknown facet {name} ***")
            term = "MISSING"
        else:
            term = pname["searchTermName"]

        translator = self.catalogue.translator
        pred = translator.get_predicate(term, scope)
        if pred == "lux:missed":
            pred = translator.get_leaf_predicate(term, scope)
            if type(pred) is list:
                pred = pred[0]
            if pred == "missed":
                pred = term
        if ":" not in pred and pred != "a":
            pred = f"lux:{pred}"
        return pred, term

    async def facet(self, scope, q, name, page=1, sort="", pageLength=-1):
        if self.settings.facet_delay:
            await asyncio.sleep(self.settings.facet_delay / 1000)
        scope = scope.value

        if pageLength < 1:
            pageLength = self.settings.facet_page_length
        offset = (int(page) - 1) * pageLength
        soffset = (offset // OFFSET_GRANULARITY) * OFFSET_GRANULARITY
        sort = sort.strip()
        uri_sort = ""
        if sort:
            try:
                if ":" not in sort:
                    ascdesc = sort.strip().lower()
                else:
                    sort, ascdesc = sort.split(":")
                    ascdesc = ascdesc.upper().strip()
                    sort = sort.strip()
                uri_sort = f"&sort={ascdesc}"
            except Exception:
                ascdesc = sort.upper().strip()
                uri_sort = f"&sort={ascdesc}"
                sort = ""
        else:
            sort = ""
            ascdesc = ""

        q = self.uris.inbound(q)
        jq = json.loads(q)
        parsed = self.catalogue.json_reader.read(jq, scope)

        uq = urllib.parse.quote(q)
        mt = self.settings.mt_uri
        base = f"{mt}api/facets/{scope}?q={uq}&name={name}"
        js = envelopes.collection_page(
            f"{base}&page={page}&pageLength={pageLength}",
            part_of={"type": "OrderedCollection", "totalItems": 0},
        )

        pred, term = self.resolve_predicate(name, scope)
        spq = self.catalogue.translator.translate_facet(
            parsed, pred, scope=scope, offset=soffset, sort=sort, order=ascdesc
        )
        qt = spq.get_text()
        # print(qt)

        res = await run_query(self.qlever, qt)

        js["partOf"]["totalItems"] = res["total"] + soffset
        js["_timing"] = res["time"]

        start = offset % OFFSET_GRANULARITY
        for r in res["results"][start : start + pageLength]:
            val = r[0]
            ct = r[1]
            if type(val) is str and val.startswith("http"):
                # is a URI
                if pred == "a":
                    if val.startswith(LUX_NS):
                        continue
                    # a class name is a plain value, {"recordType": "Person"}
                    val = val.replace(LA_NS, "")
                    clause = {term: val}
                else:
                    val = self.uris.outbound_record(val)
                    for ns in _STRIP_NAMESPACES:
                        val = val.replace(ns, "")
                    if name in FIXED_FACET_QUERIES:
                        clause = FIXED_FACET_QUERIES[name](val)
                    else:
                        clause = {term: {"id": val}}
            else:
                clause = {term: val, "_comp": "=="}

            nq = {"AND": [clause, jq]}
            # ujson escapes "/" by default, and "http:\/\/..." would slip past
            # the inbound URI rewrite when the link is followed
            qstr = urllib.parse.quote(
                json.dumps(nq, separators=(",", ":"), escape_forward_slashes=False)
            )
            js["orderedItems"].append(
                envelopes.collection(
                    f"{mt}api/search-estimate/{scope}?q={qstr}",
                    total=ct,
                    value=val,
                )
            )

        if page > 1:
            js["prev"] = envelopes.page_link(
                f"{base}&page={page - 1}&pageLength={pageLength}{uri_sort}"
            )
        if (offset + pageLength) < js["partOf"]["totalItems"]:
            js["next"] = envelopes.page_link(
                f"{base}&page={page + 1}&pageLength={pageLength}{uri_sort}"
            )

        return JSONResponse(content=js)
