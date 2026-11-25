"""HAL link generation.

A record's ``_links`` block says which related searches actually have results.
Deciding that means one query per candidate relation, so this is the expensive
part of serving a record and the reason the HAL cache exists.

Two kinds of candidate:

* an ordinary HAL relation, whose precompiled count query is run and kept if
  the count is non-zero;
* a related list, which has no single query - instead the catalogue's ordered
  cheap probes are tried in turn and the first hit wins.

A block is only cached when every query it needed answered. A probe that
fails - a timeout, a QLever error, a request shed under load - says nothing
about whether the link exists, so the block built around it is served but not
kept, and the next request for the record tries again. Caching it made a
moment of overload permanent: those records lost their links for good.

That is also why the probes are sheddable like any other query. They were
once exempt, so that a block could not be dropped halfway through, which let
every uncached record request add some 6-25 queries past the load limit.
"""

from __future__ import annotations

import urllib.parse

import ujson as json

from qleverlux.enums import scope_for_class


class HalService:
    def __init__(self, settings, catalogue, qlever, hal_cache, uris):
        self.settings = settings
        self.catalogue = catalogue
        self.qlever = qlever
        self.hal_cache = hal_cache
        self.uris = uris

    async def _probe(self, qt):
        """Run one candidate query. A failure comes back as a result with no
        rows and an ``error``, which the caller must not take for "none"."""
        try:
            return await self.qlever.query(qt)
        except Exception as e:
            return {"results": [], "error": str(e), "status": 504}

    async def _related_list_hit(self, entry, uri):
        """(hit, complete): whether any of a related list's probes finds
        something, and whether the answer can be trusted. A hit is conclusive
        whatever failed before it; a miss is only if no probe failed."""
        try:
            probes = self.catalogue.hal_related_list_tests[entry["scope"]][
                entry["relatedList"]
            ]
        except KeyError:
            print(
                f"Missing related list for {entry['scope']} and {entry['relatedList']}"
            )
            return False, True
        complete = True
        for qt in probes:
            res = await self._probe(qt.replace("V_TARGET_URI", uri))
            if res["results"]:
                return True, True
            if "error" in res:
                complete = False
        return False, complete

    async def _count(self, qt, uri):
        """(count, complete): the count a HAL relation's query reports, or 0,
        and False for ``complete`` when the query failed."""
        res = await self._probe(qt.replace("URI-HERE", uri))
        if "error" in res:
            return 0, False
        try:
            res_array = res["results"][0]
            if len(res_array) == 1:
                ttl = res_array[0]
            elif len(res_array):
                ttl = res_array[1]
            else:
                ttl = 0
        except Exception as e:
            print(f"Failed to find total: {e}\n{res}")
            ttl = 0
        if type(ttl) is not int:
            ttl = res["total"]
        return ttl, True

    def _search_href(self, hal, uri):
        """The search URL a HAL relation points at, with the record filled in."""
        info = self.catalogue.hal_queries[hal]
        jq = self.catalogue.queries[info["queryName"]]
        jqs = json.dumps(jq, separators=(",", ":")).replace("URI-HERE", uri)
        return info["template"].replace("{q}", urllib.parse.quote(jqs))

    async def links(self, scope, identifier):
        """The ``_links`` block for one record, from cache or freshly computed."""
        # Where the data URI has no class, one identifier is several records
        # (Wikidata's Q2 is a place, a group and a concept), each with links
        # of its own
        key = identifier if self.uris.classed else f"{identifier}-{scope}"
        cached = self.hal_cache.get(key)
        if cached is not None:
            return cached

        uri = self.uris.data_record(scope, identifier)
        hscope = scope_for_class(scope)

        links = {}
        complete = True
        for hal, qt in self.catalogue.sparql_hal_queries[hscope].items():
            if type(qt) is dict:
                # a related list: no count query, just the ordered probes
                hit, answered = await self._related_list_hit(qt, uri)
                complete = complete and answered
                if not hit:
                    continue
                href = qt["template"].replace("{id}", uri)
            else:
                count, answered = await self._count(qt, uri)
                complete = complete and answered
                if count <= 0:
                    continue
                href = self._search_href(hal, uri)
            links[hal] = {"href": href, "_estimate": 1}

        if complete:
            await self.hal_cache.put(key, links)
        else:
            print(f"HAL links for {key} incomplete (a query failed); served, not cached")
        return links
