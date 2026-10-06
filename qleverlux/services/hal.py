"""HAL link generation.

A record's ``_links`` block says which related searches actually have results.
Deciding that means one query per candidate relation, so this is the expensive
part of serving a record and the reason the HAL cache exists.

Two kinds of candidate:

* an ordinary HAL relation, whose precompiled count query is run and kept if
  the count is non-zero;
* a related list, which has no single query - instead the catalogue's ordered
  cheap probes are tried in turn and the first hit wins.

Queries here are issued with ``sheddable=False``: a half-built links block
would be cached and served as if it were complete.
"""

from __future__ import annotations

import urllib.parse

import ujson as json

from qleverlux.enums import scope_for_class


class HalService:
    def __init__(self, settings, catalogue, qlever, hal_cache):
        self.settings = settings
        self.catalogue = catalogue
        self.qlever = qlever
        self.hal_cache = hal_cache

    async def _probe(self, qt):
        """Run one candidate query, treating any failure as "no results"."""
        try:
            return await self.qlever.query(qt, sheddable=False)
        except Exception as e:
            return {"results": [], "error": str(e), "status": 504}

    async def _related_list_hit(self, entry, uri):
        """True if any of a related list's probes finds something."""
        try:
            probes = self.catalogue.hal_related_list_tests[entry["scope"]][
                entry["relatedList"]
            ]
        except KeyError:
            print(
                f"Missing related list for {entry['scope']} and {entry['relatedList']}"
            )
            return False
        for qt in probes:
            res = await self._probe(qt.replace("V_TARGET_URI", uri))
            if res["results"]:
                return True
        return False

    async def _count(self, qt, uri):
        """The count a HAL relation's query reports, or 0."""
        res = await self._probe(qt.replace("URI-HERE", uri))
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
        return ttl

    def _search_href(self, hal, uri):
        """The search URL a HAL relation points at, with the record filled in."""
        info = self.catalogue.hal_queries[hal]
        jq = self.catalogue.queries[info["queryName"]]
        jqs = json.dumps(jq, separators=(",", ":")).replace("URI-HERE", uri)
        return info["template"].replace("{q}", urllib.parse.quote(jqs))

    async def links(self, scope, identifier):
        """The ``_links`` block for one record, from cache or freshly computed."""
        cached = self.hal_cache.get(identifier)
        if cached is not None:
            return cached

        uri = f"{self.settings.data_uri}data/{scope}/{identifier}"
        hscope = scope_for_class(scope)

        links = {}
        for hal, qt in self.catalogue.sparql_hal_queries[hscope].items():
            if type(qt) is dict:
                # a related list: no count query, just the ordered probes
                if not await self._related_list_hit(qt, uri):
                    continue
                href = qt["template"].replace("{id}", uri)
            else:
                if await self._count(qt, uri) <= 0:
                    continue
                href = self._search_href(hal, uri)
            links[hal] = {"href": href, "_estimate": 1}

        await self.hal_cache.put(identifier, links)
        return links
