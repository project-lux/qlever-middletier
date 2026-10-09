"""Record counts per class, for the search-scope estimates."""

from __future__ import annotations

from fastapi.responses import JSONResponse

from qleverlux.errors import run_query

COUNT_QUERY = "SELECT ?class (COUNT(?class) as ?count) {?what a ?class} GROUP BY ?class"

PORTAL_COUNT_QUERY = """PREFIX lux: <https://lux.collections.yale.edu/ns/> \
SELECT ?class (COUNT(?class) as ?count) \
            WHERE {{?what a ?class ; lux:source lux:{portal} . }} GROUP BY ?class"""


class StatsService:
    def __init__(self, settings, catalogue, qlever):
        self.settings = settings
        self.catalogue = catalogue
        self.qlever = qlever

    async def stats(self):
        portal = self.catalogue.translator.portal
        if portal is not None:
            spq = PORTAL_COUNT_QUERY.format(portal=portal)
        else:
            spq = COUNT_QUERY
        # This will always be in the ALRU cache
        res = await run_query(self.qlever, spq)

        vals = {}
        for r in res["results"]:
            vals[r[0].rsplit("/")[-1].lower()] = r[1]
        cts = {s: vals.get(s, 0) for s in self.catalogue.lux_config.scopes}
        return JSONResponse(content={"estimates": {"searchScopes": cts}})
