"""Related lists: what else is connected to this record, and how.

The count query for each scope/list is precompiled by the catalogue; serving a
request is substituting the target URI in, then turning each non-zero column
into a link to the search that would show those records.
"""

from __future__ import annotations

import urllib.parse

from fastapi.responses import JSONResponse

from qleverlux.errors import run_query
from qleverlux.presentation import envelopes


class RelatedListService:
    def __init__(self, settings, catalogue, qlever, uris):
        self.settings = settings
        self.catalogue = catalogue
        self.qlever = qlever
        self.uris = uris

    async def related_list(self, scope, name, uri, page=1):
        """?name=relatedToAgent&uri=(uri-of-record)"""
        scope = scope.value if hasattr(scope, "value") else scope
        xuri = urllib.parse.quote(uri)
        mt = self.settings.mt_uri
        js = envelopes.collection_page(
            f"{mt}api/related-list/{scope}?name={name}&page={page}&uri={xuri}"
        )
        js["next"] = (
            f"{mt}api/related-list/{scope}?name={name}&page={page + 1}&uri={xuri}"
        )
        # get query from the catalogue's sparql cache and substitute in the uri
        # the uri arrives in this deployment's form; the index only knows data URIs
        target = self.uris.inbound(uri)
        spq = self.catalogue.related_list_sparql[scope][name]
        spq = spq.replace("V_TARGET_URI", target)

        res = await run_query(self.qlever, spq)

        names = [x[1:].replace("_", "-") for x in res["variables"]]
        for r in res["results"]:
            related = r[0]
            if related is None:
                print(res)
                continue
            luri = self.uris.outbound(related)
            counts = list(zip(names[2:], [x if x else 0 for x in r][2:]))
            counts.sort(key=lambda x: x[1], reverse=True)
            for k, v in counts:
                if not v:
                    break
                label = self.catalogue.related_list_names.get(
                    k, f"UNKNOWN RELATED LIST: {k}"
                )
                qscope = self.catalogue.related_list_scopes[scope][name][k]

                # FROM is the related record (the path back to this scope),
                # TO is the record the list was asked about
                qjstr = (
                    self.catalogue.related_list_json[scope][name][k]
                    .replace("V_FROM_URI", related)
                    .replace("V_TO_URI", target)
                )
                qjstr = urllib.parse.quote(qjstr)

                entry = envelopes.collection(
                    f"{mt}api/search-estimate/{qscope}?q={qjstr}",
                    total=v,
                    first=f"{mt}api/search/{qscope}?page=1&q={qjstr}",
                    value=luri,
                )
                entry["name"] = label
                js["orderedItems"].append(entry)

        return JSONResponse(content=js)
