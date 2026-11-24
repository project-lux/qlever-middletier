"""Related lists: what else is connected to this record, and how.

The count query for each scope/list is precompiled by the catalogue; serving a
request is substituting the target URI in, then turning each non-zero column
into a link to the search that would show those records.

A page counts entries, as the MarkLogic backend does: one per related record
and relation, so a record related in three ways is three entries, adjacent.
Records come back ranked by their total, so the entries can only be numbered
by expanding every record ahead of the page. Each record gives at least one
entry, so ``offset + pageLength + 1`` records always cover the page and show
whether another follows. That count is rounded up to ``RECORD_GRANULARITY`` so
nearby pages issue the same SPARQL and share an ``alru_cache`` entry; ranking
has to count every related record anyway, so the larger LIMIT costs only the
extra rows transferred.
"""

from __future__ import annotations

import urllib.parse

from fastapi.responses import JSONResponse

from qleverlux.errors import run_query
from qleverlux.presentation import envelopes

#: Records requested are rounded up to a multiple of this, to share cache entries.
RECORD_GRANULARITY = 60


class RelatedListService:
    def __init__(self, settings, catalogue, qlever, uris):
        self.settings = settings
        self.catalogue = catalogue
        self.qlever = qlever
        self.uris = uris

    async def related_list(self, scope, name, uri, page=1, pageLength=None):
        """?name=relatedToAgent&uri=(uri-of-record)[&page=N][&pageLength=N]"""
        scope = scope.value if hasattr(scope, "value") else scope
        # A pageLength the caller gave is carried into the page links; the
        # default is left implicit, as the front end requests it.
        length_param = "" if pageLength is None else f"&pageLength={pageLength}"
        if pageLength is None:
            pageLength = self.settings.related_page_length
        offset = (page - 1) * pageLength
        end = offset + pageLength
        records = -(-(end + 1) // RECORD_GRANULARITY) * RECORD_GRANULARITY

        xuri = urllib.parse.quote(uri)
        mt = self.settings.mt_uri

        def page_uri(n):
            return f"{mt}api/related-list/{scope}?name={name}&page={n}&uri={xuri}{length_param}"

        js = envelopes.collection_page(page_uri(page))
        # get query from the catalogue's sparql cache and substitute in the uri
        # the uri arrives in this deployment's form; the index only knows data URIs
        target = self.uris.inbound(uri)
        spq = self.catalogue.related_list_sparql[scope][name]
        spq = spq.replace("V_TARGET_URI", target).replace("V_LIMIT", str(records))

        res = await run_query(self.qlever, spq)

        # Every entry up to one past this page, in rank order; the page is
        # sliced out of them afterwards.
        entries = []
        names = [x[1:].replace("_", "-") for x in res["variables"]]
        for r in res["results"]:
            if len(entries) > end:
                break
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
                entries.append(entry)

        js["orderedItems"] = entries[offset:end]
        if len(entries) > end:
            js["next"] = envelopes.page_link(page_uri(page + 1))
        return JSONResponse(content=js)
