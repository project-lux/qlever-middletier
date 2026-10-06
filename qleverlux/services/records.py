"""Individual record retrieval.

The record JSON comes from the document cache, not from QLever. What this adds
is the ``_links`` block (for the full record) or a cut-down projection (for the
``name`` and ``results`` profiles), and the rewrite from data URIs to this
deployment's.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

#: AAT terms used to pick a record's primary name.
AAT_ENGLISH = "http://vocab.getty.edu/aat/300388277"
AAT_PRIMARY = "http://vocab.getty.edu/aat/300404670"

#: Fields kept by the "results" profile, on top of the name and type.
RESULTS_FIELDS = [
    "produced_by",
    "created_by",
    "encountered_by",
    "classified_as",
    "member_of",
    "language",
    "referred_to_by",
    "representation",
    "part_of",
    "broader",
    "defined_by",
    "took_place_at",
    "timespan",
    "carried_out_by",
]


def get_primary_name(names):
    """The English primary name if there is one, else the best primary name."""
    candidates = []
    for name in names:
        if name["type"] == "Name":
            langs = [
                x.get("equivalent", [{"id": None}])[0]["id"]
                for x in name.get("language", [])
            ]
            cxns = [
                x.get("equivalent", [{"id": None}])[0]["id"]
                for x in name.get("classified_as", [])
            ]
            if AAT_ENGLISH in langs and AAT_PRIMARY in cxns:
                return name
            elif AAT_PRIMARY in cxns:
                candidates.append(name)
    candidates.sort(key=lambda x: len(x.get("language", [])), reverse=True)
    return candidates[0] if candidates else None


class RecordService:
    def __init__(self, settings, cache, hal_service, uris):
        self.settings = settings
        self.cache = cache
        self.hal = hal_service
        self.uris = uris

    def _curies(self):
        mt = self.settings.mt_uri
        return [
            {"name": "lux", "href": f"{mt}api/rels/{{rel}}", "templated": True},
            {
                "name": "la",
                "href": "https://linked.art/api/1.0/rels/{rel}",
                "templated": True,
            },
        ]

    def project(self, js, profile):
        """The cut-down record returned for the name / results profiles."""
        js2 = {
            "id": js["id"],
            "type": js["type"],
            "identified_by": [get_primary_name(js["identified_by"])],
        }
        if profile == "results":
            for fld in RESULTS_FIELDS:
                if fld in js:
                    js2[fld] = js[fld]
            for nm in js["identified_by"]:
                if nm["type"] == "Identifier":
                    js2["identified_by"].append(nm)
        return js2

    async def get_record(self, scope, identifier, profile=None):
        scope = str(scope.value)
        if profile is not None:
            profile = profile.value
        identifier = str(identifier)

        try:
            res = await self.cache.fetch(identifier)
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

        cache_links = {}
        js = None
        if res:
            js = res[0]
            if len(res) > 1:
                cache_links = res[1]

        if not js:
            return JSONResponse(content={}, status_code=404)

        if not profile:
            links = {
                "curies": self._curies(),
                "self": {
                    "href": f"{self.settings.mt_uri}data/{scope}/{identifier}"
                },
            }
            if cache_links:
                links.update(cache_links)
            else:
                links.update(await self.hal.links(scope, identifier))
            js["_links"] = links
        else:
            js = self.project(js, profile)

        return JSONResponse(content=self.uris.outbound_json(js))
