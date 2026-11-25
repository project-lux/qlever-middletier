"""Bad input is the caller's mistake: a 400 or 422, never a 500, and rejected
before anything is asked of QLever.

Each of these once crashed or answered as if nothing were wrong: a facet ``q``
that is not JSON and an unknown related-list name raised out of the service (a
500), a search-estimate with no ``q`` failed URL-quoting its ``{}`` default (a
500), and an unknown facet name ran a query on a made-up predicate and
returned 200 with no values.
"""

import json
import urllib.parse
from types import SimpleNamespace

import pytest

pytest.importorskip("luxql")

PERSON = "https://lux.collections.yale.edu/data/person/target"


class RecordingQLever:
    """Answers every query with no rows, and remembers it was asked."""

    def __init__(self):
        self.queries = []

    async def query(self, sparql, sheddable=True, fmt=None):
        self.queries.append(sparql)
        return {"results": [], "total": 0, "time": 1, "variables": ["?uri", "?total"]}


@pytest.fixture(scope="module")
def parts():
    from qleverlux.presentation.uris import UriRewriter
    from qleverlux.query.catalogue import QueryCatalogue
    from qleverlux.settings import load_settings

    settings = load_settings([])
    catalogue = QueryCatalogue(settings)
    uris = UriRewriter(settings.data_uri, settings.mt_uri, settings.record_path)
    return settings, catalogue, uris


@pytest.fixture
def api(parts):
    from fastapi.testclient import TestClient

    from qleverlux.app import create_app
    from qleverlux.services.facets import FacetService
    from qleverlux.services.related import RelatedListService
    from qleverlux.services.search import SearchService

    qlever = RecordingQLever()
    settings, catalogue, uris = parts
    common = (settings, catalogue, qlever, uris)

    async def nothing():
        pass

    mt = SimpleNamespace(
        search=SearchService(*common),
        facets=FacetService(*common),
        related=RelatedListService(*common),
        start=nothing,
        aclose=nothing,
    )
    with TestClient(create_app(middletier=mt)) as client:
        yield client, qlever


def q(query):
    return urllib.parse.quote(json.dumps(query))


def rejected(response, status=400):
    assert response.status_code == status, response.text
    return response.json()


# -- facets ------------------------------------------------------------------


@pytest.mark.parametrize("query", ["notjson", "", "%7B%22text%22%3A", "%5B1%2C2%5D"])
def test_a_facet_query_that_is_not_a_json_object_is_a_400(api, query):
    client, qlever = api
    rejected(client.get(f"/api/facets/item?name=itemTypeId&q={query}"))
    assert qlever.queries == []


def test_an_unknown_facet_is_a_400_naming_it(api):
    client, qlever = api
    body = rejected(client.get(f"/api/facets/item?name=bogusFacet&q={q({'text': 'x'})}"))
    assert body == {"error": "'bogusFacet' is not a configured facet."}
    assert qlever.queries == []


def test_an_unknown_facet_is_reported_before_a_bad_query(api):
    client, _ = api
    body = rejected(client.get("/api/facets/item?name=bogusFacet&q=notjson"))
    assert "bogusFacet" in body["error"]


@pytest.mark.parametrize(
    "name",
    [
        "itemTypeId",  # from config/facets.json
        "itemRecordType",  # special-cased onto `a`
        "responsibleUnits",  # fixed property paths
        "responsibleCollections",
    ],
)
def test_configured_and_built_in_facets_still_run(api, name):
    client, qlever = api
    response = client.get(f"/api/facets/item?name={name}&q={q({'text': 'x'})}")
    assert response.status_code == 200, response.text
    assert len(qlever.queries) == 1


# -- related lists -----------------------------------------------------------


@pytest.mark.parametrize(
    "scope, name",
    [
        ("agent", "relatedToNothing"),  # a scope with related lists, not this one
        ("item", "relatedToAgent"),  # a scope with none at all
    ],
)
def test_an_unknown_related_list_is_a_400_naming_it_and_its_scope(api, scope, name):
    client, qlever = api
    body = rejected(client.get(f"/api/related-list/{scope}?name={name}&uri={PERSON}"))
    assert body == {"error": f"No configuration for the '{name}' related list in the '{scope}' scope."}
    assert qlever.queries == []


def test_a_known_related_list_still_runs(api):
    client, qlever = api
    response = client.get(f"/api/related-list/agent?name=relatedToAgent&uri={PERSON}")
    assert response.status_code == 200, response.text
    assert len(qlever.queries) == 1


# -- search estimate ---------------------------------------------------------


def test_a_search_estimate_with_no_query_is_a_422_like_search(api):
    client, qlever = api
    for path in ("/api/search-estimate/item", "/api/search/item"):
        body = rejected(client.get(path), 422)
        assert body["detail"][0]["loc"] == ["query", "q"]
    assert qlever.queries == []


def test_a_search_estimate_with_a_query_still_runs(api):
    client, qlever = api
    response = client.get(f"/api/search-estimate/item?q={q({'text': 'x'})}")
    assert response.status_code == 200, response.text
    assert response.json()["totalItems"] == 0
    assert len(qlever.queries) == 1
