"""User values never become SPARQL syntax, and a search cannot ask for unbounded rows.

The translator once formatted five kinds of value straight into the query: ``id`` (as a leaf and
under a relationship) and an http ``identifier`` inside ``<...>``, other identifiers inside
``"..."``, ``recordType`` after ``la:``, and dates inside ``"..."^^xsd:dateTime`` (luxql checks
dates with ``re.match``, which anchors only the start). Each let a public search close the term
and append its own SPARQL - including ``SERVICE <any URL>``, which QLever then fetches. The
related-list ``uri`` went into the compiled query the same way. And ``pageLength`` had no upper
bound, so one URL could make QLever return every row and the query cache hold them.
"""

import json
import urllib.parse
from types import SimpleNamespace

import pytest

pytest.importorskip("luxql")

L = "https://lux.collections.yale.edu/data/"
MARK = "?injected"  # lower case: some values are lower-cased on their way in
ESCAPE_BRACKETS = f"x> . {MARK} ?p ?o . SERVICE <http://169.254.169.254/> {{ ?a ?b ?c }} #"
ESCAPE_QUOTES = f'x" . {MARK} ?p ?o . #'

#: (scope, query) pairs that once put MARK into the SPARQL as syntax, and are now refused. (An
#: identifier is escaped instead - identifiers can contain quotes - see below.)
PAYLOADS = {
    "id leaf": ("item", {"id": f"{L}object/{ESCAPE_BRACKETS}"}),
    "id under a relationship": ("item", {"producedBy": {"id": f"{L}person/{ESCAPE_BRACKETS}"}}),
    "recordType": ("agent", {"recordType": f"Person . {MARK} ?p ?o . #"}),
    "date": ("item", {"producedDate": f'1900" . {MARK} ?p ?o . #', "_comp": ">="}),
}


class RecordingQLever:
    def __init__(self):
        self.queries = []

    async def query(self, sparql, sheddable=True):
        self.queries.append(sparql)
        return {"results": [], "total": 0, "time": 1, "variables": ["?uri", "?total"]}


@pytest.fixture(scope="module")
def catalogue():
    from qleverlux.query.catalogue import QueryCatalogue
    from qleverlux.settings import load_settings

    return QueryCatalogue(load_settings([]))


def translate(catalogue, scope, query):
    return catalogue.translator.translate_search(catalogue.json_reader.read(query, scope), scope=scope).get_text()


@pytest.mark.parametrize("label", PAYLOADS)
def test_values_that_would_end_their_term_are_refused(catalogue, label):
    scope, query = PAYLOADS[label]
    with pytest.raises(ValueError):
        translate(catalogue, scope, query)


def test_an_identifier_with_a_quote_is_escaped_not_refused(catalogue):
    sparql = translate(catalogue, "item", {"identifier": 'YCBA "B1977" \\ 14'})
    assert '"ycba \\"b1977\\" \\\\ 14"' in sparql
    # the old payload is searched for as the string it is, inside one literal
    sparql = translate(catalogue, "item", {"identifier": ESCAPE_QUOTES})
    assert '"x\\" . ?injected ?p ?o . #"' in sparql


def test_an_http_identifier_that_is_no_iri_is_matched_as_a_string_only(catalogue):
    sparql = translate(catalogue, "item", {"identifier": "http://example.org/a b"})
    assert '"http://example.org/a b"' in sparql and "la:equivalent" not in sparql
    sparql = translate(catalogue, "item", {"identifier": "http://example.org/ab"})
    assert "la:equivalent <http://example.org/ab>" in sparql  # a real IRI keeps its UNION


@pytest.mark.parametrize("value", ["1900", "1900-01-01", "1900-01-01T00:00:00", "1900-01-01T00:00:00.000Z",
                                   "-000500-01-01"])
def test_dates_in_every_documented_form_still_translate(catalogue, value):
    assert "^^xsd:dateTime" in translate(catalogue, "item", {"producedDate": value, "_comp": ">="})


@pytest.fixture
def api(catalogue):
    from fastapi.testclient import TestClient

    from qleverlux.app import create_app
    from qleverlux.presentation.uris import UriRewriter
    from qleverlux.services.facets import FacetService
    from qleverlux.services.related import RelatedListService
    from qleverlux.services.search import SearchService

    from qleverlux.settings import load_settings

    settings = load_settings([])
    qlever = RecordingQLever()
    uris = UriRewriter(settings.data_uri, settings.mt_uri, settings.record_path)
    common = (settings, catalogue, qlever, uris)

    async def nothing():
        pass

    mt = SimpleNamespace(search=SearchService(*common), facets=FacetService(*common),
                         related=RelatedListService(*common), start=nothing, aclose=nothing)
    with TestClient(create_app(middletier=mt)) as client:
        yield client, qlever


def q(query):
    return urllib.parse.quote(json.dumps(query))


@pytest.mark.parametrize("label", PAYLOADS)
@pytest.mark.parametrize("route", ["search", "search-estimate", "facets"])
def test_every_public_route_refuses_the_payloads_before_asking_qlever(api, label, route):
    client, qlever = api
    scope, query = PAYLOADS[label]
    facet = "&name=agentRecordType" if route == "facets" else ""
    response = client.get(f"/api/{route}/{scope}?q={q(query)}{facet}")
    assert response.status_code == 400, response.text
    assert qlever.queries == []


def test_an_identifier_payload_reaches_qlever_only_as_an_escaped_string(api):
    client, qlever = api
    response = client.get(f"/api/search/item?q={q({'identifier': ESCAPE_QUOTES})}")
    assert response.status_code == 200, response.text
    assert all('"x\\" . ?injected ?p ?o . #"' in sparql for sparql in qlever.queries)


@pytest.mark.parametrize("uri", [
    f"{L}person/x> . {MARK} ?p ?o . #",
    f"{L}person/x\n{MARK}",
    'urn:x" {',
    "not a uri",
])
def test_a_related_list_uri_that_is_no_http_iri_is_refused(api, uri):
    client, qlever = api
    response = client.get(f"/api/related-list/agent?name=relatedToAgent&uri={urllib.parse.quote(uri)}")
    assert response.status_code == 400, response.text
    assert qlever.queries == []


def test_a_related_list_uri_that_is_an_iri_still_runs(api):
    client, qlever = api
    response = client.get(f"/api/related-list/agent?name=relatedToAgent&uri={L}person/a-b-c")
    assert response.status_code == 200, response.text
    assert len(qlever.queries) == 1 and f"<{L}person/a-b-c>" in qlever.queries[0]


@pytest.mark.parametrize("params, status", [
    ("&pageLength=101", 422),
    ("&pageLength=100000000", 422),
    ("&pageLength=-1", 422),
    ("&page=0", 422),
    ("&pageLength=100", 200),
    ("", 200),
])
def test_search_page_size_is_bounded(api, params, status):
    client, _ = api
    response = client.get(f"/api/search/item?q={q({'text': 'x'})}{params}")
    assert response.status_code == status, response.text
