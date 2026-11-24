"""Related lists page by entry, and link to a next page only when there is one.

An entry is one related record and one way it relates, so a record related in
three ways is three adjacent entries. This endpoint once ignored ``page``,
returned the top 20 records every time and always emitted ``next``; the front
end shows its Next button whenever ``next`` is present, so Next was always on
and every page showed the same records.
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("luxql")

SCOPE, NAME = "agent", "relatedToAgent"
TARGET = "https://lux.collections.yale.edu/data/person/target"
RECORDS = 70
#: Record i relates in 1 + i % 3 ways, so 70 records make 139 entries.
ENTRIES = sum(1 + i % 3 for i in range(RECORDS))


class RankedQLever:
    """Answers the related-list count query like QLever: records ranked by
    total, at most LIMIT of them, with one count column per relation."""

    def __init__(self, sparql):
        self.columns = re.search(r"SELECT (.*?) WHERE", sparql).group(1).split()
        relations = len(self.columns) - 2
        self.rows = []
        for i in range(RECORDS):
            counts = [None] * relations
            for j in range(1 + i % 3):
                counts[j] = 1000 - i * 3 - j
            uri = f"https://lux.collections.yale.edu/data/person/r{i:03d}"
            self.rows.append([uri, sum(c for c in counts if c)] + counts)
        self.queries = []

    async def query(self, sparql, sheddable=True):
        self.queries.append(sparql)
        limit = int(re.search(r"LIMIT (\d+)\s*$", sparql).group(1))
        rows = self.rows[:limit]
        return {"results": rows, "total": len(rows), "time": 1, "variables": self.columns}


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
def related(parts):
    from qleverlux.services.related import RelatedListService

    settings, catalogue, uris = parts
    qlever = RankedQLever(catalogue.related_list_sparql[SCOPE][NAME])
    return RelatedListService(settings, catalogue, qlever, uris), qlever


def fetch(service, page=1, length=None):
    response = asyncio.run(service.related_list(SCOPE, NAME, TARGET, page, length))
    return json.loads(response.body)


def entries(body):
    return [(e["value"], e["name"]) for e in body["orderedItems"]]


def test_every_compiled_query_has_a_stable_order_and_a_limit_to_fill(parts):
    _, catalogue, _ = parts
    for scope, lists in catalogue.related_list_sparql.items():
        for name, sparql in lists.items():
            assert sparql.rstrip().endswith("ORDER BY DESC(?total) ?uri LIMIT V_LIMIT"), (scope, name)


@pytest.mark.parametrize("length", [None, 7, 25, ENTRIES, 1000])
def test_following_next_visits_every_entry_once_in_rank_order(related, length):
    service, qlever = related
    seen, page = [], 1
    while True:
        body = fetch(service, page, length)
        seen += entries(body)
        if "next" not in body:
            break
        assert body["next"]["type"] == "OrderedCollectionPage"
        assert f"page={page + 1}&" in body["next"]["id"]
        page += 1
        assert page <= ENTRIES

    assert len(seen) == ENTRIES
    assert len(set(seen)) == ENTRIES
    # rows are ranked by record, and a record's entries are adjacent
    records = [v for v, _ in seen]
    assert records == sorted(records)


def test_the_default_page_length_comes_from_settings(related, parts):
    service, _ = related
    settings, _, _ = parts
    assert len(fetch(service)["orderedItems"]) == settings.related_page_length == 25


def test_an_exact_fit_has_no_next(related):
    service, _ = related
    body = fetch(service, 1, ENTRIES)
    assert len(body["orderedItems"]) == ENTRIES
    assert "next" not in body


def test_a_page_past_the_end_is_empty_with_no_next(related):
    service, _ = related
    body = fetch(service, 99)
    assert body["orderedItems"] == []
    assert "next" not in body


def test_a_page_length_given_is_carried_into_the_links_and_a_default_is_not(related):
    service, _ = related
    given = fetch(service, 1, 7)
    assert given["id"].endswith("&pageLength=7")
    assert given["next"]["id"].endswith("&pageLength=7")
    default = fetch(service)
    assert "pageLength" not in default["id"]
    assert "pageLength" not in default["next"]["id"]


@pytest.mark.parametrize(
    "page, length, limit",
    [
        (1, None, 60),  # 26 records cover 25 entries and one more
        (2, None, 60),  # 51
        (3, None, 120),  # 76
        (1, 59, 60),  # 60 exactly
        (1, 60, 120),  # 61
        (2, 100, 240),  # 201
    ],
)
def test_records_asked_for_cover_the_page_and_one_more_rounded_to_60(related, page, length, limit):
    service, qlever = related
    fetch(service, page, length)
    assert qlever.queries[-1].rstrip().endswith(f"LIMIT {limit}")
    assert "V_LIMIT" not in qlever.queries[-1]
    assert f"<{TARGET}>" in qlever.queries[-1]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from qleverlux.app import create_app

    calls = []

    class Related:
        async def related_list(self, scope, name, uri, page, pageLength):
            calls.append((page, pageLength))
            return {}

    async def nothing():
        pass

    mt = SimpleNamespace(related=Related(), start=nothing, aclose=nothing)
    with TestClient(create_app(middletier=mt)) as c:
        yield c, calls


def url(extra=""):
    return f"/api/related-list/{SCOPE}?name={NAME}&uri={TARGET}{extra}"


@pytest.mark.parametrize("extra", ["&page=0", "&pageLength=0", "&pageLength=1001"])
def test_the_route_rejects_out_of_range_paging(client, extra):
    c, _ = client
    assert c.get(url(extra)).status_code == 422


def test_the_route_passes_paging_through_and_leaves_an_absent_length_unset(client):
    c, calls = client
    assert c.get(url("&page=3&pageLength=1000")).status_code == 200
    assert c.get(url()).status_code == 200
    assert calls[-2:] == [(3, 1000), (1, None)]
