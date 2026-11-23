"""A search page reports the whole result set's size, whatever page it is, and
fetches one window of rows rather than everything from its offset on.

QLever counts the rows after OFFSET, and search pages run at offsets rounded
down to 60, so this once reported 155 on pages 1-3, 95 on 4-6 and 35 on 7-8.
With a LIMIT it counts only the window, so the total is a separate COUNT(*).
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("luxql")

TOTAL = 155


class WindowedQLever:
    """Answers like QLever: a COUNT(*) query gets one row; a search gets the rows
    from OFFSET for at most LIMIT, and their number as its total."""

    def __init__(self):
        self.queries = []

    async def query(self, sparql, sheddable=True):
        self.queries.append(sparql)
        if "(COUNT(*) AS ?count)" in sparql:
            return {"results": [[TOTAL]], "total": 1, "time": 1, "variables": ["count"]}
        offset = re.search(r"OFFSET\s+(\d+)", sparql)
        limit = re.search(r"LIMIT\s+(\d+)", sparql)
        start = int(offset.group(1)) if offset else 0
        stop = min(TOTAL, start + int(limit.group(1))) if limit else TOTAL
        rows = [[f"https://lux.collections.yale.edu/data/object/{i:04d}", "https://linked.art/ns/terms/HumanMadeObject"]
                for i in range(start, stop)]
        return {"results": rows, "total": len(rows), "time": 1, "variables": ["uri", "type"]}


@pytest.fixture(scope="module")
def service():
    from qleverlux.presentation.uris import UriRewriter
    from qleverlux.query.catalogue import QueryCatalogue
    from qleverlux.services.search import SearchService
    from qleverlux.settings import load_settings

    settings = load_settings([])
    catalogue = QueryCatalogue(settings)
    uris = UriRewriter(settings.data_uri, settings.mt_uri, settings.record_path)
    return lambda qlever: SearchService(settings, catalogue, qlever, uris)


def page(service, number, qlever=None, length=20):
    qlever = qlever or WindowedQLever()
    response = asyncio.run(service(qlever).search(
        SimpleNamespace(value="item"), json.dumps({"name": "banjo"}), page=number, pageLength=length))
    body = json.loads(response.body)
    return body["partOf"]["totalItems"], body["orderedItems"], qlever


@pytest.mark.parametrize("number, items", [(1, 20), (3, 20), (4, 20), (6, 20), (7, 20), (8, 15)])
def test_every_page_reports_the_whole_total(service, number, items):
    total, ordered, _ = page(service, number)
    assert total == TOTAL
    assert len(ordered) == items
    assert ordered[0]["id"].endswith(f"/{(number - 1) * 20:04d}")


def test_a_page_past_the_end_still_has_the_total(service):
    total, ordered, _ = page(service, 20)  # offset 380, rounded to 360: no rows at all
    assert (total, ordered) == (TOTAL, [])


def test_each_page_asks_for_one_window_and_the_count(service):
    _, _, qlever = page(service, 5)  # offset 80 -> the window at 60
    window, count = sorted(qlever.queries, key=lambda q: "COUNT(*)" in q)
    assert "LIMIT 60" in window and "OFFSET 60" in window
    assert "(COUNT(*) AS ?count)" in count and "LIMIT" not in count and "OFFSET" not in count


@pytest.mark.parametrize("length, number, first, items", [(25, 3, 50, 25), (100, 2, 100, 55), (7, 9, 56, 7)])
def test_a_page_crossing_the_window_end_gets_a_wider_window(service, length, number, first, items):
    total, ordered, qlever = page(service, number, length=length)
    assert total == TOTAL and len(ordered) == items
    assert ordered[0]["id"].endswith(f"/{first:04d}")


def test_the_window_rounds_to_the_granularity():
    from qleverlux.services.search import SearchService

    assert SearchService.window(1, 20) == (0, 60)
    assert SearchService.window(4, 20) == (60, 60)
    assert SearchService.window(3, 25) == (0, 120)   # rows 50-74 cross 60
    assert SearchService.window(2, 100) == (60, 180)  # rows 100-199


def test_estimate_runs_only_the_count(service):
    qlever = WindowedQLever()
    response = asyncio.run(service(qlever).estimate("item", json.dumps({"name": "banjo"})))
    assert json.loads(response.body)["totalItems"] == TOTAL
    assert len(qlever.queries) == 1 and "(COUNT(*) AS ?count)" in qlever.queries[0]
