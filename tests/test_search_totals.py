"""A search page reports the whole result set's size, whatever page it is.

QLever counts the rows after OFFSET, and search pages run at offsets rounded
down to 60, so this once reported 155 on pages 1-3, 95 on 4-6 and 35 on 7-8.
"""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("luxql")

TOTAL = 155


class OffsetCountingQLever:
    """Answers like QLever: rows from OFFSET on, and their count as the total."""

    def __init__(self):
        self.queries = []

    async def query(self, sparql, sheddable=True):
        self.queries.append(sparql)
        found = re.search(r"OFFSET\s+(\d+)", sparql)
        offset = int(found.group(1)) if found else 0
        rows = [[f"https://lux.collections.yale.edu/data/object/{i:04d}", "https://linked.art/ns/terms/HumanMadeObject"]
                for i in range(offset, TOTAL)]
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


def page(service, number, qlever=None):
    qlever = qlever or OffsetCountingQLever()
    response = asyncio.run(service(qlever).search(
        SimpleNamespace(value="item"), json.dumps({"name": "banjo"}), page=number, pageLength=20))
    body = json.loads(response.body)
    return body["partOf"]["totalItems"], body["orderedItems"], qlever


@pytest.mark.parametrize("number, items", [(1, 20), (3, 20), (4, 20), (6, 20), (7, 20), (8, 15)])
def test_every_page_reports_the_whole_total(service, number, items):
    total, ordered, _ = page(service, number)
    assert total == TOTAL
    assert len(ordered) == items
    assert ordered[0]["id"].endswith(f"/{(number - 1) * 20:04d}")


def test_a_page_past_the_end_takes_the_total_from_the_first_window(service):
    total, ordered, qlever = page(service, 20)  # offset 380, rounded to 360: no rows at all
    assert (total, ordered) == (TOTAL, [])
    assert "OFFSET 360" in qlever.queries[0] and "OFFSET" not in qlever.queries[1]
