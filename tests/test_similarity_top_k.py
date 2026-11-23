"""A similarity clause at the top of a query keeps the long list; one under a relationship a short one."""

import re

import pytest

pytest.importorskip("luxql")

URI = "https://lux.collections.yale.edu/data/concept/a"


@pytest.fixture(scope="module")
def compile_search():
    from qleverlux.query.catalogue import QueryCatalogue
    from qleverlux.settings import load_settings

    catalogue = QueryCatalogue(load_settings(["--vector-service", "http://127.0.0.1:8081/sparql",
                                              "--vector-top-k", "100", "--vector-nested-top-k", "5"]))

    def top_ks(query, scope):
        parsed = catalogue.json_reader.read(query, scope)
        return re.findall(r"vec:topK (\d+)", catalogue.translator.translate_search(parsed, scope=scope).get_text())

    return top_ks


@pytest.mark.parametrize("scope, query, expected", [
    ("concept", {"similar": "quantum physics"}, ["100"]),
    ("concept", {"AND": [{"similar": "quantum physics"}, {"name": "physics"}]}, ["100"]),
    ("concept", {"similarRecords": {"id": URI}}, ["100"]),
    ("work", {"aboutConcept": {"similar": "quantum physics"}}, ["5"]),
    ("work", {"aboutConcept": {"similarRecords": {"id": URI}}}, ["5"]),
    ("item", {"producedBy": {"memberOf": {"similar": "dutch painters"}}}, ["5"]),
    ("work", {"AND": [{"aboutConcept": {"similar": "quantum physics"}}, {"similar": "textbooks"}]}, ["5", "100"]),
])
def test_top_k_depends_on_depth(compile_search, scope, query, expected):
    assert compile_search(query, scope) == expected
