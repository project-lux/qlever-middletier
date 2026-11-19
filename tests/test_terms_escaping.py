"""
Escaping helpers for user values, and SERVICE rendering with a full IRI.

These are what keep a user's ``similar`` value (record URI or free text) from
ending up as query text.
"""

from __future__ import annotations

import pytest

from qleverlux.sparql import GraphPattern, SelectQuery, Triple, iri, string_literal

from .sparql_testing import algebra


@pytest.mark.parametrize(
    "raw, rendered",
    [
        ("roman coins", '"roman coins"'),
        ('say "hi"', r'"say \"hi\""'),
        ("back\\slash", r'"back\\slash"'),
        # The backslash must be escaped before the quote, or \" becomes \\".
        ('\\"', r'"\\\""'),
        ("two\nlines\r\tx", r'"two\nlines\r\tx"'),
        ("", '""'),
    ],
)
def test_string_literal(raw, rendered):
    assert string_literal(raw) == rendered


@pytest.mark.parametrize(
    "raw",
    ['" } ?s ?p ?o { "', "a\\", 'x" . ?s ?p ?o . ?x <y> "', "\n", 'é "ü"'],
)
def test_string_literal_parses_back(raw):
    """The rendered literal must match exactly the original string, and nothing else."""
    rdflib = pytest.importorskip("rdflib")
    g = rdflib.Graph()
    p = rdflib.URIRef("http://ex/p")
    g.add((rdflib.URIRef("http://ex/hit"), p, rdflib.Literal(raw)))
    g.add((rdflib.URIRef("http://ex/miss"), p, rdflib.Literal(raw + "x")))
    query = f"SELECT ?s WHERE {{ ?s <http://ex/p> {string_literal(raw)} }}"
    assert [str(r.s) for r in g.query(query)] == ["http://ex/hit"]


def test_iri_accepts_record_uri():
    uri = "https://lux.collections.yale.edu/data/set/0025ab12-dc14-5a95-bdc7-e4c7e1db2349"
    assert iri(uri) == f"<{uri}>"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "https://x/a> . ?s ?p ?o <b",
        "https://x/a b",
        'https://x/"a',
        "https://x/{a}",
        "https://x/a|b",
        "https://x/a^b",
        "https://x/a`b",
        "https://x/a\\b",
        "https://x/a\nb",
    ],
)
def test_iri_rejects_breakout(bad):
    with pytest.raises(ValueError):
        iri(bad)


def test_service_with_full_iri_is_not_given_a_colon():
    svc = GraphPattern(service="<http://127.0.0.1:8081/sparql>")
    svc.add_triples([Triple("?uri", "<http://ex/similarTo>", '"x"')])
    where = GraphPattern()
    where.add_nested_graph_pattern(svc)
    q = SelectQuery()
    q.add_variables(["?uri"])
    q.set_where_pattern(where)
    text = q.get_text()
    assert "SERVICE <http://127.0.0.1:8081/sparql> {" in text
    algebra(text)  # parses
