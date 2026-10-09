"""
Tests for the comparison used by the two builder suites.

Those suites are only worth their runtime if ``same_query`` says "no" when two
queries genuinely differ. It deliberately ignores blank node labels, SERVICE
source layout and derived variable sets, so these cases pin down what it must
still notice.
"""

from __future__ import annotations

import pytest

from .sparql_testing import algebra, same_query

FOAF = "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n"
EX = "PREFIX : <http://example.org/>\n"


DIFFERENT = [
    (
        "predicate",
        FOAF + "SELECT ?a WHERE { ?a foaf:name ?b }",
        FOAF + "SELECT ?a WHERE { ?a foaf:mbox ?b }",
    ),
    (
        "aggregate function",
        "SELECT (COUNT(?o) AS ?c) WHERE { ?s ?p ?o }",
        "SELECT (SUM(?o) AS ?c) WHERE { ?s ?p ?o }",
    ),
    (
        "min vs max",
        "SELECT (MIN(?o) AS ?c) WHERE { ?s ?p ?o }",
        "SELECT (MAX(?o) AS ?c) WHERE { ?s ?p ?o }",
    ),
    (
        "service endpoint",
        FOAF + "SELECT ?s WHERE { SERVICE <http://a/x> { ?s ?p ?o } }",
        FOAF + "SELECT ?s WHERE { SERVICE <http://b/y> { ?s ?p ?o } }",
    ),
    (
        "service body",
        FOAF + "SELECT ?s WHERE { SERVICE <http://a/x> { ?s foaf:name ?o } }",
        FOAF + "SELECT ?s WHERE { SERVICE <http://a/x> { ?s foaf:mbox ?o } }",
    ),
    (
        "path operator",
        EX + "SELECT * WHERE { :a :p+ ?z }",
        EX + "SELECT * WHERE { :a :p* ?z }",
    ),
    (
        "optional vs required",
        FOAF + "SELECT ?a WHERE { ?a foaf:name ?b OPTIONAL { ?a foaf:mbox ?c } }",
        FOAF + "SELECT ?a WHERE { ?a foaf:name ?b . ?a foaf:mbox ?c }",
    ),
    (
        "exists vs not exists",
        FOAF + "SELECT ?a WHERE { ?a ?p ?o FILTER EXISTS { ?a foaf:name ?n } }",
        FOAF + "SELECT ?a WHERE { ?a ?p ?o FILTER NOT EXISTS { ?a foaf:name ?n } }",
    ),
    (
        "distinct blank nodes vs one reused",
        EX + "SELECT ?o WHERE { [] :p1 ?o . [] :p2 ?o }",
        EX + "SELECT ?o WHERE { _:x :p1 ?o . _:x :p2 ?o }",
    ),
    (
        "union vs join",
        FOAF + "SELECT ?a WHERE { { ?a foaf:name ?b } UNION { ?a foaf:mbox ?b } }",
        FOAF + "SELECT ?a WHERE { { ?a foaf:name ?b } { ?a foaf:mbox ?b } }",
    ),
    (
        "limit",
        FOAF + "SELECT ?a WHERE { ?a foaf:name ?b } LIMIT 5",
        FOAF + "SELECT ?a WHERE { ?a foaf:name ?b } LIMIT 10",
    ),
]

SAME = [
    (
        "layout",
        FOAF + "SELECT ?a WHERE { ?a foaf:name ?b }",
        FOAF + "SELECT ?a\nWHERE {\n   ?a foaf:name ?b . \n}",
    ),
    (
        "blank node labels",
        EX + "SELECT ?o WHERE { _:x :p1 ?o . _:x :p2 ?o }",
        EX + "SELECT ?o WHERE { _:y :p1 ?o . _:y :p2 ?o }",
    ),
    (
        "service block layout",
        FOAF + "SELECT ?s WHERE { SERVICE <http://a/x> { ?s ?p ?o } }",
        FOAF + "SELECT ?s WHERE { SERVICE <http://a/x> {\n      ?s ?p ?o . \n   } }",
    ),
    (
        "predicate object list expansion",
        FOAF + "SELECT ?n WHERE { ?x foaf:name ?n ; foaf:mbox ?m }",
        FOAF + "SELECT ?n WHERE { ?x foaf:name ?n . ?x foaf:mbox ?m . }",
    ),
]


@pytest.mark.parametrize(
    ("label", "left", "right"), DIFFERENT, ids=[c[0] for c in DIFFERENT]
)
def test_different_queries_are_not_equivalent(
    label: str, left: str, right: str
) -> None:
    assert not same_query(algebra(left), algebra(right)), (
        f"{label}: two different queries compared equal, so the suites would not "
        f"catch a builder that produced one instead of the other"
    )


@pytest.mark.parametrize(("label", "left", "right"), SAME, ids=[c[0] for c in SAME])
def test_equivalent_queries_are_equivalent(label: str, left: str, right: str) -> None:
    assert same_query(algebra(left), algebra(right)), (
        f"{label}: two equivalent queries compared unequal, so the suites would "
        f"report spurious failures"
    )
