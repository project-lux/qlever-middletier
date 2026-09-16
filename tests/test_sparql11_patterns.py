"""
Build each pattern SPARQL 1.1 added, using the qleverlux builder classes.

Source: https://www.w3.org/2009/sparql/docs/tests/summary.html

The companion module ``test_w3c_examples`` covers the SPARQL 1.0 examples. This
one takes the SPARQL 1.1 test suite and picks one test per construct that
version of the language introduced - property paths, BIND, VALUES, aggregates,
sub-SELECTs, EXISTS/NOT EXISTS, SERVICE, projected expressions and the built-in
function library - so every new pattern is represented once. The suite has 500+
tests, most of them repeated coverage of the same construct or protocol,
entailment and result-format tests that say nothing about query construction.

As in the 1.0 suite, comparison is by SPARQL algebra. Several of these tests are
written as ASK queries, which the builders cannot produce; those carry an
``equivalent_to`` SELECT with the same pattern and a note saying so.

Patterns the builders cannot express are listed in UNSUPPORTED with a probe that
fails once support is added.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

import qleverlux.SPARQLQueryBuilder as builder_module
from qleverlux.SPARQLQueryBuilder import (
    GraphPattern,
    SelectQuery,
    SPARQLSelectQuery,
    SPARQLUpdateQuery,
)
from qleverlux.SPARQLSyntaxTerms import (
    Binding,
    Filter,
    GroupBy,
    Having,
    OrderBy,
    Prefix,
    Triple,
    Values,
)

from .sparql11_reference_queries import QUERIES, TITLES
from .sparql_testing import (
    Example,
    ExampleRegistry,
    assert_valid_sparql,
    check_example,
    example_id,
    parser,
)

example = ExampleRegistry(QUERIES)
EXAMPLES = example.examples

EX = Prefix("ex", "http://www.example.org/schema#")
IN = Prefix("in", "http://www.example.org/instance#")


# --- property paths ----------------------------------------------------------
#
# Paths live in the predicate slot of a Triple, which the builder passes through
# as written. The LUX translator already relies on this for `broader+` and for
# the inverse `^lux:...` predicates.


@example("property-path-pp01")
def build_sequence_path():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    where = GraphPattern()
    where.add_triples(Triple("in:a", "ex:p1/ex:p2/ex:p3", "?x"))
    return query.set_where_pattern(where)


@example("property-path-pp02")
def build_zero_or_more_over_group():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    where = GraphPattern()
    where.add_triples(Triple("in:a", "(ex:p1/ex:p2/ex:p3)*", "?x"))
    return query.set_where_pattern(where)


@example(
    "property-path-pp08",
    equivalent_to="""prefix ex: <http://www.example.org/schema#>
prefix in: <http://www.example.org/instance#>
select * where { in:b ^ex:p in:a }""",
    note="published as an ASK; the builders only produce SELECT, so the same "
    "pattern is compared as a SELECT.",
)
def build_reverse_path():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    where = GraphPattern()
    where.add_triples(Triple("in:b", "^ex:p", "in:a"))
    return query.set_where_pattern(where)


@example("property-path-pp09")
def build_reverse_sequence_path():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    where = GraphPattern()
    where.add_triples(Triple("in:c", "^(ex:p1/ex:p2)", "?x"))
    return query.set_where_pattern(where)


@example("property-path-pp10")
def build_negated_property_set():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    where = GraphPattern()
    where.add_triples(Triple("in:a", "!(ex:p1|ex:p2)", "?x"))
    return query.set_where_pattern(where)


@example(
    "property-path-pp14",
    equivalent_to=QUERIES["property-path-pp14"].replace(
        "ORDER BY ?X ?Y", "ORDER BY ASC(?X) ASC(?Y)"
    ),
    note="each OrderBy is one condition, so two are added; OrderBy(['?X','?Y']) "
    "would render ASC(?X ?Y), which is not legal SPARQL.",
)
def build_zero_or_more_path():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("", "http://example.org/"),
            Prefix("foaf", "http://xmlns.com/foaf/0.1/"),
        ]
    )
    where = GraphPattern()
    where.add_triples(Triple("?X", "foaf:knows*", "?Y"))
    query.set_where_pattern(where)
    query.add_order_by(OrderBy(["?X"]))
    return query.add_order_by(OrderBy(["?Y"]))


@example("property-path-pp21")
def build_one_or_more_path():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example/"))
    where = GraphPattern()
    where.add_triples(Triple(":a", ":p+", "?z"))
    return query.set_where_pattern(where)


@example("property-path-pp28a")
def build_zero_or_one_path():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example/"))
    where = GraphPattern()
    where.add_triples(Triple(":a", "(:p/:p)?", "?t"))
    return query.set_where_pattern(where)


@example("property-path-pp30")
def build_alternative_path_precedence():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables("?t")
    where = GraphPattern()
    where.add_triples(Triple(":a", ":p1|:p2/:p3|:p4", "?t"))
    return query.set_where_pattern(where)


@example("property-path-pp34")
def build_path_inside_named_graph():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables("?t")
    where = GraphPattern()
    graph = GraphPattern(graph_name="<ng-01.ttl>")
    graph.add_triples(Triple("?s", ":p1*", "?t"))
    where.add_nested_graph_pattern(graph)
    return query.set_where_pattern(where)


# --- assignment and inline data ----------------------------------------------


@example("bind-bind01")
def build_bind():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables("?z")
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))
    where.add_binding(Binding("?o+10", "?z"))
    return query.set_where_pattern(where)


@example("bindings-inline1")
def build_inline_values():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("dc", "http://purl.org/dc/elements/1.1/"),
            Prefix("", "http://example.org/book/"),
            Prefix("ns", "http://example.org/ns#"),
        ]
    )
    query.add_variables(["?book", "?title", "?price"])
    where = GraphPattern()
    where.add_value(Values([":book1"], "?book"))
    where.add_triples(
        [Triple("?book", "dc:title", "?title"), Triple("?book", "ns:price", "?price")]
    )
    return query.set_where_pattern(where)


# --- negation and existence --------------------------------------------------


@example(
    "exists-exists01",
    note="there is no positive-EXISTS flag on GraphPattern; the inner pattern is "
    "rendered and wrapped by a Filter, which is how the construct is reached.",
)
def build_positive_exists():
    query = SelectQuery()
    query.add_prefix(Prefix("ex", "http://www.example.org/"))
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))

    inner = GraphPattern()
    inner.add_triples(Triple("?s", "?p", "ex:o"))
    where.add_filter(Filter(f"EXISTS {inner.get_text(1)}"))

    return query.set_where_pattern(where)


@example("negation-subset-by-exclusion-nex-1")
def build_filter_not_exists():
    query = SelectQuery()
    query.add_prefix(
        Prefix("ex", "http://www.w3.org/2009/sparql/docs/tests/data-sparql11/negation#")
    )
    query.add_variables("?animal")
    where = GraphPattern()
    where.add_triples(Triple("?animal", "a", "ex:Animal"))
    absent = GraphPattern(not_exists=True)
    absent.add_triples(Triple("?animal", "a", "ex:Insect"))
    where.add_nested_graph_pattern(absent)
    return query.set_where_pattern(where)


# --- sub-queries -------------------------------------------------------------


@example("subquery-subquery01")
def build_subquery_within_graph_pattern():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    query.add_variables(["?x", "?p"])
    where = GraphPattern()
    graph = GraphPattern(graph_name="?g")
    inner = SelectQuery()
    inner_where = GraphPattern()
    inner_where.add_triples(Triple("?x", "?p", "?y"))
    inner.set_where_pattern(inner_where)
    graph.add_nested_select_query(inner)
    where.add_nested_graph_pattern(graph)
    return query.set_where_pattern(where)


@example("subquery-subquery08")
def build_subquery_with_aggregate():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    query.add_variables(["?x", "?max"])
    where = GraphPattern()

    inner = SelectQuery()
    inner.add_variables("(max(?y) as ?max)")
    inner_where = GraphPattern()
    inner_where.add_triples(Triple("?x", "ex:p", "?y"))
    inner.set_where_pattern(inner_where)

    where.add_nested_select_query(inner)
    where.add_triples(Triple("?x", "ex:p", "?max"))
    return query.set_where_pattern(where)


# --- aggregates --------------------------------------------------------------
#
# Aggregates are projected expressions, so they go through add_variables; GROUP
# BY has its own class.


@example("aggregates-agg01")
def build_count():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org"))
    query.add_variables("(COUNT(?O) AS ?C)")
    where = GraphPattern()
    where.add_triples(Triple("?S", "?P", "?O"))
    return query.set_where_pattern(where)


@example("aggregates-agg-sum-01")
def build_sum():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables("(SUM(?o) AS ?sum)")
    where = GraphPattern()
    where.add_triples(Triple("?s", ":dec", "?o"))
    return query.set_where_pattern(where)


@example("aggregates-agg-avg-01")
def build_avg():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables("(AVG(?o) AS ?avg)")
    where = GraphPattern()
    where.add_triples(Triple("?s", ":dec", "?o"))
    return query.set_where_pattern(where)


@example("aggregates-agg-max-01")
def build_max():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables("(MAX(?o) AS ?max)")
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))
    return query.set_where_pattern(where)


@example("aggregates-agg-avg-02")
def build_group_by_with_having():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables(["?s", "(AVG(?o) AS ?avg)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))
    query.set_where_pattern(where)
    query.add_group_by(GroupBy(["?s"]))
    return query.add_having(Having("AVG(?o) <= 2.0"))


@example("aggregates-agg-min-02")
def build_min_with_group_by():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables(["?s", "(MIN(?o) AS ?min)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))
    query.set_where_pattern(where)
    return query.add_group_by(GroupBy(["?s"]))


@example(
    "aggregates-agg-sample-01",
    equivalent_to="""PREFIX : <http://www.example.org/>
SELECT (SAMPLE(?o) AS ?sample) WHERE { ?s :dec ?o }""",
    note="published as an ASK wrapping a sub-SELECT; the inner SELECT is what "
    "the builders can produce.",
)
def build_sample():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables("(SAMPLE(?o) AS ?sample)")
    where = GraphPattern()
    where.add_triples(Triple("?s", ":dec", "?o"))
    return query.set_where_pattern(where)


@example(
    "aggregates-agg-groupconcat-03",
    equivalent_to="""PREFIX : <http://www.example.org/>
SELECT (GROUP_CONCAT(?o;SEPARATOR=":") AS ?g) WHERE { [] :p1 ?o }""",
    note="published as an ASK wrapping a sub-SELECT; the inner SELECT is what "
    "the builders can produce.",
)
def build_group_concat_with_separator():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables('(GROUP_CONCAT(?o;SEPARATOR=":") AS ?g)')
    where = GraphPattern()
    where.add_triples(Triple("[]", ":p1", "?o"))
    return query.set_where_pattern(where)


# --- projected expressions ---------------------------------------------------


@example("project-expression-projexp01")
def build_projected_expression():
    query = SelectQuery()
    query.add_prefixes([EX, IN])
    query.add_variables(["?x", "?y", "?z", "((?y = ?z) as ?eq)"])
    where = GraphPattern()
    where.add_triples([Triple("?x", "ex:p", "?y"), Triple("?x", "ex:q", "?z")])
    return query.set_where_pattern(where)


# --- federation --------------------------------------------------------------


@example("service-service1")
def build_service():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables(["?s", "?o1", "?o2"])
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p1", "?o1"))
    remote = GraphPattern(service="<http://example.org/sparql>")
    remote.add_triples(Triple("?s", "?p2", "?o2"))
    where.add_nested_graph_pattern(remote)
    return query.set_where_pattern(where)


# --- built-in functions ------------------------------------------------------
#
# Every 1.1 function reaches the query the same way: as expression text inside a
# Filter, a Binding or a projected variable. These cover one per category rather
# than all 61 function tests, since the mechanism does not vary.


@example("functions-abs01")
def build_numeric_function_in_filter():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    where = GraphPattern()
    where.add_triples(Triple("?s", ":num", "?num"))
    where.add_filter(Filter("ABS(?num) >= 2"))
    return query.set_where_pattern(where)


@example("functions-concat01")
def build_string_function_in_projection():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables("(CONCAT(?str1,?str2) AS ?str)")
    where = GraphPattern()
    where.add_triples([Triple(":s6", ":str", "?str1"), Triple(":s7", ":str", "?str2")])
    return query.set_where_pattern(where)


@example("functions-length01")
def build_strlen():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables(["?str", "(STRLEN(?str) AS ?len)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", ":str", "?str"))
    return query.set_where_pattern(where)


@example("functions-substring01")
def build_substr():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables(["?s", "?str", "(SUBSTR(?str,1,1) AS ?substr)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", ":str", "?str"))
    return query.set_where_pattern(where)


@example("functions-replace01")
def build_replace():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("", "http://example.org/"),
            Prefix("xsd", "http://www.w3.org/2001/XMLSchema#"),
        ]
    )
    query.add_variables(["?s", '(REPLACE(?str,"[^a-z0-9]", "-") AS ?new)'])
    where = GraphPattern()
    where.add_triples(Triple("?s", ":str", "?str"))
    return query.set_where_pattern(where)


@example(
    "functions-iri01",
    equivalent_to="""SELECT (URI("http://example.org/uri") AS ?uri)
(IRI("http://example.org/iri") AS ?iri) WHERE {}""",
    note="the published test resolves relative IRIs against a BASE, which the "
    "builders cannot emit, so absolute IRIs are used instead.",
)
def build_iri_constructor():
    query = SelectQuery()
    query.add_variables(
        [
            '(URI("http://example.org/uri") AS ?uri)',
            '(IRI("http://example.org/iri") AS ?iri)',
        ]
    )
    return query.set_where_pattern(GraphPattern())


@example("functions-bnode01")
def build_bnode_constructor():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("", "http://example.org/"),
            Prefix("xsd", "http://www.w3.org/2001/XMLSchema#"),
        ]
    )
    query.add_variables(["?s1", "?s2", "(BNODE(?s1) AS ?b1)", "(BNODE(?s2) AS ?b2)"])
    where = GraphPattern()
    where.add_triples([Triple("?a", ":str", "?s1"), Triple("?b", ":str", "?s2")])
    where.add_filter(Filter("?a = :s1 || ?a = :s3"))
    where.add_filter(Filter("?b = :s1 || ?b = :s3"))
    return query.set_where_pattern(where)


@example("functions-strdt01")
def build_strdt_constructor():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("", "http://example.org/"),
            Prefix("xsd", "http://www.w3.org/2001/XMLSchema#"),
        ]
    )
    query.add_variables(["?s", "(STRDT(?str,xsd:string) AS ?str1)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", ":str", "?str"))
    where.add_filter(Filter('LANGMATCHES(LANG(?str), "en")'))
    return query.set_where_pattern(where)


@example("functions-isnumeric01")
def build_type_test_function():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables(["?s", "?num"])
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?num"))
    where.add_filter(Filter("isNumeric(?num)"))
    return query.set_where_pattern(where)


@example("functions-year")
def build_date_function():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables(["?s", "(YEAR(?date) AS ?x)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", ":date", "?date"))
    return query.set_where_pattern(where)


@example("functions-md5-01")
def build_hash_function():
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://example.org/"))
    query.add_variables("(MD5(?l) AS ?hash)")
    where = GraphPattern()
    where.add_triples(Triple(":s1", ":str", "?l"))
    return query.set_where_pattern(where)


@example("functions-struuid01")
def build_uuid_function_with_bind():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("", "http://example.org/"),
            Prefix("xsd", "http://www.w3.org/2001/XMLSchema#"),
        ]
    )
    query.add_variables("(STRLEN(?uuid) AS ?length)")
    where = GraphPattern()
    where.add_binding(Binding("STRUUID()", "?uuid"))
    where.add_filter(
        Filter(
            "ISLITERAL(?uuid) && REGEX(?uuid, "
            '"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$", "i")'
        )
    )
    return query.set_where_pattern(where)


@example("functions-if01")
def build_if_function():
    query = SelectQuery()
    query.add_prefix(Prefix("xsd", "http://www.w3.org/2001/XMLSchema#"))
    query.add_variables(["?o", '(IF(lang(?o) = "ja", true, false) AS ?integer)'])
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))
    return query.set_where_pattern(where)


@example("functions-coalesce01")
def build_coalesce_function():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("", "http://example.org/"),
            Prefix("xsd", "http://www.w3.org/2001/XMLSchema#"),
        ]
    )
    query.add_variables(
        [
            "(COALESCE(?x, -1) AS ?cx)",
            "(COALESCE(?o/?x, -2) AS ?div)",
            "(COALESCE(?z, -3) AS ?def)",
            "(COALESCE(?z) AS ?err)",
        ]
    )
    where = GraphPattern()
    where.add_triples(Triple("?s", ":p", "?o"))
    optional = GraphPattern(optional=True)
    optional.add_triples(Triple("?s", ":q", "?x"))
    where.add_nested_graph_pattern(optional)
    return query.set_where_pattern(where)


@example(
    "functions-in01",
    equivalent_to="SELECT * WHERE { FILTER(2 IN (1, 2, 3)) }",
    note="published as an ASK.",
)
def build_in_operator():
    query = SelectQuery()
    where = GraphPattern()
    where.add_filter(Filter("2 IN (1, 2, 3)"))
    return query.set_where_pattern(where)


@example(
    "functions-notin01",
    equivalent_to="SELECT * WHERE { FILTER(2 NOT IN ()) }",
    note="published as an ASK.",
)
def build_not_in_operator():
    query = SelectQuery()
    where = GraphPattern()
    where.add_filter(Filter("2 NOT IN ()"))
    return query.set_where_pattern(where)


# --- patterns the builders cannot express ------------------------------------


def _no_init_parameter(owner, name: str) -> bool:
    return name not in inspect.signature(owner.__init__).parameters


def _no_attribute(owner, *names: str) -> bool:
    return not any(hasattr(owner, name) for name in names)


def _no_class(*names: str) -> bool:
    return not any(hasattr(builder_module, name) for name in names)


UNSUPPORTED: list[tuple[str, str, callable]] = [
    (
        "negation-full-minuend",
        "MINUS",
        lambda: _no_init_parameter(GraphPattern, "minus"),
    ),
    (
        "bindings-values1",
        "VALUES after the WHERE clause",
        lambda: _no_attribute(SPARQLSelectQuery, "add_value", "add_values"),
    ),
    (
        "bindings-values4",
        "VALUES over a variable list, with UNDEF",
        lambda: [f.name for f in dataclasses.fields(Values)] == ["values", "name"],
    ),
    (
        "service-service6",
        "SERVICE SILENT",
        lambda: _no_init_parameter(GraphPattern, "silent"),
    ),
    (
        "construct-constructwhere01",
        "CONSTRUCT WHERE",
        lambda: _no_class("SPARQLConstructQuery"),
    ),
    (
        "basic-update-insert-data-spo-named1",
        "INSERT DATA",
        lambda: _no_attribute(SPARQLUpdateQuery, "set_insert_data", "add_insert_data"),
    ),
    (
        "delete-data-dawg-delete-data-01",
        "DELETE DATA",
        lambda: _no_attribute(SPARQLUpdateQuery, "set_delete_data", "add_delete_data"),
    ),
    (
        "delete-where-dawg-delete-where-01",
        "DELETE WHERE",
        lambda: _no_attribute(SPARQLUpdateQuery, "set_delete_where"),
    ),
    (
        "delete-dawg-delete-with-01",
        "WITH",
        lambda: _no_attribute(SPARQLUpdateQuery, "set_with", "with_graph"),
    ),
    (
        "delete-dawg-delete-using-01",
        "USING",
        lambda: _no_attribute(SPARQLUpdateQuery, "add_using", "set_using"),
    ),
    ("update-silent-load-silent", "LOAD", lambda: _no_class("SPARQLLoadQuery")),
    ("clear-dawg-clear-all-01", "CLEAR", lambda: _no_class("SPARQLClearQuery")),
    ("update-silent-create-silent", "CREATE", lambda: _no_class("SPARQLCreateQuery")),
    ("drop-dawg-drop-all-01", "DROP", lambda: _no_class("SPARQLDropQuery")),
    ("copy-copy01", "COPY", lambda: _no_class("SPARQLCopyQuery")),
    ("move-move01", "MOVE", lambda: _no_class("SPARQLMoveQuery")),
    ("add-add01", "ADD", lambda: _no_class("SPARQLAddQuery")),
]


# --- tests -------------------------------------------------------------------


@pytest.mark.parametrize("ex", EXAMPLES, ids=example_id)
def test_builder_reproduces_sparql11_test(ex: Example) -> None:
    """The builders produce valid SPARQL that means what the published test means."""
    check_example(ex)


@pytest.mark.parametrize("key", sorted(QUERIES))
def test_reference_query_is_valid(key: str) -> None:
    """Guard the extracted reference text itself."""
    query = QUERIES[key]
    if key == "service-service6":
        pytest.xfail("rdflib cannot parse this test's nested SERVICE (recursion limit)")
    if key.startswith(
        ("add-", "clear-", "copy-", "move-", "drop-", "delete-", "update-", "basic-")
    ):
        parser.parseUpdate(query)  # update requests use the other entry point
    else:
        assert_valid_sparql(query, f"{key} ({TITLES[key]})")


@pytest.mark.parametrize(
    ("key", "feature", "probe"),
    UNSUPPORTED,
    ids=[f"{key}-{feature}" for key, feature, _ in UNSUPPORTED],
)
def test_unsupported_pattern_is_still_unsupported(key, feature, probe) -> None:
    """
    Record the SPARQL 1.1 patterns the builders cannot express.

    These fail once support is added, which is the signal to build the test
    above rather than leave it listed here.
    """
    assert key in QUERIES
    assert probe(), f"{feature} appears to be supported now; build {key} with it"


def test_having_is_a_query_level_clause() -> None:
    """
    HAVING belongs to the query, not to a graph pattern.

    It has to land after GROUP BY and before ORDER BY, and a pattern must not
    accept one - inside the WHERE braces it would not parse.
    """
    query = SelectQuery()
    query.add_prefix(Prefix("", "http://www.example.org/"))
    query.add_variables(["?s", "(AVG(?o) AS ?avg)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))
    query.set_where_pattern(where)
    query.add_group_by(GroupBy(["?s"]))
    query.add_having(Having("AVG(?o) <= 2.0"))
    query.add_order_by(OrderBy(["?avg"], descending=True))

    built = query.get_text()
    assert_valid_sparql(built, "HAVING query")
    assert built.index("GROUP BY") < built.index("HAVING") < built.index("ORDER BY")
    assert built.index("}") < built.index("HAVING"), (
        "HAVING must follow the WHERE clause"
    )

    assert not hasattr(GraphPattern, "add_having"), (
        "a graph pattern cannot hold a HAVING clause"
    )
    with pytest.raises(TypeError):
        where.add_filter(Having("AVG(?o) <= 2.0"))


def test_multiple_having_conditions_share_one_keyword() -> None:
    """``HAVING (a) (b)`` is the legal form; ``HAVING (a) HAVING (b)`` is not."""
    query = SelectQuery()
    query.add_variables(["?s", "(AVG(?o) AS ?avg)", "(COUNT(?o) AS ?n)"])
    where = GraphPattern()
    where.add_triples(Triple("?s", "?p", "?o"))
    query.set_where_pattern(where)
    query.add_group_by(GroupBy(["?s"]))
    query.add_having(Having("AVG(?o) <= 2.0"))
    query.add_having(Having("COUNT(?o) > 1"))

    built = query.get_text()
    assert_valid_sparql(built, "multi-condition HAVING query")
    assert built.count("HAVING") == 1
    assert "HAVING (AVG(?o) <= 2.0) (COUNT(?o) > 1)" in built


def test_having_term_still_renders_standalone() -> None:
    """The term keeps its own keyword when rendered on its own."""
    having = Having("COUNT(?x) > 2")
    assert having.get_text() == "HAVING (COUNT(?x) > 2)"
    assert having.condition == "(COUNT(?x) > 2)"


def test_every_selected_test_is_accounted_for() -> None:
    """Every reference query is either built or listed as unsupported."""
    built = example.keys
    unsupported = {key for key, _, _ in UNSUPPORTED}

    assert not (built & unsupported), "a test is both built and listed unsupported"
    assert built | unsupported == set(QUERIES), (
        f"unaccounted for: {sorted(set(QUERIES) - built - unsupported)}"
    )
