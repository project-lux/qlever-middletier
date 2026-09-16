"""
Rebuild the W3C SPARQL example queries using the qleverlux builder classes.

Source: https://www.w3.org/2001/sw/DataAccess/rq23/examples.html

Each example is constructed with the public builder API only - SelectQuery,
GraphPattern, Triple, Prefix, Filter and friends - and then checked against the
query as published. Comparison is by SPARQL algebra (via rdflib), not by string
equality, because the builder's layout is its own: it writes one triple per
line, expands ``;`` predicate-object lists, and always qualifies ORDER BY
conditions. Algebra comparison ignores all of that while still catching a
genuinely different query.

Two things are asserted for every example: that the builder's output is valid
SPARQL, and that it means the same as the W3C text.

Where the builders cannot express an example at all, it is listed in
UNSUPPORTED with the reason instead of being quietly dropped;
``test_every_w3c_example_is_accounted_for`` fails if an example appears in
neither place.

Requires pytest and rdflib.
"""

from __future__ import annotations

import pytest

from qleverlux.SPARQLQueryBuilder import (
    BNode,
    GraphPattern,
    SelectQuery,
    SPARQLQuery,
    SPARQLSelectQuery,
    SPARQLUpdateQuery,
)
from qleverlux.SPARQLSyntaxTerms import (
    Binding,
    Bound,
    Filter,
    GroupBy,
    IfClause,
    OrderBy,
    Prefix,
    Triple,
    Values,
)

from .sparql_testing import (
    Example,
    ExampleRegistry,
    algebra,
    assert_valid_sparql,
    check_example,
    example_id,
    parser,
)
from .w3c_reference_queries import QUERIES

example = ExampleRegistry(QUERIES)
EXAMPLES = example.examples


DC = Prefix("dc", "http://purl.org/dc/elements/1.1/")
FOAF = Prefix("foaf", "http://xmlns.com/foaf/0.1/")
XSD = Prefix("xsd", "http://www.w3.org/2001/XMLSchema#")
VCARD = Prefix("vcard", "http://www.w3.org/2001/vcard-rdf/3.0#")


# --- 2. Making Simple Queries ------------------------------------------------


@example("Q2")
def build_triple_pattern_with_full_iris():
    query = SelectQuery()
    query.add_variables("?title")
    where = GraphPattern()
    where.add_triples(
        Triple(
            "<http://example.org/book/book1>",
            "<http://purl.org/dc/elements/1.1/title>",
            "?title",
        )
    )
    return query.set_where_pattern(where)


@example("Q3")
def build_triple_pattern_with_prefix():
    query = SelectQuery()
    query.add_prefix(DC)
    query.add_variables("?title")
    where = GraphPattern()
    where.add_triples(Triple("<http://example.org/book/book1>", "dc:title", "?title"))
    return query.set_where_pattern(where)


@example("Q4")
def build_default_prefix_and_dollar_variables():
    query = SelectQuery()
    query.add_prefixes([DC, Prefix("", "http://example.org/book/")])
    query.add_variables("$title")
    where = GraphPattern()
    where.add_triples(Triple(":book1", "dc:title", "$title"))
    return query.set_where_pattern(where)


@example("Q9")
def build_two_projected_variables():
    query = SelectQuery()
    query.add_prefix(DC)
    query.add_variables(["?book", "?title"])
    where = GraphPattern()
    where.add_triples(Triple("?book", "dc:title", "?title"))
    return query.set_where_pattern(where)


@example("Q11")
def build_literal_object_constraint():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables("?mbox")
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?x", "foaf:name", '"Johnny Lee Outlaw"'),
            Triple("?x", "foaf:mbox", "?mbox"),
        ]
    )
    return query.set_where_pattern(where)


@example("Q13")
def build_shared_subject_variable():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "foaf:mbox", "?mbox")]
    )
    return query.set_where_pattern(where)


@example("Q15")
def build_blank_node_result_projection():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?x", "?name"])
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    return query.set_where_pattern(where)


@example("Q17")
def build_simple_book_titles():
    query = SelectQuery()
    query.add_prefix(DC)
    query.add_variables(["?book", "?title"])
    where = GraphPattern()
    where.add_triples(Triple("?book", "dc:title", "?title"))
    return query.set_where_pattern(where)


@example("Q18")
def build_reification_pattern():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            DC,
            Prefix("", "http://example/ns#"),
        ]
    )
    query.add_variables(["?book", "?title"])
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?t", "rdf:subject", "?book"),
            Triple("?t", "rdf:predicate", "dc:title"),
            Triple("?t", "rdf:object", "?title"),
            Triple("?t", ":saidBy", '"Bob"'),
        ]
    )
    return query.set_where_pattern(where)


@example("Q20")
def build_integer_literal_object():
    query = SelectQuery()
    query.add_variables("?v")
    where = GraphPattern()
    where.add_triples(Triple("?v", "?p", "42"))
    return query.set_where_pattern(where)


@example("Q21")
def build_typed_literal_object():
    query = SelectQuery()
    query.add_variables("?v")
    where = GraphPattern()
    where.add_triples(
        Triple("?v", "?p", '"abc"^^<http://example.org/datatype#specialDatatype>')
    )
    return query.set_where_pattern(where)


@example("Q22")
def build_plain_literal_object():
    query = SelectQuery()
    query.add_variables("?x")
    where = GraphPattern()
    where.add_triples(Triple("?x", "?p", '"cat"'))
    return query.set_where_pattern(where)


@example("Q23")
def build_language_tagged_literal_object():
    query = SelectQuery()
    query.add_variables("?x")
    where = GraphPattern()
    where.add_triples(Triple("?x", "?p", '"cat"@en'))
    return query.set_where_pattern(where)


@example(
    "Q25",
    note="The builder always emits FILTERs after the triples of their group; "
    "position within a group has no effect on meaning.",
)
def build_filter_between_triples():
    query = SelectQuery()
    query.add_prefixes([DC, Prefix("ns", "http://example.org/ns#")])
    query.add_variables(["?title", "?price"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "ns:price", "?price"), Triple("?x", "dc:title", "?title")]
    )
    where.add_filter(Filter("?price < 30"))
    return query.set_where_pattern(where)


# --- 5. Graph Patterns -------------------------------------------------------


@example("Q26")
def build_basic_graph_pattern():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "foaf:mbox", "?mbox")]
    )
    return query.set_where_pattern(where)


@example("Q27")
def build_two_nested_groups():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    first = GraphPattern()
    first.add_triples(Triple("?x", "foaf:name", "?name"))
    second = GraphPattern()
    second.add_triples(Triple("?x", "foaf:mbox", "?mbox"))
    where.add_nested_graph_pattern(first)
    where.add_nested_graph_pattern(second)
    return query.set_where_pattern(where)


@example("Q29")
def build_optional_pattern():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    optional = GraphPattern(optional=True)
    optional.add_triples(Triple("?x", "foaf:mbox", "?mbox"))
    where.add_nested_graph_pattern(optional)
    return query.set_where_pattern(where)


@example("Q31")
def build_optional_with_filter():
    query = SelectQuery()
    query.add_prefixes([DC, Prefix("ns", "http://example.org/ns#")])
    query.add_variables(["?title", "?price"])
    where = GraphPattern()
    where.add_triples(Triple("?x", "dc:title", "?title"))
    optional = GraphPattern(optional=True)
    optional.add_triples(Triple("?x", "ns:price", "?price"))
    optional.add_filter(Filter("?price < 30"))
    where.add_nested_graph_pattern(optional)
    return query.set_where_pattern(where)


@example("Q33")
def build_multiple_optionals():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox", "?hpage"])
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    for predicate, variable in (("foaf:mbox", "?mbox"), ("foaf:homepage", "?hpage")):
        optional = GraphPattern(optional=True)
        optional.add_triples(Triple("?x", predicate, variable))
        where.add_nested_graph_pattern(optional)
    return query.set_where_pattern(where)


@example("Q35")
def build_nested_optionals():
    query = SelectQuery()
    query.add_prefixes([FOAF, VCARD])
    query.add_variables(["?foafName", "?mbox", "?gname", "?fname"])
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?foafName"))

    mbox = GraphPattern(optional=True)
    mbox.add_triples(Triple("?x", "foaf:mbox", "?mbox"))
    where.add_nested_graph_pattern(mbox)

    vcard = GraphPattern(optional=True)
    vcard.add_triples(
        [Triple("?x", "vcard:N", "?vc"), Triple("?vc", "vcard:Given", "?gname")]
    )
    family = GraphPattern(optional=True)
    family.add_triples(Triple("?vc", "vcard:Family", "?fname"))
    vcard.add_nested_graph_pattern(family)
    where.add_nested_graph_pattern(vcard)

    return query.set_where_pattern(where)


@example("Q37")
def build_union_of_alternatives():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("dc10", "http://purl.org/dc/elements/1.0/"),
            Prefix("dc11", "http://purl.org/dc/elements/1.1/"),
        ]
    )
    query.add_variables("?title")
    where = GraphPattern()
    first = GraphPattern()
    first.add_triples(Triple("?book", "dc10:title", "?title"))
    second = GraphPattern(union=True)
    second.add_triples(Triple("?book", "dc11:title", "?title"))
    where.add_nested_graph_pattern(first)
    where.add_nested_graph_pattern(second)
    return query.set_where_pattern(where)


@example("Q38")
def build_union_with_distinct_variables():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("dc10", "http://purl.org/dc/elements/1.0/"),
            Prefix("dc11", "http://purl.org/dc/elements/1.1/"),
        ]
    )
    query.add_variables(["?x", "?y"])
    where = GraphPattern()
    first = GraphPattern()
    first.add_triples(Triple("?book", "dc10:title", "?x"))
    second = GraphPattern(union=True)
    second.add_triples(Triple("?book", "dc11:title", "?y"))
    where.add_nested_graph_pattern(first)
    where.add_nested_graph_pattern(second)
    return query.set_where_pattern(where)


@example("Q39")
def build_union_of_multi_triple_groups():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("dc10", "http://purl.org/dc/elements/1.1/"),
            Prefix("dc11", "http://purl.org/dc/elements/1.0/"),
        ]
    )
    query.add_variables(["?title", "?author"])
    where = GraphPattern()
    first = GraphPattern()
    first.add_triples(
        [
            Triple("?book", "dc10:title", "?title"),
            Triple("?book", "dc10:creator", "?author"),
        ]
    )
    second = GraphPattern(union=True)
    second.add_triples(
        [
            Triple("?book", "dc11:title", "?title"),
            Triple("?book", "dc11:creator", "?author"),
        ]
    )
    where.add_nested_graph_pattern(first)
    where.add_nested_graph_pattern(second)
    return query.set_where_pattern(where)


# --- 8. RDF Dataset ----------------------------------------------------------


@example("Q48")
def build_graph_with_variable_name():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?src", "?bobNick"])
    where = GraphPattern()
    graph = GraphPattern(graph_name="?src")
    graph.add_triples(
        [
            Triple("?x", "foaf:mbox", "<mailto:bob@work.example>"),
            Triple("?x", "foaf:nick", "?bobNick"),
        ]
    )
    where.add_nested_graph_pattern(graph)
    return query.set_where_pattern(where)


@example("Q49")
def build_graph_with_iri_name():
    query = SelectQuery()
    query.add_prefixes([FOAF, Prefix("data", "http://example.org/foaf/")])
    query.add_variables("?nick")
    where = GraphPattern()
    graph = GraphPattern(graph_name="data:bobFoaf")
    graph.add_triples(
        [
            Triple("?x", "foaf:mbox", "<mailto:bob@work.example>"),
            Triple("?x", "foaf:nick", "?nick"),
        ]
    )
    where.add_nested_graph_pattern(graph)
    return query.set_where_pattern(where)


@example("Q50")
def build_two_graph_blocks():
    query = SelectQuery()
    query.add_prefixes(
        [
            Prefix("data", "http://example.org/foaf/"),
            FOAF,
            Prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
        ]
    )
    query.add_variables(["?mbox", "?nick", "?ppd"])
    where = GraphPattern()

    alice = GraphPattern(graph_name="data:aliceFoaf")
    alice.add_triples(
        [
            Triple("?alice", "foaf:mbox", "<mailto:alice@work.example>"),
            Triple("?alice", "foaf:knows", "?whom"),
            Triple("?whom", "foaf:mbox", "?mbox"),
            Triple("?whom", "rdfs:seeAlso", "?ppd"),
            Triple("?ppd", "a", "foaf:PersonalProfileDocument"),
        ]
    )
    where.add_nested_graph_pattern(alice)

    ppd = GraphPattern(graph_name="?ppd")
    ppd.add_triples(
        [Triple("?w", "foaf:mbox", "?mbox"), Triple("?w", "foaf:nick", "?nick")]
    )
    where.add_nested_graph_pattern(ppd)

    return query.set_where_pattern(where)


@example("Q54")
def build_graph_metadata_and_contents():
    query = SelectQuery()
    query.add_prefixes([FOAF, DC])
    query.add_variables(["?name", "?mbox", "?date"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?g", "dc:publisher", "?name"), Triple("?g", "dc:date", "?date")]
    )
    graph = GraphPattern(graph_name="?g")
    graph.add_triples(
        [
            Triple("?person", "foaf:name", "?name"),
            Triple("?person", "foaf:mbox", "?mbox"),
        ]
    )
    where.add_nested_graph_pattern(graph)
    return query.set_where_pattern(where)


# --- 9. Solution Sequences and Modifiers -------------------------------------


@example("Q65")
def build_unmodified_solution_sequence():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    return query.set_where_pattern(where)


@example("Q67")
def build_distinct_projection():
    query = SelectQuery(distinct=True)
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    return query.set_where_pattern(where)


@example(
    "Q68",
    equivalent_to=QUERIES["Q68"].replace("ORDER BY ?name", "ORDER BY ASC(?name)"),
    note="OrderBy always qualifies the condition; ORDER BY ?name and "
    "ORDER BY ASC(?name) are the same ordering, but rdflib's algebra "
    "distinguishes them, so the reference is normalized.",
)
def build_ascending_order():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    query.set_where_pattern(where)
    return query.add_order_by(OrderBy(["?name"]))


@example("Q69")
def build_descending_order():
    query = SelectQuery()
    query.add_prefixes([Prefix("", "http://example.org/ns#"), FOAF, XSD])
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", ":empId", "?emp")]
    )
    query.set_where_pattern(where)
    return query.add_order_by(OrderBy(["?emp"], descending=True))


@example(
    "Q70",
    equivalent_to=QUERIES["Q70"]
    .replace(
        "PREFIX foaf:",
        "PREFIX     :    <http://example.org/ns#>\nPREFIX foaf:",
    )
    .replace("ORDER BY ?name DESC(?emp)", "ORDER BY ASC(?name) DESC(?emp)"),
    note="As published this example uses :empId without declaring the default "
    "prefix, so it is not valid SPARQL; the reference adds the declaration "
    "from the preceding example and qualifies the first ORDER BY condition.",
)
def build_multiple_order_conditions():
    query = SelectQuery()
    query.add_prefixes([Prefix("", "http://example.org/ns#"), FOAF])
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", ":empId", "?emp")]
    )
    query.set_where_pattern(where)
    query.add_order_by(OrderBy(["?name"]))
    return query.add_order_by(OrderBy(["?emp"], descending=True))


@example("Q71")
def build_limit():
    query = SelectQuery(limit=20)
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    return query.set_where_pattern(where)


@example(
    "Q72",
    equivalent_to=QUERIES["Q72"].replace("ORDER BY ?name", "ORDER BY ASC(?name)"),
    note="ORDER BY normalized to the qualified form, as in Q68.",
)
def build_order_limit_and_offset():
    query = SelectQuery(limit=5, offset=10)
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    query.set_where_pattern(where)
    return query.add_order_by(OrderBy(["?name"]))


@example("Q74")
def build_join_across_people():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?nameX", "?nameY", "?nickY"])
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?x", "foaf:knows", "?y"),
            Triple("?x", "foaf:name", "?nameX"),
            Triple("?y", "foaf:name", "?nameY"),
        ]
    )
    optional = GraphPattern(optional=True)
    optional.add_triples(Triple("?y", "foaf:nick", "?nickY"))
    where.add_nested_graph_pattern(optional)
    return query.set_where_pattern(where)


# --- 11. Testing Values ------------------------------------------------------


@example("Q100")
def build_datetime_comparison_filter():
    query = SelectQuery()
    query.add_prefixes(
        [Prefix("a", "http://www.w3.org/2000/10/annotation-ns#"), DC, XSD]
    )
    query.add_variables("?annot")
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?annot", "a:annotates", "<http://www.w3.org/TR/rdf-sparql-query/>"),
            Triple("?annot", "dc:date", "?date"),
        ]
    )
    where.add_filter(Filter('?date > "2005-01-01T00:00:00Z"^^xsd:dateTime'))
    return query.set_where_pattern(where)


@example("Q102")
def build_bound_filter():
    query = SelectQuery()
    query.add_prefixes([FOAF, DC, XSD])
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:givenName", "?givenName"))
    optional = GraphPattern(optional=True)
    optional.add_triples(Triple("?x", "dc:date", "?date"))
    where.add_nested_graph_pattern(optional)
    where.add_filter(Filter(str(Bound("?date"))))
    return query.set_where_pattern(where)


@example("Q103")
def build_negated_bound_filter():
    query = SelectQuery()
    query.add_prefixes([FOAF, DC])
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:givenName", "?name"))
    optional = GraphPattern(optional=True)
    optional.add_triples(Triple("?x", "dc:date", "?date"))
    where.add_nested_graph_pattern(optional)
    where.add_filter(Filter(f"!{Bound('?date')}"))
    return query.set_where_pattern(where)


@example("Q105")
def build_is_iri_filter():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "foaf:mbox", "?mbox")]
    )
    where.add_filter(Filter("isIRI(?mbox)"))
    return query.set_where_pattern(where)


@example("Q107")
def build_is_blank_filter_with_optional():
    query = SelectQuery()
    query.add_prefixes(
        [Prefix("a", "http://www.w3.org/2000/10/annotation-ns#"), DC, FOAF]
    )
    query.add_variables(["?given", "?family"])
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?annot", "a:annotates", "<http://www.w3.org/TR/rdf-sparql-query/>"),
            Triple("?annot", "dc:creator", "?c"),
        ]
    )
    optional = GraphPattern(optional=True)
    optional.add_triples(
        [Triple("?c", "foaf:given", "?given"), Triple("?c", "foaf:family", "?family")]
    )
    where.add_nested_graph_pattern(optional)
    where.add_filter(Filter("isBlank(?c)"))
    return query.set_where_pattern(where)


@example("Q109")
def build_is_literal_filter():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "foaf:mbox", "?mbox")]
    )
    where.add_filter(Filter("isLiteral(?mbox)"))
    return query.set_where_pattern(where)


@example("Q111")
def build_regex_on_str_filter():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "foaf:mbox", "?mbox")]
    )
    where.add_filter(Filter('regex(str(?mbox), "@work.example")'))
    return query.set_where_pattern(where)


@example("Q113")
def build_lang_filter():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "?mbox"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "foaf:mbox", "?mbox")]
    )
    where.add_filter(Filter('lang(?name) = "ES"'))
    return query.set_where_pattern(where)


@example("Q115")
def build_datatype_filter():
    query = SelectQuery()
    query.add_prefixes([FOAF, XSD, Prefix("eg", "http://biometrics.example/ns#")])
    query.add_variables(["?name", "?shoeSize"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "eg:shoeSize", "?shoeSize")]
    )
    where.add_filter(Filter("datatype(?shoeSize) = xsd:integer"))
    return query.set_where_pattern(where)


@example("Q117")
def build_conjunction_filter():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name1", "?name2"])
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?x", "foaf:name", "?name1"),
            Triple("?x", "foaf:mbox", "?mbox1"),
            Triple("?y", "foaf:name", "?name2"),
            Triple("?y", "foaf:mbox", "?mbox2"),
        ]
    )
    where.add_filter(Filter("?mbox1 = ?mbox2 && ?name1 != ?name2"))
    return query.set_where_pattern(where)


@example("Q119")
def build_disjunction_filter():
    query = SelectQuery()
    query.add_prefixes(
        [Prefix("a", "http://www.w3.org/2000/10/annotation-ns#"), DC, XSD]
    )
    query.add_variables("?annotates")
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?annot", "a:annotates", "?annotates"),
            Triple("?annot", "dc:date", "?date"),
        ]
    )
    where.add_filter(
        Filter(
            '?date = xsd:dateTime("2004-01-01T00:00:00Z") || '
            '?date = xsd:dateTime("2005-01-01T00:00:00Z")'
        )
    )
    return query.set_where_pattern(where)


@example("Q121")
def build_lang_matches_filter():
    query = SelectQuery()
    query.add_prefix(DC)
    query.add_variables("?title")
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?x", "dc:title", '"That Seventies Show"@en'),
            Triple("?x", "dc:title", "?title"),
        ]
    )
    where.add_filter(Filter('langMatches( lang(?title), "FR" )'))
    return query.set_where_pattern(where)


@example("Q122")
def build_lang_matches_wildcard_filter():
    query = SelectQuery()
    query.add_prefix(DC)
    query.add_variables("?title")
    where = GraphPattern()
    where.add_triples(Triple("?x", "dc:title", "?title"))
    where.add_filter(Filter('langMatches( lang(?title), "*" )'))
    return query.set_where_pattern(where)


@example("Q124")
def build_regex_with_flags_filter():
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    where.add_filter(Filter('regex(?name, "^ali", "i")'))
    return query.set_where_pattern(where)


@example("Q125")
def build_extension_function_filter():
    query = SelectQuery()
    query.add_prefixes([FOAF, Prefix("func", "http://example.org/functions#")])
    query.add_variables(["?name", "?id"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "func:empId", "?id")]
    )
    where.add_filter(Filter("func:even(?id)"))
    return query.set_where_pattern(where)


@example("Q126")
def build_multi_argument_extension_function_filter():
    query = SelectQuery()
    query.add_prefix(Prefix("aGeo", "http://example.org/geo#"))
    query.add_variables("?neighbor")
    where = GraphPattern()
    where.add_triples(
        [
            Triple("?a", "aGeo:placeName", '"Grenoble"'),
            Triple("?a", "aGeo:location", "?axLoc"),
            Triple("?a", "aGeo:location", "?ayLoc"),
            Triple("?b", "aGeo:placeName", "?neighbor"),
            Triple("?b", "aGeo:location", "?bxLoc"),
            Triple("?b", "aGeo:location", "?byLoc"),
        ]
    )
    where.add_filter(Filter("aGeo:distance(?axLoc, ?ayLoc, ?bxLoc, ?byLoc) < 10"))
    return query.set_where_pattern(where)


# --- examples the builders cannot express ------------------------------------

UNSUPPORTED: list[tuple[str, str]] = [
    ("Q5", "BASE declarations"),
    ("Q6", "BASE declarations"),
    ("Q56", "FROM (dataset clauses)"),
    ("Q59", "FROM NAMED (dataset clauses)"),
    ("Q63", "FROM and FROM NAMED (dataset clauses)"),
    ("Q77", "CONSTRUCT queries"),
    ("Q80", "CONSTRUCT queries"),
    ("Q82", "CONSTRUCT queries"),
    ("Q83", "CONSTRUCT queries"),
    ("Q85", "CONSTRUCT queries"),
    ("Q87", "DESCRIBE queries"),
    ("Q88", "DESCRIBE queries"),
    ("Q89", "DESCRIBE queries"),
    ("Q90", "DESCRIBE queries"),
    ("Q91", "DESCRIBE queries"),
    ("Q94", "ASK queries"),
    ("Q97", "ASK queries"),
]

#: Attribute names that would indicate a gap has since been filled. If one of
#: these appears, the corresponding examples should move into EXAMPLES above.
FEATURE_PROBES: dict[str, tuple[object, tuple[str, ...]]] = {
    "BASE declarations": (SPARQLQuery, ("base", "set_base", "add_base")),
    "FROM (dataset clauses)": (
        SPARQLSelectQuery,
        ("add_from", "add_from_named", "add_dataset", "set_dataset"),
    ),
    "FROM NAMED (dataset clauses)": (
        SPARQLSelectQuery,
        ("add_from", "add_from_named", "add_dataset", "set_dataset"),
    ),
    "FROM and FROM NAMED (dataset clauses)": (
        SPARQLSelectQuery,
        ("add_from", "add_from_named", "add_dataset", "set_dataset"),
    ),
}

#: Query forms with no builder class at all.
MISSING_QUERY_FORMS = {
    "CONSTRUCT queries": "SPARQLConstructQuery",
    "DESCRIBE queries": "SPARQLDescribeQuery",
    "ASK queries": "SPARQLAskQuery",
}


# --- tests -------------------------------------------------------------------


@pytest.mark.parametrize("ex", EXAMPLES, ids=example_id)
def test_builder_reproduces_w3c_example(ex: Example) -> None:
    """The builders produce valid SPARQL that means what the W3C example means."""
    check_example(ex)


@pytest.mark.parametrize("key", sorted(QUERIES, key=lambda k: int(k[1:])))
def test_reference_query_is_valid_sparql(key: str) -> None:
    """Guard the extracted reference text itself."""
    if key == "Q70":
        pytest.xfail("published example omits the declaration of the ':' prefix")
    assert_valid_sparql(QUERIES[key], f"W3C example {key}")


@pytest.mark.parametrize(
    ("key", "reason"), UNSUPPORTED, ids=[key for key, _ in UNSUPPORTED]
)
def test_unsupported_example_is_still_unsupported(key: str, reason: str) -> None:
    """
    Record the examples the builders cannot express.

    These fail once support is added, which is the signal to write a real test
    for the example and take it off this list.
    """
    assert key in QUERIES

    if reason in MISSING_QUERY_FORMS:
        import qleverlux.SPARQLQueryBuilder as builder

        class_name = MISSING_QUERY_FORMS[reason]
        assert not hasattr(builder, class_name), (
            f"{class_name} now exists; build W3C example {key} with it"
        )
    else:
        owner, attributes = FEATURE_PROBES[reason]
        present = [name for name in attributes if hasattr(owner, name)]
        assert not present, (
            f"{owner.__name__} gained {present}; build W3C example {key} with it"
        )


def test_every_w3c_example_is_accounted_for() -> None:
    """Every query in the document is either built or listed as unsupported."""
    built = {ex.key for ex in EXAMPLES}
    unsupported = {key for key, _ in UNSUPPORTED}

    assert not (built & unsupported), "an example is both built and listed unsupported"
    assert built | unsupported == set(QUERIES), (
        f"unaccounted for: {sorted(set(QUERIES) - built - unsupported)}"
    )


def test_example_keys_are_unique() -> None:
    keys = [ex.key for ex in EXAMPLES]
    assert len(keys) == len(set(keys))


# --- builder features the 2005 examples do not reach --------------------------
#
# The document predates GROUP BY, BIND, VALUES and sub-SELECTs, so nothing above
# exercises those classes even though the LUX middletier depends on all of them.
# These cover the remaining public API.


def test_group_by_and_aggregate_projection() -> None:
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?name", "(COUNT(?mbox) AS ?boxes)"])
    where = GraphPattern()
    where.add_triples(
        [Triple("?x", "foaf:name", "?name"), Triple("?x", "foaf:mbox", "?mbox")]
    )
    query.set_where_pattern(where)
    query.add_group_by(GroupBy(["?name"]))
    query.add_order_by(OrderBy(["?boxes"], descending=True))

    built = query.get_text()
    assert_valid_sparql(built, "GROUP BY query")
    assert algebra(built) == algebra(
        """PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name (COUNT(?mbox) AS ?boxes)
        WHERE { ?x foaf:name ?name . ?x foaf:mbox ?mbox }
        GROUP BY ?name ORDER BY DESC(?boxes)"""
    )


def test_bind_values_and_filter_not_exists() -> None:
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables(["?x", "?label"])
    where = GraphPattern()
    where.add_value(Values(["http://example.org/a", "http://example.org/b"], "?x"))
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    where.add_binding(
        Binding(IfClause(Bound("?name"), '"named"', '"anonymous"'), "?label")
    )
    absent = GraphPattern(not_exists=True)
    absent.add_triples(Triple("?x", "foaf:mbox", "?mbox"))
    where.add_nested_graph_pattern(absent)
    query.set_where_pattern(where)

    built = query.get_text()
    assert_valid_sparql(built, "BIND/VALUES query")
    assert algebra(built) == algebra(
        """PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?x ?label WHERE {
          VALUES ?x { <http://example.org/a> <http://example.org/b> }
          ?x foaf:name ?name .
          BIND (IF (BOUND (?name), "named", "anonymous") AS ?label)
          FILTER NOT EXISTS { ?x foaf:mbox ?mbox }
        }"""
    )


def test_nested_select_query() -> None:
    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()

    inner = SelectQuery(distinct=True, limit=5)
    inner.add_variables("?x")
    inner_where = GraphPattern()
    inner_where.add_triples(Triple("?x", "a", "foaf:Person"))
    inner.set_where_pattern(inner_where)
    where.add_nested_select_query(inner)

    where.add_triples(Triple("?x", "foaf:name", "?name"))
    query.set_where_pattern(where)

    built = query.get_text()
    assert_valid_sparql(built, "sub-SELECT query")
    assert algebra(built) == algebra(
        """PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT ?name WHERE {
          { SELECT DISTINCT ?x WHERE { ?x a foaf:Person } LIMIT 5 }
          ?x foaf:name ?name .
        }"""
    )


def test_blank_node_property_list() -> None:
    """
    BNode renders an inline ``[ ... ]`` property list.

    Blank node labels are freshly generated on every parse, so this checks the
    parsed structure rather than comparing algebra to a reference.
    """
    import rdflib

    query = SelectQuery()
    query.add_prefix(FOAF)
    query.add_variables("?name")
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?name"))
    where.add_bnode(
        BNode([Triple("", "foaf:knows", "?x"), Triple("", "foaf:nick", '"anon"')])
    )
    query.set_where_pattern(where)

    built = query.get_text()
    assert_valid_sparql(built, "blank node query")

    triples = algebra(built)["p"]["p"]["triples"]
    knows = rdflib.URIRef("http://xmlns.com/foaf/0.1/knows")
    nick = rdflib.URIRef("http://xmlns.com/foaf/0.1/nick")

    subjects = {
        subject
        for subject, predicate, _ in triples
        if predicate in (knows, nick) and isinstance(subject, rdflib.BNode)
    }
    assert len(subjects) == 1, "both predicates should hang off one blank node"
    assert (subjects.pop(), knows, rdflib.Variable("x")) in triples


def test_update_query_round_trips() -> None:
    query = SPARQLUpdateQuery()
    query.add_prefix(FOAF)
    delete = GraphPattern()
    delete.add_triples(Triple("?x", "foaf:name", "?old"))
    insert = GraphPattern()
    insert.add_triples(Triple("?x", "foaf:name", '"redacted"'))
    where = GraphPattern()
    where.add_triples(Triple("?x", "foaf:name", "?old"))
    query.set_delete_pattern(delete)
    query.set_insert_pattern(insert)
    query.set_where_pattern(where)

    built = query.get_text()
    parsed = parser.parseUpdate(built)
    assert parsed is not None
