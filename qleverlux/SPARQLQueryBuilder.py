"""
SPARQL query builder: graph patterns and the queries that contain them.

Rendering is fragment-based. Every node appends its pieces to a shared list via
``emit_into()`` and the outermost ``get_text()`` performs a single join, so a
deeply nested query is assembled in one pass instead of by repeated string
concatenation at each level. Adding a new kind of pattern element means
implementing ``emit_into()`` on it; the containers do no type dispatch.

Derived from SPARQL Burger, created by Panos Mitzias (http://pmitzias.com/SPARQLBurger)
and powered by Catalink Ltd (http://catalink.eu).
Rewritten for qleverlux by Rob Sanderson (robert.sanderson@yale.edu).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 3.11+ only; annotations are strings here, so it is never needed at runtime.
    from typing import Self

from qleverlux.SPARQLSyntaxTerms import (
    AbstractTerm,
    Binding,
    Filter,
    GroupBy,
    Having,
    OrderBy,
    Prefix,
    Triple,
    Values,
    indent,
)

__all__ = [
    "POPULAR_PREFIXES",
    "BNode",
    "GraphPattern",
    "SPARQLGraphPattern",
    "SPARQLQuery",
    "SPARQLSelectQuery",
    "SPARQLUpdateQuery",
    "SelectQuery",
    "UpdateQuery",
]

POPULAR_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xml": "http://www.w3.org/2001/XMLSchema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "prov": "http://www.w3.org/ns/prov#",
    "foaf": "http://xmlns.com/foaf/0.1/",
}


def _as_triples(triples: Triple | Iterable[Triple]) -> tuple[Triple, ...]:
    """Normalize a single triple or any iterable of triples into a tuple."""
    if isinstance(triples, Triple):
        return (triples,)
    result = tuple(triples)
    if not all(isinstance(element, Triple) for element in result):
        raise TypeError("add_triples() expects a Triple or an iterable of Triples")
    return result


def _check(value: object, expected: type, method: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(
            f"{method}() expects a {expected.__name__}, got {type(value).__name__}"
        )


class BNode(AbstractTerm):
    """
    A blank node written inline, e.g. ``[ view:column-word "fish" ; view:column-uri ?s ]``.

    Only the predicate and object of each triple are used; the subject is the
    blank node itself and is ignored.
    """

    __slots__ = ("graph",)

    def __init__(self, triples: Triple | Iterable[Triple] | None = None) -> None:
        self.graph: list[Triple] = []
        if triples is not None:
            self.add_triples(triples)

    def add_triples(self, triples: Triple | Iterable[Triple]) -> Self:
        """Add predicate/object pairs to the blank node. Returns self, for chaining."""
        self.graph.extend(_as_triples(triples))
        return self

    def get_text(self, indentation_depth: int = 0) -> str:
        outer = indent(indentation_depth)
        inner = indent(indentation_depth + 1)
        # Predicate lines carry the inner indent twice: once from the separator
        # (or the opening bracket) and once from the line itself.
        entries = f" ;\n{inner}".join(
            f"{inner}{triple.predicate} {triple.object}" for triple in self.graph
        )
        return f"{outer}[\n{inner}{entries} {outer}]"

    def emit_into(self, parts: list[str], indentation_depth: int = 0) -> None:
        parts += (indent(indentation_depth + 1), self.get_text(), "\n")


class SPARQLGraphPattern(AbstractTerm):
    """
    A braced group of triples, nested patterns, bindings, filters and values.

    The group is plain by default; ``optional``, ``union``, ``not_exists``,
    ``graph_name`` and ``service`` select the wrapping construct instead, and are
    honoured in that order of precedence.
    """

    __slots__ = (
        "bindings",
        "filters",
        "graph",
        "graph_name",
        "is_not_exists",
        "is_optional",
        "is_union",
        "service_name",
        "values",
    )

    def __init__(
        self,
        optional: bool = False,
        union: bool = False,
        not_exists: bool = False,
        service: str = "",
        graph_name: str = "",
    ) -> None:
        if not_exists and (optional or union):
            raise ValueError("FILTER NOT EXISTS cannot be used with OPTIONAL or UNION")

        self.is_optional = optional
        self.is_union = union
        self.is_not_exists = not_exists
        self.graph_name = graph_name
        self.service_name = service

        self.graph: list[AbstractTerm] = []
        self.filters: list[Filter | Having] = []
        self.bindings: list[Binding] = []
        self.values: list[Values] = []

    def add_triples(self, triples: Triple | Iterable[Triple]) -> Self:
        """Add one or more triples to the pattern. Returns self, for chaining."""
        self.graph.extend(_as_triples(triples))
        return self

    def add_bnode(self, bnode: BNode) -> Self:
        """Add an inline blank node to the pattern."""
        _check(bnode, BNode, "add_bnode")
        self.graph.append(bnode)
        return self

    def add_nested_graph_pattern(self, graph_pattern: SPARQLGraphPattern) -> Self:
        """Nest another graph pattern inside this one."""
        _check(graph_pattern, SPARQLGraphPattern, "add_nested_graph_pattern")
        self.graph.append(graph_pattern)
        return self

    def add_nested_select_query(self, select_query: SPARQLSelectQuery) -> Self:
        """Nest a sub-SELECT inside this pattern."""
        _check(select_query, SPARQLSelectQuery, "add_nested_select_query")
        self.graph.append(select_query)
        return self

    def add_filter(self, filter: Filter) -> Self:
        """Add a FILTER expression, rendered after the pattern's triples."""
        _check(filter, Filter, "add_filter")
        self.filters.append(filter)
        return self

    def add_having(self, filter: Having) -> Self:
        """Add a HAVING expression, rendered alongside the filters."""
        _check(filter, Having, "add_having")
        self.filters.append(filter)
        return self

    def add_binding(self, binding: Binding) -> Self:
        """Add a BIND expression, rendered after the pattern's triples."""
        _check(binding, Binding, "add_binding")
        self.bindings.append(binding)
        return self

    def add_value(self, value: Values) -> Self:
        """Add a VALUES clause, rendered before the pattern's triples."""
        _check(value, Values, "add_value")
        self.values.append(value)
        return self

    def get_text(self, indentation_depth: int = 0) -> str:
        """Render the pattern at the given depth."""
        parts: list[str] = []
        # emit_into() is told the depth of the *enclosing* pattern, so step back
        # one level to land on indentation_depth.
        self.emit_into(parts, indentation_depth - 1)
        return "".join(parts)

    def emit_into(self, parts: list[str], indentation_depth: int = -1) -> None:
        """Emit the pattern one level inside a container at ``indentation_depth``."""
        depth = indentation_depth + 1
        outer = indent(depth)
        inner = indent(depth + 1)

        if self.is_optional:
            parts += (outer, "OPTIONAL {\n")
        elif self.is_union:
            parts += (outer, "UNION\n", outer, "{\n")
        elif self.is_not_exists:
            parts += (outer, "FILTER NOT EXISTS {\n")
        elif self.graph_name:
            parts += (outer, "GRAPH ", self.graph_name, " {\n")
        elif self.service_name:
            service = self.service_name
            if ":" not in service:
                service = f"{service}:"
            parts += (outer, "SERVICE ", service, " {\n")
        else:
            parts += (outer, "{\n")

        for value in self.values:
            parts += (inner, value.get_text(), "\n")

        for entry in self.graph:
            entry.emit_into(parts, depth)

        for binding in self.bindings:
            parts += (inner, binding.get_text(), "\n")

        for filter in self.filters:
            parts += (inner, filter.get_text(), "\n")

        parts += (outer, "}\n")


class SPARQLQuery(AbstractTerm):
    """Shared prefix and WHERE handling for the concrete query types."""

    __slots__ = ("prefixes", "where")

    def __init__(self, include_popular_prefixes: bool = False) -> None:
        self.prefixes: list[Prefix] = []
        self.where: SPARQLGraphPattern | None = None

        if include_popular_prefixes:
            self.add_popular_prefixes()

    def add_prefix(self, prefix: Prefix) -> Self:
        """Add a single PREFIX declaration."""
        _check(prefix, Prefix, "add_prefix")
        self.prefixes.append(prefix)
        return self

    def add_prefixes(self, prefixes: Iterable[Prefix]) -> Self:
        """Add several PREFIX declarations."""
        for prefix in prefixes:
            self.add_prefix(prefix)
        return self

    def add_popular_prefixes(self) -> Self:
        """Add the common RDF/RDFS/OWL/FOAF namespace declarations."""
        return self.add_prefixes(
            Prefix(prefix, namespace) for prefix, namespace in POPULAR_PREFIXES.items()
        )

    def set_where_pattern(self, graph_pattern: SPARQLGraphPattern) -> Self:
        """Set the graph pattern used for the WHERE clause."""
        _check(graph_pattern, SPARQLGraphPattern, "set_where_pattern")
        self.where = graph_pattern
        return self

    def _emit_clause(self, parts: list[str], keyword: str, pattern, depth: int) -> None:
        """Emit ``<keyword> { ... }``, dropping the pattern's trailing newline."""
        parts += ("\n", indent(depth), keyword, " ")
        pattern.emit_into(parts, depth - 1)
        parts[-1] = parts[-1][:-1]


class SPARQLSelectQuery(SPARQLQuery):
    """A SELECT query, optionally distinct, grouped, ordered, limited and offset."""

    __slots__ = ("distinct", "group_by", "limit", "offset", "order_by", "variables")

    def __init__(
        self,
        distinct: bool = False,
        limit: int = 0,
        include_popular_prefixes: bool = False,
        offset: int = 0,
    ) -> None:
        super().__init__(include_popular_prefixes)

        self.distinct = distinct
        self.limit = limit
        self.offset = offset
        self.variables: list[str] = []
        self.group_by: list[GroupBy] = []
        self.order_by: list[OrderBy] = []

    def add_variables(self, variables: str | Iterable[str]) -> Self:
        """Add projected variables or expressions, e.g. ``(COUNT(?s) AS ?c)``."""
        if isinstance(variables, str):
            variables = (variables,)
        else:
            variables = tuple(variables)
            if not all(isinstance(element, str) for element in variables):
                raise TypeError("add_variables() expects strings")
        self.variables.extend(variables)
        return self

    def add_group_by(self, group: GroupBy) -> Self:
        """Add a GROUP BY clause."""
        _check(group, GroupBy, "add_group_by")
        self.group_by.append(group)
        return self

    def add_order_by(self, order_by: OrderBy) -> Self:
        """Add an ORDER BY condition; conditions are rendered in insertion order."""
        _check(order_by, OrderBy, "add_order_by")
        self.order_by.append(order_by)
        return self

    def get_text(self, indentation_depth: int = 0) -> str:
        outer = indent(indentation_depth)
        parts: list[str] = [prefix.get_text() for prefix in self.prefixes]

        parts += ("\n", outer, "SELECT ")
        if self.distinct:
            parts.append("DISTINCT ")
        parts.append(" ".join(self.variables) if self.variables else " *")

        if self.where is not None:
            self._emit_clause(parts, "WHERE", self.where, indentation_depth)
        else:
            parts += ("\n", outer, "WHERE ")

        for group in self.group_by:
            parts += ("\n", outer, group.get_text())

        if self.order_by:
            parts += ("\n", outer, "ORDER BY")
            for order in self.order_by:
                parts += (" ", order.get_text())

        if self.limit > 0:
            parts += ("\nLIMIT ", str(self.limit))
        if self.offset > 0:
            parts += ("\nOFFSET ", str(self.offset))

        return "".join(parts)

    def emit_into(self, parts: list[str], indentation_depth: int = 0) -> None:
        inner = indent(indentation_depth + 1)
        parts += (inner, "{", self.get_text(indentation_depth + 2), inner, "}\n")


class SPARQLUpdateQuery(SPARQLQuery):
    """A DELETE/INSERT/WHERE update query."""

    __slots__ = ("delete", "insert")

    def __init__(self, include_popular_prefixes: bool = False) -> None:
        super().__init__(include_popular_prefixes)
        self.delete: SPARQLGraphPattern | None = None
        self.insert: SPARQLGraphPattern | None = None

    def set_delete_pattern(self, graph_pattern: SPARQLGraphPattern) -> Self:
        """Set the graph pattern used for the DELETE clause."""
        _check(graph_pattern, SPARQLGraphPattern, "set_delete_pattern")
        self.delete = graph_pattern
        return self

    def set_insert_pattern(self, graph_pattern: SPARQLGraphPattern) -> Self:
        """Set the graph pattern used for the INSERT clause."""
        _check(graph_pattern, SPARQLGraphPattern, "set_insert_pattern")
        self.insert = graph_pattern
        return self

    def get_text(self, indentation_depth: int = 0) -> str:
        parts: list[str] = [prefix.get_text() for prefix in self.prefixes]

        for keyword, pattern in (
            ("DELETE", self.delete),
            ("INSERT", self.insert),
            ("WHERE", self.where),
        ):
            if pattern is not None:
                self._emit_clause(parts, keyword, pattern, indentation_depth)

        return "".join(parts)


GraphPattern = SPARQLGraphPattern
SelectQuery = SPARQLSelectQuery
UpdateQuery = SPARQLUpdateQuery
