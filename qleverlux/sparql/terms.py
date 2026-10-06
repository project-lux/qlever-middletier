"""
SPARQL syntax terms: the leaf nodes used to build a query.

Every term renders itself with ``get_text()``. Terms that can appear directly
inside a graph pattern also implement ``emit_into()``, which appends their
rendering to a list of string fragments; the enclosing pattern joins those
fragments once, rather than concatenating strings at every level.

Derived from SPARQL Burger, created by Panos Mitzias (http://pmitzias.com/SPARQLBurger)
and powered by Catalink Ltd (http://catalink.eu).
Rewritten for qleverlux by Rob Sanderson (robert.sanderson@yale.edu).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "INDENT_UNIT",
    "AbstractTerm",
    "Binding",
    "Bound",
    "Filter",
    "GroupBy",
    "Having",
    "IfClause",
    "OrderBy",
    "Prefix",
    "Triple",
    "Values",
    "Variable",
    "in_brackets",
    "indent",
]

#: One level of indentation in the generated SPARQL.
INDENT_UNIT = "   "

# Indentation strings are recomputed constantly while rendering, and nesting is
# never more than a handful of levels deep, so memoize them in a flat list.
_INDENTS: list[str] = [""]


def indent(depth: int) -> str:
    """Return the indentation string for the given nesting depth."""
    try:
        return _INDENTS[depth]
    except IndexError:
        while len(_INDENTS) <= depth:
            _INDENTS.append(_INDENTS[-1] + INDENT_UNIT)
        return _INDENTS[depth]


class AbstractTerm(ABC):
    """Base class for anything that can render itself as SPARQL."""

    __slots__ = ()

    @abstractmethod
    def get_text(self, indentation_depth: int = 0) -> str:
        """Render this term as SPARQL text."""

    def emit_into(self, parts: list[str], indentation_depth: int = 0) -> None:
        """
        Append this term's fragments as an element of an enclosing graph pattern.

        The default is the form used by triples: one indented, self-terminating
        line. Terms that nest (patterns, sub-selects, blank nodes) override it.
        """
        parts.append(indent(indentation_depth + 1))
        parts.append(self.get_text())

    def __str__(self) -> str:
        return self.get_text()

    def __repr__(self) -> str:
        # Dataclass terms generate a more informative repr; this covers the
        # mutable container terms, whose contents are too large to show.
        return f"{self.__class__.__name__}(...)"


def text_of(value: str | AbstractTerm) -> str:
    """Render a value that may be either literal SPARQL text or a nested term."""
    return value if isinstance(value, str) else value.get_text()


def in_brackets(uri: str) -> str:
    """
    Enclose a URI in angle brackets, leaving anything else untouched.

    Already-bracketed URIs and non-URI tokens (variables, prefixed names) are
    returned unchanged.
    """
    if uri.startswith("<"):
        return uri
    if uri.startswith("http"):
        return f"<{uri}>"
    return uri


@dataclass(slots=True, eq=False)
class Prefix(AbstractTerm):
    """A PREFIX declaration, e.g. ``PREFIX ex: <http://www.example.com#>``."""

    prefix: str
    namespace: str

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"PREFIX {self.prefix}: <{self.namespace}>\n"


@dataclass(slots=True, eq=False)
class Triple(AbstractTerm):
    """A single triple pattern, e.g. ``?person ex:hasName 'John'@en``."""

    subject: str
    predicate: str
    object: str

    def __post_init__(self) -> None:
        # Callers pass variables, literals and other terms interchangeably.
        self.subject = str(self.subject)
        self.predicate = str(self.predicate)
        self.object = str(self.object)

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"{self.subject} {self.predicate} {self.object} . \n"


@dataclass(slots=True, eq=False)
class Filter(AbstractTerm):
    """A FILTER expression, e.g. ``FILTER (?age > 30)``."""

    expression: str

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"FILTER ({self.expression})"


@dataclass(slots=True, eq=False)
class Having(AbstractTerm):
    """
    A HAVING condition, e.g. ``HAVING (COUNT(?x) > 2)``.

    HAVING belongs to the query rather than to a graph pattern: add it with
    ``SPARQLSelectQuery.add_having()``, which renders it after GROUP BY.
    """

    expression: str

    @property
    def condition(self) -> str:
        """The condition alone, for a query listing several under one keyword."""
        return f"({self.expression})"

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"HAVING {self.condition}"


@dataclass(slots=True, eq=False)
class Binding(AbstractTerm):
    """A BIND expression. The bound value may itself be a term, e.g. an IfClause."""

    value: str | AbstractTerm
    variable: str

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"BIND ({text_of(self.value)} AS {self.variable})"


@dataclass(slots=True, eq=False)
class Bound(AbstractTerm):
    """A BOUND test, e.g. ``BOUND (?name)``."""

    variable: str | AbstractTerm

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"BOUND ({text_of(self.variable)})"


@dataclass(slots=True, eq=False)
class IfClause(AbstractTerm):
    """An IF expression, e.g. ``IF (?age > 18, 'adult', 'minor')``. Nestable."""

    condition: str | AbstractTerm
    true_value: str | AbstractTerm
    false_value: str | AbstractTerm

    def get_text(self, indentation_depth: int = 0) -> str:
        return (
            f"IF ({text_of(self.condition)}, "
            f"{text_of(self.true_value)}, "
            f"{text_of(self.false_value)})"
        )


@dataclass(slots=True, eq=False)
class GroupBy(AbstractTerm):
    """A GROUP BY clause over one or more variables."""

    variables: Sequence[str]

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"GROUP BY {' '.join(self.variables)}"


@dataclass(slots=True, eq=False)
class OrderBy(AbstractTerm):
    """One ordering condition, e.g. ``DESC(?score)``."""

    variables: Sequence[str]
    descending: bool = False

    @property
    def order(self) -> str:
        return "DESC" if self.descending else "ASC"

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"{self.order}({' '.join(self.variables)})"


@dataclass(slots=True, eq=False)
class Values(AbstractTerm):
    """A VALUES clause binding a variable to a fixed set of terms."""

    values: Sequence[str]
    name: str

    def get_text(self, indentation_depth: int = 0) -> str:
        enclosed = " ".join(in_brackets(value) for value in self.values)
        return f"VALUES {self.name} {{{enclosed}}}"


@dataclass(slots=True, eq=False)
class Variable(AbstractTerm):
    """A variable reference. The name is given without the leading ``?``."""

    name: str

    def get_text(self, indentation_depth: int = 0) -> str:
        return f"?{self.name}"
