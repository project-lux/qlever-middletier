"""
Shared machinery for the builder test suites.

Queries are compared by SPARQL algebra rather than by string equality: the
builders have their own layout, expand ``;`` predicate-object lists, always
qualify ORDER BY conditions, and place FILTERs at the end of their group.
Algebra comparison is blind to all of that but still catches a query that means
something different.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

parser = pytest.importorskip("rdflib.plugins.sparql.parser")
algebra_module = pytest.importorskip("rdflib.plugins.sparql.algebra")


def algebra(query_text: str):
    """Parse a query and return its SPARQL algebra, the normalized comparison form."""
    return algebra_module.translateQuery(parser.parseQuery(query_text)).algebra


_BNODE_LABEL = re.compile(r"BNode\('([^']*)'\)")
_SERVICE_STRING = re.compile(r"'service_string': '(.*?)(?<!\\)'", re.S)
_VARS_SET = re.compile(r"'_vars': (?:\{[^{}]*\}|set\(\))")


def _canonical(tree) -> str:
    """
    Render an algebra tree as text, with three incidental differences removed.

    Comparing the rendered tree rather than the tree itself is deliberate:
    rdflib's CompValue equality ignores the node's name, so an algebra holding
    ``Aggregate_Count_`` compares equal to one holding ``Aggregate_Sum_`` when
    their contents match. The rendered form keeps the names.

    What is removed:

    * blank node labels, which rdflib mints afresh on every parse, are renumbered
      in order of first appearance - so two distinct blank nodes are still told
      apart from one blank node used twice;
    * the raw source text rdflib stores beside a parsed SERVICE block, which
      carries the layout of the original query rather than its meaning;
    * the ``_vars`` sets, which are derived from the rest of the tree and whose
      rendering depends on set iteration order.
    """
    text = _SERVICE_STRING.sub("'service_string': <elided>", repr(tree))
    text = _VARS_SET.sub("'_vars': <derived>", text)

    labels: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        label = match.group(1)
        if label not in labels:
            labels[label] = f"_b{len(labels)}"
        return f"BNode('{labels[label]}')"

    return _BNODE_LABEL.sub(rename, text)


def same_query(left, right) -> bool:
    """True when two algebra trees say the same thing."""
    return _canonical(left) == _canonical(right)


def assert_valid_sparql(query_text: str, label: str) -> None:
    try:
        parser.parseQuery(query_text)
    except Exception as exc:  # noqa: BLE001 - the rdflib parser raises bare Exception
        pytest.fail(f"{label} is not valid SPARQL: {exc}\n\n{query_text}")


@dataclass(frozen=True)
class Example:
    """One published test query, together with the builder code that reproduces it."""

    key: str
    name: str
    build: Callable[[], object]
    references: dict[str, str] = field(repr=False, default_factory=dict)
    equivalent_to: str | None = None
    note: str | None = None

    @property
    def reference(self) -> str:
        """The query exactly as published."""
        return self.references[self.key]

    @property
    def expected(self) -> str:
        """What we compare against: the published query unless overridden."""
        return self.equivalent_to if self.equivalent_to is not None else self.reference


class ExampleRegistry:
    """Collects builder functions and pairs each with its published reference."""

    def __init__(self, references: dict[str, str]) -> None:
        self.references = references
        self.examples: list[Example] = []

    def __call__(
        self, key: str, *, equivalent_to: str | None = None, note: str | None = None
    ):
        """Register the decorated function as the builder for the given test."""
        if key not in self.references:
            raise KeyError(f"no reference query for {key!r}")

        def register(fn: Callable[[], object]) -> Callable[[], object]:
            self.examples.append(
                Example(
                    key=key,
                    name=fn.__name__.removeprefix("build_"),
                    build=fn,
                    references=self.references,
                    equivalent_to=equivalent_to,
                    note=note,
                )
            )
            return fn

        return register

    def __iter__(self):
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def keys(self) -> set[str]:
        return {example.key for example in self.examples}


def check_example(example: Example) -> None:
    """Assert the builder output is valid SPARQL and means what the published test means."""
    built = example.build().get_text()

    assert_valid_sparql(built, f"{example.key} builder output")
    assert_valid_sparql(example.expected, f"{example.key} reference query")

    if not same_query(algebra(built), algebra(example.expected)):
        note = f"note: {example.note}\n" if example.note else ""
        pytest.fail(
            f"{example.key} does not match the published test.\n{note}"
            f"--- built ---\n{built}\n"
            f"--- expected ---\n{example.expected}\n"
            f"--- built algebra ---\n{algebra(built)}\n"
            f"--- expected algebra ---\n{algebra(example.expected)}\n"
        )


def example_id(example: Example) -> str:
    return f"{example.key}-{example.name}"
