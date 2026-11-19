"""
The middle tier must not import the vector build or the sidecar.

``lux_embed`` (the build) and ``sparql_service`` (the SERVICE QLever calls) live
in this repo but run as separate commands in their own venv, with torch and the
patched USearch. The middle tier reaches vectors only through QLever, so an
import from ``qleverlux`` into either, or into ``qleverlux.vectors`` (the store
they share), would drag that stack into the server for no reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIDDLE_TIER = ROOT / "qleverlux"
FORBIDDEN = ("lux_embed", "sparql_service", "qleverlux.vectors", "usearch", "torch", "sentence_transformers")


def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module
            if node.module == "qleverlux":
                for alias in node.names:
                    yield node.lineno, f"qleverlux.{alias.name}"


def test_middle_tier_does_not_import_vector_stack():
    offenders = []
    for path in sorted(MIDDLE_TIER.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[:2] == ("qleverlux", "vectors"):
            continue
        for lineno, name in _imports(path):
            if any(name == f or name.startswith(f + ".") for f in FORBIDDEN):
                offenders.append(f"{rel}:{lineno} imports {name}")
    assert not offenders, "\n".join(offenders)
