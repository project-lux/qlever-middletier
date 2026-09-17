#!/usr/bin/env python3
"""Re-derive ``queries/*.json`` and the generated ``config/*.json`` files from
the upstream LUX repositories.

Two upstreams are involved, because the files have two different origins:

``../lux-middletier`` (the Node.js middle tier that talks to MarkLogic)
    ``queries/*.json``      <- ``lib/build-query/queries.js`` (each query
                               function called with the ``URI-HERE`` sentinel)
    ``config/hal_links.json`` <- ``lib/build-query/builder.js``
                               (``keyFuncNameMap`` for the scope,
                                ``queryBuilders`` for the query name + href)

``../lux-marklogic`` (the MarkLogic backend, where the search vocabulary lives)
    ``config/facets.json``              <- ``config/facetsConfig.mjs``
    ``config/related_lists.json``       <- ``config/relationNames.mjs``
    ``config/terms_inverse.json``       <- ``config/searchTermsConfig.mjs``
    ``config/related_list_scopes.json`` <- a Python port of the two deployment
                                           generators in ``runDuringDeployment/``
                                           (``generateRemainingSearchTerms.mjs``
                                           and ``generateRelatedListsConfig.mjs``)

``config/stopwords.json`` comes from the deployed advanced search config, which
``luxql`` ships as ``advanced-search-config.json``.

``config/sorts.json`` is *not* derivable: its keys come from MarkLogic's
``searchResultsSortConfig.mjs`` but its values are QLever ``lux:`` predicate
paths that have no counterpart upstream.  It is audited instead, so new or
removed sort keys get reported and can be mapped by hand.

Requires ``node`` on PATH (used to evaluate the upstream JS; no npm install is
needed, neither upstream module graph has third-party dependencies).

``--base-dir`` picks which middle tier gets updated: any directory holding
``config/`` and ``queries/``, so a deployed instance can be refreshed without
going through this checkout.  It defaults to ``$QLMT_BASE_DIR``, else the
checkout this script lives in.  Only the written files move with it -- the
QLever predicate vocabulary still comes from the ``qleverlux`` serving that
instance (whichever one is importable, else this checkout's).

Usage::

    python files/derive_from_upstream.py                # dry run, report drift
    python files/derive_from_upstream.py --write        # apply
    python files/derive_from_upstream.py --write --prune # also delete stale queries
    python files/derive_from_upstream.py --only queries,hal_links
    python files/derive_from_upstream.py --check        # exit 1 if out of date

    # update a deployed instance instead of this checkout
    python files/derive_from_upstream.py -b ~/instances/mds/middletier --write
"""

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Where this script lives.  Only used to find qleverlux/sparql.py and the
# fallback overrides file -- the files being written live under --base-dir,
# which may be a deployed instance somewhere else entirely.
CODE_REPO = os.path.dirname(HERE)

# MarkLogic pattern names, from lib/search/patterns/*.mjs.  Inlined because the
# defining modules import the MarkLogic runtime and cannot be evaluated here.
PATTERN_HOP_WITH_FIELD = "hopWithField"
PATTERN_HOP_INVERSE = "hopInverse"
PATTERN_INDEXED_VALUE = "indexedValue"
PATTERN_INDEXED_RANGE = "indexedRange"
PATTERN_DATE_RANGE = "dateRange"
PATTERN_IRI = "iri"
PATTERN_RELATED_LIST = "relatedList"

# RELATION_KEYS_TO_SUPPRESS, from runDuringDeployment/generateRelatedListsConfig.mjs.
SUPPRESSED_RELATION_KEYS = {
    "curated-containingItem-memberOf-usedForEvent",
    "curated-usedForEvent",
    "used-containingItem-memberOf-curatedBy",
    "used-curatedBy",
    # ignoring keys with relationScope of "set"
    "classificationOfSet-containingItem-memberOf-usedForEvent",
    "classificationOfSet-usedForEvent",
    # ignore all keys that start with publishedHere - #284
    "publishedHere-aboutAgent",
    "publishedHere-aboutConcept",
    "publishedHere-aboutPlace",
    "publishedHere-carriedBy-memberOf-usedForEvent",
    "publishedHere-classification",
    "publishedHere-createdAt",
    "publishedHere-createdBy",
    "publishedHere-language",
    "publishedHere-publishedAt",
    "publishedHere-publishedBy",
    # ignore all keys that end with publishedAt - #284
    "subjectOfAgent-publishedAt",
    "subjectOfConcept-publishedAt",
    "subjectOfPlace-publishedAt",
    "classificationOfWork-publishedAt",
    "created-publishedAt",
    "createdHere-publishedAt",
    "languageOf-publishedAt",
    "published-publishedAt",
}


# --------------------------------------------------------------------------- #
# Node helpers
# --------------------------------------------------------------------------- #

# Imports builder.js/queries.js from an arbitrary checkout and dumps everything
# we need as JSON.  Calling the real functions beats parsing the source: the
# query JSON is exactly what the Node middle tier would send.
DUMP_MIDDLETIER_JS = r"""
import { pathToFileURL } from 'node:url'
import path from 'node:path'

const root = process.argv[2]
const href = (p) => pathToFileURL(path.join(root, p)).href

const { keyFuncNameMap, queryBuilders } = await import(href('lib/build-query/builder.js'))
const { default: queries } = await import(href('lib/build-query/queries.js'))

const out = { queries: {}, keyFuncNameMap, builders: {} }
for (const [name, fn] of Object.entries(queries)) {
  out.queries[name] = fn('URI-HERE')
}
for (const [key, fn] of Object.entries(queryBuilders)) {
  // The source is enough: we want the href as a template, not a resolved URL.
  out.builders[key] = fn.toString()
}
process.stdout.write(JSON.stringify(out))
"""

# Pulls a single top-level `const NAME = {...}` object literal out of an .mjs
# file and evaluates it.  The MarkLogic config modules import MarkLogic-only
# modules, so they cannot be imported wholesale.
EXTRACT_LITERAL_JS = r"""
import fs from 'node:fs'

const [file, name] = process.argv.slice(2)
const src = fs.readFileSync(file, 'utf8')
const start = src.indexOf(`const ${name} = {`)
if (start < 0) throw new Error(`no top-level const ${name} in ${file}`)

let depth = 0
let end = -1
let inString = null
let inComment = null
const open = src.indexOf('{', start)
for (let i = open; i < src.length; i++) {
  const c = src[i]
  const n = src[i + 1]
  if (inComment === 'line') { if (c === '\n') inComment = null; continue }
  if (inComment === 'block') { if (c === '*' && n === '/') { inComment = null; i++ } continue }
  if (inString) { if (c === '\\') { i++; continue } if (c === inString) inString = null; continue }
  if (c === '/' && n === '/') { inComment = 'line'; i++; continue }
  if (c === '/' && n === '*') { inComment = 'block'; i++; continue }
  if (c === '"' || c === "'" || c === '`') { inString = c; continue }
  if (c === '{') depth++
  else if (c === '}') { depth--; if (depth === 0) { end = i; break } }
}
if (end < 0) throw new Error(`unbalanced braces reading ${name} from ${file}`)

const value = (0, eval)(`(${src.slice(open, end + 1)})`)
process.stdout.write(JSON.stringify(value))
"""


class NodeRunner:
    """Writes the helper scripts to a temp dir and runs them."""

    def __init__(self):
        if shutil.which("node") is None:
            sys.exit("error: `node` is required on PATH to read the upstream JS")
        self.dir = tempfile.mkdtemp(prefix="lux-derive-")
        self.scripts = {}
        for name, src in (
            ("dump_middletier.mjs", DUMP_MIDDLETIER_JS),
            ("extract_literal.mjs", EXTRACT_LITERAL_JS),
        ):
            path = os.path.join(self.dir, name)
            with open(path, "w") as fh:
                fh.write(src)
            self.scripts[name] = path

    def run(self, script, *args):
        proc = subprocess.run(
            ["node", self.scripts[script], *args],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.exit(f"error: node {script} failed:\n{proc.stderr.strip()}")
        return json.loads(proc.stdout)

    def literal(self, mjs_path, const_name):
        return self.run("extract_literal.mjs", mjs_path, const_name)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# lux-middletier: queries/*.json and config/hal_links.json
# --------------------------------------------------------------------------- #

_RETURNED_TEMPLATE = re.compile(r"return\s+`([^`]*)`")
_QUERY_CALL = re.compile(r"queries\.(\w+)\(")
_RELATED_LIST_NAME = re.compile(r"[?&]name=([^&`]+)")


def derive_queries(dump):
    """Each entry of the `queries` map, built with the URI-HERE sentinel.

    The file name is the key in `queries.js`, which is what `hal_links.json`
    refers to -- note that it is not always the source file name, nor the
    function's own `.name`.
    """
    return dict(dump["queries"])


def derive_hal_links(dump):
    """HAL relation -> {queryName, template, scope} (+ relatedList when the
    link points at /api/related-list rather than a query)."""
    scopes = {}
    for scope, relations in dump["keyFuncNameMap"].items():
        for relation in relations:
            # lux:workCarriedBy / lux:workWorksAbout are spread into both the
            # set and work maps; last one wins, matching the checked-in file.
            scopes[relation] = scope

    links = {}
    for relation, source in sorted(dump["builders"].items()):
        match = _RETURNED_TEMPLATE.search(source)
        if match is None:
            print(f"  ! no template literal in queryBuilders[{relation}], skipped")
            continue
        template = (
            match.group(1)
            .replace("${config.searchUriHost}", "{searchUriHost}")
            .replace("${q}", "{q}")
            .replace("${idEnc}", "{id}")
            .replace("${encodeURIComponent(id)}", "{id}")
        )
        if "${" in template:
            print(f"  ! unresolved substitution in {relation}: {template}")

        entry = {"queryName": "-", "template": template, "scope": scopes.get(relation)}
        call = _QUERY_CALL.search(source)
        if call is not None:
            entry["queryName"] = call.group(1)
        else:
            name = _RELATED_LIST_NAME.search(template)
            if name is not None:
                entry["relatedList"] = name.group(1)
        links[relation] = {
            k: entry[k]
            for k in ("queryName", "relatedList", "template", "scope")
            if k in entry
        }
    return links


# --------------------------------------------------------------------------- #
# lux-marklogic: search term configuration
# --------------------------------------------------------------------------- #

# facetToScopeAndTermName, from utils/searchTermUtils.mjs.
_FACET_NAME = re.compile(r"([a-z]+)([^a-z])(.*)")


def facet_to_scope_and_term(facet_name, facet_config):
    match = _FACET_NAME.match(facet_name)
    if match is None:
        raise ValueError(f"cannot derive scope and term from facet: {facet_name}")
    scope = match.group(1)
    term = facet_config.get("searchTermName") or (
        match.group(2).lower() + match.group(3)
    )
    # Grace for searchTermName values that should end with 'Id'.
    if facet_name.endswith("Id") and not term.endswith("Id"):
        term += "Id"
    return scope, term


def derive_facets(facets_config):
    """facet name -> {searchTermName}.

    Same rule the MarkLogic build uses, minus the trailing ``Id``: QLever
    facets on the parent (relationship) term, not the ``...Id`` child term.
    """
    facets = {}
    for facet_name, config in facets_config.items():
        _, term = facet_to_scope_and_term(facet_name, config)
        if term.endswith("Id"):
            term = term[:-2]
        facets[facet_name] = {"searchTermName": term}
    return dict(sorted(facets.items()))


def _has_hop_inverse_info(term):
    return bool(
        term.get("hopInverseName")
        and term.get("targetScope")
        and term.get("predicates")
        and term.get("generated") is not True
    )


def build_full_search_terms(search_terms_config, facets_config):
    """Port of runDuringDeployment/generateRemainingSearchTerms.mjs for the
    unrestricted tenant (no per-unit dropping, which needs MarkLogic security).

    The checked-in searchTermsConfig.mjs is only the hand-written half; the
    inverse, facet and record-type terms are added at deployment time, and the
    related-lists generator needs all of them.
    """
    config = copy.deepcopy(search_terms_config)

    # Generate facet search terms.
    for facet_name, facet in facets_config.items():
        scope, term = facet_to_scope_and_term(facet_name, facet)
        index_reference = facet["indexReference"]
        is_id_term = term.endswith("Id")
        is_date = index_reference.endswith("DateLong")
        is_dimension = index_reference.endswith("DimensionValue")
        is_zero_or_one = term in ("hasDigitalImage", "isOnline")

        references = [index_reference]
        if is_date:
            references.append(index_reference.replace("Start", "End"))

        if is_date:
            scalar_type = "dateTime"
        elif is_dimension:
            scalar_type = "float"
        elif is_zero_or_one:
            scalar_type = "integer"
        else:
            scalar_type = "string"

        scope_terms = config.setdefault(scope, {})
        if is_id_term:
            parent = term[:-2]
            if parent not in scope_terms:
                scope_terms[parent] = {"idIndexReferences": references}
            else:
                existing = scope_terms[parent].get("idIndexReferences")
                if existing is None:
                    scope_terms[parent]["idIndexReferences"] = references
                else:
                    extra = [r for r in references if r not in existing]
                    if extra:
                        scope_terms[parent]["idIndexReferences"] = existing + extra
        else:
            scope_terms[term] = {
                "patternName": PATTERN_DATE_RANGE
                if is_date
                else PATTERN_INDEXED_RANGE
                if is_dimension
                else PATTERN_INDEXED_VALUE,
                "indexReferences": references,
                "scalarType": scalar_type,
                "generated": True,
            }

    # Generate the hop inverse and transitive search terms.  Iterating over a
    # snapshot of the keys mirrors Object.keys() in the upstream loop, so the
    # terms generated here are not themselves revisited.
    for scope in list(config):
        for term_name in list(config[scope]):
            term = config[scope][term_name]
            make_transitive = term.get("makeTransitive")
            if make_transitive:
                transitive = dict(term)
                transitive["transitive"] = True
                transitive.pop("makeTransitive", None)
                transitive["hopInverseName"] = transitive["hopInverseName"] + "+"
                config[scope][term_name + "+"] = transitive
            if _has_hop_inverse_info(term):
                inverse_scope = term["targetScope"]
                inverse_name = term["hopInverseName"]
                config.setdefault(inverse_scope, {})
                config[inverse_scope][inverse_name] = {
                    "patternName": PATTERN_HOP_INVERSE,
                    "predicates": term["predicates"],
                    "targetScope": scope,
                    "hopInverseName": term_name,
                    "generated": True,
                }
                if make_transitive:
                    transitive = dict(config[inverse_scope][inverse_name])
                    transitive["transitive"] = True
                    transitive.pop("makeTransitive", None)
                    transitive["hopInverseName"] = transitive["hopInverseName"] + "+"
                    config[inverse_scope][inverse_name + "+"] = transitive

    # A recordType and an iri term on every scope.
    for scope in list(config):
        config[scope]["recordType"] = {
            "patternName": PATTERN_INDEXED_VALUE,
            "indexReferences": ["anyDataTypeName"],
            "scalarType": "string",
            "forceExactMatch": True,
            "generated": True,
        }
        config[scope]["iri"] = {"patternName": PATTERN_IRI, "generated": True}

    return {scope: dict(sorted(config[scope].items())) for scope in sorted(config)}


def derive_terms_inverse(search_terms_config):
    """scope -> term -> inverse term, both directions of every declared
    ``hopInverseName``.  Only the hand-written half of the config declares
    them; the deployment generator just materialises the other side."""
    inverses = {}
    for scope, terms in search_terms_config.items():
        for term_name, term in terms.items():
            inverse_name = term.get("hopInverseName")
            target_scope = term.get("targetScope")
            if inverse_name and target_scope:
                inverses.setdefault(scope, {})[term_name] = inverse_name
                inverses.setdefault(target_scope, {})[inverse_name] = term_name
    return {s: dict(sorted(inverses[s].items())) for s in sorted(inverses)}


class RelatedListBuilder:
    """Port of runDuringDeployment/generateRelatedListsConfig.mjs, projected to
    the shape qleverlux wants: scope -> related list -> {relationKey: scope}.

    ``_scoped_patterns`` is populated as the walk proceeds and read back by
    ``_is_hop_with_field``; upstream it is a module-level accumulator shared by
    every related list, so it is kept on the instance rather than per-walk.
    """

    def __init__(self, search_terms_config):
        self.config = search_terms_config
        self._scoped_patterns = {}

    def _inverse_search_term_info(self, scope, term_name):
        term = self.config.get(scope, {}).get(term_name)
        if not term:
            return None
        target_scope = term.get("targetScope")
        inverse_name = term.get("hopInverseName")
        if not (target_scope and inverse_name):
            return None
        inverse = self.config.get(target_scope, {}).get(inverse_name)
        if inverse is None:
            return None
        return {"scopeName": target_scope, "patternName": inverse.get("patternName")}

    def _inverse_is_disqualified(self, scope, term_name):
        info = self._inverse_search_term_info(scope, term_name)
        return bool(
            info
            and info["scopeName"] in ("agent", "concept", "place")
            and info["patternName"] == PATTERN_HOP_WITH_FIELD
        )

    def _entries(self, start, end, in_between, seen, max_level, level):
        found = []
        for term_name in list(self.config[start]):
            term = self.config[start][term_name]
            target = term.get("targetScope")
            matches_end = target == end
            # Skip scopes already seen, to avoid circular paths.
            if target in seen or not (matches_end or target in in_between):
                continue
            if (
                term.get("patternName") == PATTERN_RELATED_LIST
                or self._inverse_is_disqualified(start, term_name)
                or (level == 1 and matches_end)
            ):
                continue
            self._scoped_patterns[f"{start}.{term_name}"] = term.get("patternName")
            if matches_end:
                found.append((start, target, term_name, None))
            else:
                # No in-between scopes past the max level: those paths dilute
                # the entity pages more than they help.
                sub_between = in_between if level < max_level else []
                for sub in self._entries(
                    target, end, sub_between, seen + [target], max_level, level + 1
                ):
                    found.append((start, target, term_name, sub))
        return found

    @staticmethod
    def _relation_key(entry):
        parts = []
        while entry is not None:
            parts.append(entry[2])
            entry = entry[3]
        return "-".join(parts)

    def build(self):
        related_terms = []
        for scope, terms in self.config.items():
            for term_name, term in terms.items():
                if term_name.startswith("relatedTo"):
                    related_terms.append(
                        (
                            scope,
                            term_name,
                            term.get("targetScope"),
                            term.get("inBetweenScopes") or [],
                            term.get("maxLevel"),
                        )
                    )

        result = {}
        for from_scope, term_name, to_scope, in_between, max_level in related_terms:
            entries = self._entries(from_scope, to_scope, in_between, [], max_level, 1)
            if not entries:
                continue
            relations = {}
            for entry in entries:
                key = self._relation_key(entry)
                if key in SUPPRESSED_RELATION_KEYS:
                    continue
                relations[key] = entry[1]  # targetScope of the first hop
            result.setdefault(from_scope, {})[term_name] = dict(
                sorted(relations.items())
            )
        return {
            scope: dict(sorted(result[scope].items())) for scope in sorted(result)
        }


def load_scope_fields():
    """`SparqlTranslator.scope_fields`: the QLever predicate vocabulary.

    A related list relation is only usable if every hop in it has a `lux:`
    predicate here, so this is what decides which generated relations survive.
    Read straight out of the source with `ast` rather than by instantiating the
    translator, which would need a whole MTConfig.  Returns (path, fields), or
    (None, None) if no copy of sparql.py yields the literal.

    An instance directory holds only config/ and queries/, so the vocabulary
    comes from the code serving it: whichever qleverlux is installed, else the
    checkout this script lives in.
    """
    import ast

    for path in sparql_candidates():
        try:
            with open(path) as fh:
                tree = ast.parse(fh.read(), filename=path)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "scope_fields"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    try:
                        return path, ast.literal_eval(node.value)
                    except ValueError:
                        break
    return None, None


def sparql_candidates():
    """Places qleverlux/sparql.py might be, best first."""
    paths = []
    try:
        import qleverlux

        paths.append(os.path.join(os.path.dirname(qleverlux.__file__), "sparql.py"))
    except ImportError:
        pass
    paths.append(os.path.join(CODE_REPO, "qleverlux", "sparql.py"))
    return [p for i, p in enumerate(paths) if p not in paths[:i]]


def validate_related_list_scopes(related, search_terms, scope_fields):
    """Drop relation keys whose hops have no QLever predicate.

    The generators upstream describe MarkLogic's full search vocabulary, which
    runs ahead of what sparql.py knows how to translate; keeping an unmapped
    relation makes `MTConfig.make_related_query_stub` raise at startup.
    """
    if not scope_fields:
        return related, []

    kept = {}
    dropped = []
    for scope, lists in related.items():
        for list_name, relations in lists.items():
            for key, relation_scope in relations.items():
                hops = key.split("-")
                current = scope
                missing = None
                for i, hop in enumerate(hops):
                    # The last hop is resolved against the scope the relation
                    # returns, matching make_related_query_stub.
                    lookup = relation_scope if i == len(hops) - 1 else current
                    if hop not in scope_fields.get(lookup, {}):
                        missing = f"{lookup}.{hop}"
                        break
                    target = search_terms.get(current, {}).get(hop, {})
                    current = target.get("targetScope", current)
                if missing is None:
                    kept.setdefault(scope, {}).setdefault(list_name, {})[key] = (
                        relation_scope
                    )
                else:
                    dropped.append(f"{scope}.{list_name} {key} (no predicate for {missing})")
    return kept, dropped


def derive_related_lists(relation_names):
    """relation key -> user-facing label."""
    return dict(sorted(relation_names.items()))


def audit_sorts(sort_bindings, current_sorts, scopes):
    """sorts.json maps LUX sort keys onto QLever `lux:` predicate paths, which
    have no upstream counterpart, so it can only be audited.  Sort keys are
    scoped by their name prefix; `anySortName` and the unprefixed keys apply to
    every scope."""
    upstream = {scope: set() for scope in scopes}
    unscoped = set()
    for key in sort_bindings:
        for scope in scopes:
            if key.startswith(scope) and len(key) > len(scope):
                upstream[scope].add(key)
                break
        else:
            unscoped.add(key)

    report = []
    for scope in sorted(scopes):
        have = set(current_sorts.get(scope, {}))
        want = upstream[scope] | (unscoped & have)
        missing = sorted(want - have)
        extra = sorted(have - want - unscoped)
        if missing:
            report.append(f"  {scope}: upstream sort keys with no predicate: {missing}")
        if extra:
            report.append(f"  {scope}: local sort keys not upstream: {extra}")
    if unscoped - set().union(*[set(v) for v in current_sorts.values()] or [set()]):
        report.append(
            f"  (unscoped upstream keys, mapped per-scope as needed: "
            f"{sorted(unscoped)})"
        )
    return report


def derive_stopwords(path_or_url):
    """The deployed advanced search config's stopWords, de-duplicated."""
    if path_or_url.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(path_or_url) as response:
            config = json.load(response)
    else:
        with open(path_or_url) as fh:
            config = json.load(fh)
    return sorted(set(config["stopWords"]))


def default_stopwords_source():
    """luxql ships the deployed config; fall back to the live endpoint."""
    try:
        import luxql

        candidate = os.path.join(
            os.path.dirname(luxql.__file__), "advanced-search-config.json"
        )
        if os.path.exists(candidate):
            return candidate
    except ImportError:
        pass
    return "https://lux.collections.yale.edu/api/advanced-search-config"


# --------------------------------------------------------------------------- #
# Overrides, diffing and writing
# --------------------------------------------------------------------------- #

OVERRIDES_NAME = "derive_overrides.json"


def load_overrides(explicit, config_path):
    """Per-instance corrections, falling back to the ones in this checkout.

    An instance that wants none of the checkout's overrides can shadow them
    with an empty `derive_overrides.json` of its own.
    """
    if explicit:
        candidates = [explicit]
    else:
        candidates = [os.path.join(config_path, OVERRIDES_NAME)]
        for repo in (CODE_REPO, installed_repo()):
            if repo:
                candidates.append(os.path.join(repo, "config", OVERRIDES_NAME))
    for path in candidates:
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh), path
    if explicit:
        sys.exit(f"error: no overrides file at {explicit}")
    return {}, None


def deep_merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(name, value, overrides):
    """Hand-maintained corrections that must survive regeneration.

    Each entry is ``{"patch": {...}, "delete": ["dotted.path", ...]}``.
    """
    rule = overrides.get(name)
    if not rule or not isinstance(value, dict):
        return value
    for path in rule.get("delete", []):
        parts = path.split(".")
        target = value
        for part in parts[:-1]:
            target = target.get(part)
            if not isinstance(target, dict):
                target = None
                break
        if isinstance(target, dict):
            target.pop(parts[-1], None)
    if rule.get("patch"):
        deep_merge(value, copy.deepcopy(rule["patch"]))
    return value


def dumps(value):
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def read_json(path):
    """Parsed contents of path, or None if it is absent or unparseable."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def report_diff(label, path, new_value, new_text):
    """Summarise the change to one file.  Returns True if it differs."""
    if not os.path.exists(path):
        print(f"{label}: NEW ({len(new_text)} bytes)")
        return True
    with open(path) as fh:
        old_text = fh.read()
    old = read_json(path)
    if old == new_value:
        if old_text == new_text:
            print(f"{label}: unchanged")
        else:
            print(f"{label}: unchanged (reformatted only)")
        return old_text != new_text

    print(f"{label}: CHANGED")
    new = new_value
    if isinstance(old, dict) and isinstance(new, dict):
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
        for name, keys in (("added", added), ("removed", removed), ("changed", changed)):
            if keys:
                shown = ", ".join(keys[:12])
                more = f" (+{len(keys) - 12} more)" if len(keys) > 12 else ""
                print(f"    {name} ({len(keys)}): {shown}{more}")
    elif isinstance(old, list) and isinstance(new, list):
        print(f"    {len(old)} -> {len(new)} entries")
    return True


def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

TARGETS = (
    "queries",
    "hal_links",
    "facets",
    "related_lists",
    "related_list_scopes",
    "terms_inverse",
    "stopwords",
    "sorts",
)


def default_base_dir():
    """The middle tier to update when none is named.

    Handles both layouts this script gets run from: inside a checkout's
    ``files/``, and dropped straight into an instance directory.
    """
    for candidate in (CODE_REPO, HERE):
        if os.path.isdir(os.path.join(candidate, "config")) and os.path.isdir(
            os.path.join(candidate, "queries")
        ):
            return candidate
    return CODE_REPO


def installed_repo():
    """The checkout the installed qleverlux lives in, if it is an editable one.

    Worth checking when this script has been copied next to an instance: the
    upstream checkouts are far more likely to sit beside that checkout than
    beside the instance.
    """
    try:
        import qleverlux
    except ImportError:
        return None
    return os.path.dirname(os.path.dirname(os.path.abspath(qleverlux.__file__)))


def find_upstream(name, base_dir):
    """Locate an upstream checkout beside this one, the instance, or the cwd."""
    installed = installed_repo()
    for parent in (
        os.path.dirname(CODE_REPO),
        os.path.dirname(installed) if installed else None,
        os.path.dirname(base_dir),
        os.getcwd(),
        os.path.dirname(os.getcwd()),
    ):
        if parent is None:
            continue
        candidate = os.path.join(parent, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(os.path.dirname(CODE_REPO), name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=os.environ.get("QLMT_BASE_DIR") or default_base_dir(),
        help="middle tier instance to update: the directory holding config/ and "
        "queries/ (default: $QLMT_BASE_DIR, else this checkout)",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="override the config directory (default: <base-dir>/config)",
    )
    parser.add_argument(
        "--queries-path",
        default=None,
        help="override the queries directory (default: <base-dir>/queries)",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help=f"hand corrections to re-apply (default: <config-path>/"
        f"{OVERRIDES_NAME}, else this checkout's copy)",
    )
    parser.add_argument(
        "--lux-middletier",
        default=None,
        help="checkout of the Node.js middle tier "
        "(default: lux-middletier beside this checkout or the base dir)",
    )
    parser.add_argument(
        "--lux-marklogic",
        default=None,
        help="checkout of the MarkLogic backend "
        "(default: lux-marklogic beside this checkout or the base dir)",
    )
    parser.add_argument(
        "--stopwords-source",
        default=None,
        help="advanced-search-config JSON path or URL "
        "(default: the copy shipped with luxql, else the live endpoint)",
    )
    parser.add_argument(
        "--only",
        default=",".join(TARGETS),
        help=f"comma separated subset of: {', '.join(TARGETS)}",
    )
    parser.add_argument("--write", action="store_true", help="write the files")
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="keep related list relations even when sparql.py has no predicate "
        "for one of their hops (the middle tier will fail to start)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="with --write, delete queries/*.json no longer defined upstream",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if anything is out of date (implies dry run)",
    )
    args = parser.parse_args()

    targets = [t.strip() for t in args.only.split(",") if t.strip()]
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        sys.exit(f"error: unknown target(s): {', '.join(unknown)}")
    if args.check:
        args.write = False

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    config_path = os.path.abspath(args.config_path or os.path.join(base_dir, "config"))
    queries_path = os.path.abspath(
        args.queries_path or os.path.join(base_dir, "queries")
    )
    if not os.path.isdir(base_dir):
        sys.exit(f"error: no such base directory: {base_dir}")
    # Refuse to scatter files into a directory that is not a middle tier, which
    # is what a mistyped --base-dir looks like.
    if not (os.path.isdir(config_path) or os.path.isdir(queries_path)):
        sys.exit(
            f"error: {base_dir} holds neither config/ nor queries/; "
            f"is it a middle tier instance?"
        )
    args.lux_middletier = args.lux_middletier or find_upstream(
        "lux-middletier", base_dir
    )
    args.lux_marklogic = args.lux_marklogic or find_upstream(
        "lux-marklogic", base_dir
    )

    print(f"target: {base_dir}")
    overrides, overrides_path = load_overrides(args.overrides, config_path)
    if overrides_path:
        print(f"overrides: {overrides_path}")

    node = NodeRunner()
    stale = False
    try:
        ml_config = os.path.join(
            args.lux_marklogic, "src/main/ml-modules/root/config"
        )
        needs_middletier = {"queries", "hal_links"} & set(targets)
        needs_marklogic = {
            "facets",
            "related_lists",
            "related_list_scopes",
            "terms_inverse",
            "sorts",
        } & set(targets)

        if needs_middletier and not os.path.isdir(args.lux_middletier):
            sys.exit(f"error: no lux-middletier checkout at {args.lux_middletier}")
        if needs_marklogic and not os.path.isdir(ml_config):
            sys.exit(f"error: no lux-marklogic config at {ml_config}")

        dump = None
        if needs_middletier:
            print(f"reading {args.lux_middletier}")
            dump = node.run("dump_middletier.mjs", os.path.abspath(args.lux_middletier))

        search_terms = facets_config = relation_names = sort_bindings = None
        if needs_marklogic:
            print(f"reading {ml_config}")
            if {"terms_inverse", "related_list_scopes"} & set(targets):
                search_terms = node.literal(
                    os.path.join(ml_config, "searchTermsConfig.mjs"),
                    "SEARCH_TERMS_CONFIG",
                )
            if {"facets", "related_list_scopes"} & set(targets):
                facets_config = node.literal(
                    os.path.join(ml_config, "facetsConfig.mjs"), "FACETS_CONFIG"
                )
            if "related_lists" in targets:
                relation_names = node.literal(
                    os.path.join(ml_config, "relationNames.mjs"), "RELATION_NAMES"
                )
            if "sorts" in targets:
                sort_bindings = node.literal(
                    os.path.join(ml_config, "searchResultsSortConfig.mjs"),
                    "SORT_BINDINGS",
                )
        print()

        # queries/*.json ---------------------------------------------------- #
        if "queries" in targets:
            queries = derive_queries(dump)
            queries_dir = queries_path
            existing = {
                f[:-5] for f in os.listdir(queries_dir) if f.endswith(".json")
            } if os.path.isdir(queries_dir) else set()

            changed = []
            reformatted = 0
            for name in sorted(queries):
                path = os.path.join(queries_dir, f"{name}.json")
                text = dumps(queries[name])
                if not os.path.exists(path):
                    changed.append(name)
                elif read_json(path) != queries[name]:
                    changed.append(name)
                else:
                    with open(path) as fh:
                        if fh.read() != text:
                            reformatted += 1
                if args.write:
                    write_file(path, text)
            obsolete = sorted(existing - set(queries))
            if changed or obsolete or reformatted:
                stale = True
            print(f"queries/ ({len(queries)} upstream): {len(changed)} new or changed")
            if changed:
                print(f"    {', '.join(changed)}")
            if reformatted:
                print(f"    {reformatted} identical but reformatted")
            if obsolete:
                print(f"    no longer upstream ({len(obsolete)}): {', '.join(obsolete)}")
                if args.write and args.prune:
                    for name in obsolete:
                        os.remove(os.path.join(queries_dir, f"{name}.json"))
                    print("    pruned")
                elif not args.prune:
                    print("    kept; pass --prune to delete")

        # config/*.json ------------------------------------------------------ #
        generated = {}
        if "hal_links" in targets:
            generated["hal_links.json"] = derive_hal_links(dump)
        if "facets" in targets:
            generated["facets.json"] = derive_facets(facets_config)
        if "related_lists" in targets:
            generated["related_lists.json"] = derive_related_lists(relation_names)
        if "terms_inverse" in targets:
            generated["terms_inverse.json"] = derive_terms_inverse(search_terms)
        if "related_list_scopes" in targets:
            full_terms = build_full_search_terms(search_terms, facets_config)
            related = RelatedListBuilder(full_terms).build()
            if args.validate:
                sparql_path, scope_fields = load_scope_fields()
                if scope_fields is None:
                    print(
                        "  ! no qleverlux/sparql.py found, skipping predicate "
                        "validation (pass --no-validate to silence)"
                    )
                else:
                    related, dropped = validate_related_list_scopes(
                        related, full_terms, scope_fields
                    )
                    if dropped:
                        print(
                            f"related_list_scopes: dropped {len(dropped)} relation(s) "
                            f"with no QLever predicate in {sparql_path}:"
                        )
                        for line in dropped:
                            print(f"    {line}")
            generated["related_list_scopes.json"] = related
        if "stopwords" in targets:
            source = args.stopwords_source or default_stopwords_source()
            print(f"stopwords source: {source}")
            generated["stopwords.json"] = derive_stopwords(source)

        for name in sorted(generated):
            value = apply_overrides(name, generated[name], overrides)
            path = os.path.join(config_path, name)
            text = dumps(value)
            if report_diff(f"config/{name}", path, value, text):
                stale = True
            if args.write:
                write_file(path, text)

        # sorts.json --------------------------------------------------------- #
        if "sorts" in targets:
            sorts_path = os.path.join(config_path, "sorts.json")
            with open(sorts_path) as fh:
                current_sorts = json.load(fh)
            report = audit_sorts(sort_bindings, current_sorts, list(current_sorts))
            if report:
                print("config/sorts.json: AUDIT (values are QLever predicates, "
                      "so this file is maintained by hand)")
                print("\n".join(report))
            else:
                print("config/sorts.json: in step with upstream sort keys")
    finally:
        node.cleanup()

    print()
    if args.write:
        print("written.")
    elif stale:
        print("dry run: nothing written (pass --write to apply)")
        if args.check:
            sys.exit(1)
    else:
        print("everything up to date")


if __name__ == "__main__":
    main()
