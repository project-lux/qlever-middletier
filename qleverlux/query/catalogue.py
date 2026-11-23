"""The config-driven query catalogue.

Everything under ``config/`` and ``queries/`` is loaded once at startup and
precompiled to SPARQL text, so serving a request is string substitution rather
than translation. This used to be the back half of ``MTConfig.__init__``; it is
a build step, not configuration, and it is the slow part of starting up.

What comes out:

``queries``
    ~90 named LUX JSON queries with ``URI-HERE`` placeholders.
``sparql_hal_queries``
    scope -> HAL relation -> count query text. Related-list relations map to
    the raw config entry (a dict) instead, and are resolved per request.
``hal_related_list_tests``
    scope -> related list -> ordered cheap existence probes.
``related_list_sparql`` / ``related_list_json``
    the per-request count query and the user-facing JSON query for each
    related list.
"""

from __future__ import annotations

import json
import os

import luxql.config as lconfig
from luxql import JsonReader, LuxConfig

from qleverlux.enums import MULTI_SCOPE
from qleverlux.query.related import RelatedListBuilder
from qleverlux.query.translator import SparqlTranslator

#: Search terms luxql does not ship but the middle tier relies on.
_EXTRA_TERMS = {
    "work": {
        "workCreationOrPublicationDate": {
            "label": "Creation or Publication Date",
            "relation": "date",
        }
    },
    "set": {
        "setCreationOrPublicationDate": {
            "label": "Creation or Publication Date",
            "relation": "date",
        }
    },
}

#: What each scope's records are called, for the similarRecords help text.
_SCOPE_NOUNS = {
    "item": "Objects",
    "work": "Works",
    "set": "Collections",
    "agent": "People & Groups",
    "place": "Places",
    "concept": "Concepts & Types",
    "event": "Events",
}

# similarRecords: records whose vectors are nearest to those of the given
# records, in every scope. A relationship to its own scope, so luxql reads
# {"similarRecords": {"id": <uri>}} (or an OR of ids) as records of that scope;
# the translator turns them into seeds for the vector SERVICE rather than a
# predicate. `similar` is the free-text counterpart.
for _scope, _noun in _SCOPE_NOUNS.items():
    _EXTRA_TERMS.setdefault(_scope, {})["similarRecords"] = {
        "label": "are similar to the records",
        "helpText": f"Search for {_noun} that are similar to the specified {_noun}.",
        "relation": _scope,
    }


#: The advanced search config file looked for in the config directory.
SEARCH_CONFIG_NAME = "advanced-search-config.json"


def resolve_search_config(search_config: str | None, config_path: str = "config") -> str | None:
    """Which advanced search config file to read, or None for luxql's own.

    1. ``QLMT_SEARCH_CONFIG``, when it is set. It must exist: luxql quietly
       falls back to the live LUX config for a path that does not, which is
       exactly the silent vocabulary change this setting is here to prevent.
    2. ``<config_path>/advanced-search-config.json``, when that exists.
    3. None: luxql's saved copy in its package directory, or, without one, a
       fresh copy fetched from the live LUX endpoint.
    """
    if search_config and search_config.strip().lower() not in ("null", "none"):
        path = os.path.abspath(os.path.expanduser(search_config.strip()))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"QLMT_SEARCH_CONFIG is {search_config!r}, but there is no such file ({path})"
            )
        return path
    candidate = os.path.join(config_path, SEARCH_CONFIG_NAME)
    return os.path.abspath(candidate) if os.path.isfile(candidate) else None


def build_lux_config(path: str | None = None):
    """A ``LuxConfig`` with the middle tier's extra search terms patched in.

    ``path`` is an advanced search config file (see `resolve_search_config`);
    None leaves the choice to luxql.

    The instance is also published back into luxql, because every query node
    takes its config from a module global rather than from the reader it was
    built by. ``luxql.query`` does ``from .config import _cached_lux_config`` at
    import time, so it holds its own reference: rebinding the name in
    ``luxql.config`` alone leaves query nodes on the unpatched original, and the
    extra terms resolve everywhere except where they are actually used. Rebind
    wherever the name was captured.
    """
    lux_config = LuxConfig(lux_config=path) if path else LuxConfig()
    for scope, terms in _EXTRA_TERMS.items():
        for name, defn in terms.items():
            lux_config.lux_config["terms"][scope][name] = defn
            scopes = lux_config.inverted.setdefault(name, [])  # a term may be in several
            if scope not in scopes:
                scopes.append(scope)

    lconfig._cached_lux_config = lux_config
    for module in _modules_holding_lux_config():
        module._cached_lux_config = lux_config
    return lux_config


def _modules_holding_lux_config():
    """luxql modules that imported ``_cached_lux_config`` by name."""
    import sys

    return [
        m
        for name, m in list(sys.modules.items())
        if name.startswith("luxql")
        and m is not None
        and getattr(m, "_cached_lux_config", None) is not None
    ]


class MultiScopeError(ValueError):
    """A ``_scope: "multi"`` query that does not follow the multi contract."""


def read_multi_branches(json_reader, query):
    """Read a multi-scope query into [(scope, parsed), ...].

    A multi query is an ``OR`` of subqueries, each naming its own ``_scope``.
    Every branch is a perfectly ordinary single-scope query, so each is read on
    its own rather than teaching the reader about ``multi``.
    """
    branches = query.get("OR")
    if not isinstance(branches, list) or not branches:
        raise MultiScopeError(
            "a search with scope 'multi' must contain a non-empty 'OR' array"
        )
    read = []
    for branch in branches:
        if not isinstance(branch, dict):
            raise MultiScopeError("every branch of a 'multi' search must be a query")
        scope = branch.get("_scope")
        if not scope:
            raise MultiScopeError(
                "every branch of a 'multi' search must name its own '_scope'"
            )
        if scope == MULTI_SCOPE:
            raise MultiScopeError("'multi' branches cannot themselves be 'multi'")
        read.append((scope, json_reader.read(dict(branch), scope)))
    return read


class QueryCatalogue:
    """Loads ``config/`` and ``queries/`` and precompiles the query catalogue."""

    def __init__(self, settings, lux_config=None):
        self.settings = settings
        if lux_config is None:
            path = resolve_search_config(settings.search_config, settings.config_path)
            print(f"Search config:  {path or 'luxql default (its saved copy, else the live LUX config)'}")
            lux_config = build_lux_config(path)
        self.lux_config = lux_config

        self._load_files()
        self._trim_facets()

        self.json_reader = JsonReader(self.lux_config)
        self.translator = SparqlTranslator(
            self.lux_config,
            stopwords=self.stopwords if settings.use_stopwords else None,
        )
        if settings.portal:
            self.translator.portal = settings.portal
        self.translator.configure_scoring(
            text_scale=settings.text_score_scale,
            vector_scale=settings.vector_score_scale,
        )
        self.translator.configure_paths(max_hops=settings.max_hops)
        self.translator.configure_vectors(
            service=settings.vector_service,
            top_k=settings.vector_top_k,
            nested_top_k=settings.vector_nested_top_k,
            min_score=settings.vector_min_score,
            max_gap=settings.vector_max_gap,
            nested_min_score=settings.vector_nested_min_score,
            nested_max_gap=settings.vector_nested_max_gap,
            constrain_in_and=settings.vector_constrain_in_and,
        )

        self.sparql_hal_queries = {}
        self.related_list_sparql = {}
        self.related_list_json = {}
        self.hal_related_list_tests = {}
        self.compile()

    # -- loading ---------------------------------------------------------

    def _config_file(self, name):
        return os.path.join(self.settings.config_path, name)

    def _load_json(self, name):
        with open(self._config_file(name)) as fh:
            return json.load(fh)

    def _load_text(self, name):
        with open(self._config_file(name)) as fh:
            return fh.read().strip()

    def _load_files(self):
        settings = self.settings

        self.facets = self._load_json("facets.json")
        self.sorts = self._load_json("sorts.json")
        self.related_list_names = self._load_json("related_lists.json")
        self.related_list_scopes = self._load_json("related_list_scopes.json")
        self.inverses = self._load_json("terms_inverse.json")
        self.stopwords = self._load_json("stopwords.json")

        self.hal_queries = self._load_json("hal_links.json")
        for v in self.hal_queries.values():
            v["template"] = v["template"].replace(
                "{searchUriHost}", settings.mt_uri[:-1]
            )

        self.queries = {}
        for q in os.listdir(settings.queries_path):
            if q.endswith(".json"):
                with open(os.path.join(settings.queries_path, q)) as f:
                    self.queries[q[:-5]] = json.load(f)

        self.system_prompt_translate = self._load_text("system-prompt-translate.txt")
        self.system_prompt_improve = self._load_text("system-prompt-improve.txt")

    def _trim_facets(self):
        """Drop facets whose search term luxql does not know about."""
        terms = self.lux_config.lux_config["terms"]
        for k, v in list(self.facets.items()):
            stn = v["searchTermName"]
            if not any(stn in scope_terms for scope_terms in terms.values()):
                print(
                    f" *** Could not find search term {stn} for facet {k} "
                    "in search terms, deleting ***"
                )
                del self.facets[k]

    # -- compilation -----------------------------------------------------

    def compile(self):
        self._compile_hal_queries()
        self._compile_related_lists()

    def _compile_hal_queries(self):
        """One count query per HAL relation; a non-zero count emits the link."""
        for entry in self.hal_queries.values():
            self.sparql_hal_queries.setdefault(entry["scope"], {})
            self.hal_related_list_tests.setdefault(entry["scope"], {})

        for hal, entry in self.hal_queries.items():
            scope = entry["scope"]
            qname = entry["queryName"]
            if qname == "-" and "related-list" in entry["template"]:
                # related lists are generated below, not from queries/
                self.sparql_hal_queries[scope][hal] = entry
                continue

            query = self.queries.get(qname, {})
            if not query:
                print(f"Could not find query {qname} referenced from HAL {hal}")
                continue
            try:
                qscope = query["_scope"]
            except KeyError:
                # Already been processed
                print(f"Couldn't find scope for {hal} / {qname}?")
                continue
            try:
                if qscope == MULTI_SCOPE:
                    branches = read_multi_branches(self.json_reader, query)
                    spq = self.translator.translate_multi_search_count(branches)
                else:
                    parsed = self.json_reader.read(query, qscope)
                    spq = self.translator.translate_search_count(parsed, qscope)
            except Exception as e:
                print(f"Error parsing query for {hal}: {e}\n{query}")
                continue
            self.sparql_hal_queries[scope][hal] = spq.get_text()

    def _compile_related_lists(self):
        builder = RelatedListBuilder(
            self.lux_config, self.inverses, self.related_list_scopes
        )
        sparql, as_json, probes = builder.build_all()
        self.related_list_sparql = sparql
        self.related_list_json = as_json
        for scope, by_type in probes.items():
            self.hal_related_list_tests.setdefault(scope, {}).update(by_type)
