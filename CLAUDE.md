# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`qleverlux` is the LUX middle tier reimplemented on top of a [QLever](https://github.com/ad-freiburg/qlever) SPARQL
endpoint instead of MarkLogic. It serves the LUX public API (search, facets, related lists, record retrieval, HAL
links) by translating LUX JSON queries into SPARQL, executing them against QLever, and returning Linked Art /
ActivityStreams JSON.

Record JSON itself is *not* stored in QLever — it comes from a PostgreSQL and/or LMDB cache of the pre-built
documents. QLever answers "which URIs match?"; the caches answer "what does this URI look like?".

## Running

```bash
pip install -e .                        # plus luxql: git+https://github.com/project-lux/luxql.git
pip install -r requirements.txt

python uvicorn-serve.py                 # dev: HTTP on :5001, single process, calls mt.start()
python serve.py                         # hypercorn HTTPS/2 on QLMT_MTPORT (needs files/<cert_name>.pem + -key.pem)
python workers.py                       # hypercorn multi-process; module is re-imported per worker
python tester.py                        # REPL: type a query string or JSON, prints the generated SPARQL
```

There are no tests, linters, or CI configured. `tester.py` is the fastest feedback loop for query-translation
changes — it exercises the parser → JsonReader → SparqlTranslator chain without needing QLever running.

Configuration is env-var first (`.env` via `dotenv`, all keys prefixed `QLMT_`), overridden by CLI flags parsed in
`MTConfig.__init__`. Passing `--pgpass`/`--qlpass` deliberately aborts the process; use `.env`. `parse_known_args`
is used, so unknown flags (e.g. uvicorn's) are tolerated and stashed in `config.remaining_args`.

QLever setup (indexing, materialized views, the UI) is described in `README.md` and `installing_qlever.md`; the
canonical index/server settings live in `files/Qleverfile`.

## Architecture

### Request path

`qleverlux/middletier.py` holds both the FastAPI routes and the `QLeverLuxMiddleTier` class. The routes do not
instantiate anything — they delegate to a module-level global `mt` that each entry point sets *after* constructing
it (`qleverlux.middletier.mt = mt`), read through `local_module = sys.modules[__name__]`. If you add a route,
follow that pattern; importing `mt` directly at module scope will get `None`.

`mt.start()` must run inside the event loop (it opens the httpx/aiohttp pool and the async psycopg connection).
`serve.py` and `workers.py` do this piecemeal rather than calling `start()`.

Every SPARQL round-trip goes through `fetch_qlever_sparql`, which dispatches to an httpx (HTTP/2, default) or
aiohttp implementation. Both are wrapped in `@alru_cache(maxsize=500)` keyed on query text, and both bail out with
a 504-ish stub when `open_requests > max_qlever_requests` unless called with `drop_okay=False` (HAL generation
uses `drop_okay=False` so it can't be shed). Results come back in QLever's own
`application/qlever-results+json` and are unwrapped by `process_qlever_results` into
`{results, total, time, variables}` with `<uri>` brackets stripped and typed literals coerced.

### Query translation

Three layers, outermost first:

1. **luxql** (external package) — `QueryParser` turns a simple string query into an AST; `JsonReader` validates a
   LUX JSON query against `LuxConfig` and returns a tree of `LuxBoolean` / `LuxRelationship` / `LuxLeaf`.
2. `qleverlux/sparql.py` — `SparqlTranslator` walks that tree and emits a query object. Entry points:
   `translate_search`, `translate_search_count` (wraps in `COUNT(*)`, used for HAL estimates),
   `translate_facet`, `translate_facet_count`, `translate_search_related`.
3. `qleverlux/SPARQLQueryBuilder.py` + `SPARQLSyntaxTerms.py` — a vendored/modified SPARQL Burger. Objects
   (`SelectQuery`, `GraphPattern`/`Pattern`, `Triple`, `Filter`, `Binding`, `BNode`, `Values`, …) render via
   `get_text()`. `BNode` and the `service=` argument on `GraphPattern` exist for QLever's `SERVICE` extensions.

The predicate vocabulary lives in `SparqlTranslator.scope_fields` (relationships) and `scope_leaf_fields` (leaf
values), keyed by scope then LUX search-term name, mapping to `lux:` predicate names. A leading `^` means inverse
(`^lux:foo`). `get_predicate` / `get_leaf_predicate` special-case classification, memberOf, dimensions, and the
date families (which expand to `startOf…`/`endOf…` pairs). "missed" is the sentinel for an unmapped term.

Scopes are `item work agent place concept set event`; record classes (URL path segments) are the finer set in
`classEnum` (`object digital text visual person group period activity` …) and get folded back to scopes in
`do_hal_links`.

### Text search and relevance

Free-text (`text` field) searches go through QLever **materialized views** named `<scope>Words`, one `SERVICE
view:<scope>Words { [ view:column-word "w" ; view:column-uri ?uri ; view:column-score ?tf ] }` block per word.
Those views are defined in `files/Qleverfile` under `MATERIALIZED_VIEWS` and weight primary name 14 / record text
5 / any-name 1 — the weights the relevance ordering depends on. Name-field searches instead use `ql:has-word`
inside a `GRAPH ?tf` pattern against `lux:<scope>Name`. Quoted phrases become `CONTAINS()` filters over a UNION of
the three text sources. Per-word `?tf_*` scores are summed into `?score_N`, then into `?sscore` for
`ORDER BY DESC`. Stopwords (`config/stopwords.json`) are stripped; an all-stopword query raises `ValueError`,
surfaced as a 400.

Pagination is deliberately coarse: SPARQL is issued at offsets rounded down to multiples of 60 (so the alru cache
hits across nearby pages) and the exact page is sliced out of the result list in Python.

### Config-driven query catalogue (`config/` + `queries/`)

`MTConfig` loads these at startup and `cache_sparql_queries()` precompiles everything to SPARQL text once, so
runtime work is string substitution:

- `queries/*.json` — ~90 named LUX JSON queries with `URI-HERE` placeholders. Generated from the LUX frontend's
  `builder.js` by `files/translate_query.py`.
- `config/hal_links.json` — HAL relation → `{queryName, template, scope}`. Each becomes a count query;
  a non-zero count means the link is emitted on the record.
- `config/related_list_scopes.json` — for each scope/related-list, `from-to` field-path → result scope.
  `make_related_query_stub` turns each into an `OPTIONAL { SELECT … COUNT }` fragment plus a cheap
  `LIMIT 1` HAL existence test; both are ordered by a hand-tuned heuristic score (classification and
  `workAbout` boost, multi-hop paths and Set/Event penalise) so HAL checks can short-circuit on the first hit.
  `make_related_json_stub` builds the equivalent user-facing JSON query using `config/terms_inverse.json`.
- `config/facets.json` — facet name → `searchTermName`; entries whose term isn't in `LuxConfig` are dropped at
  startup with a warning. `config/sorts.json` — sort key → predicate, per scope.

Placeholders used across the templates: `URI-HERE`, `V_TARGET_URI`, `V_FROM_URI`, `V_TO_URI`,
`V_NAME_REL`/`V_URI_REL`/`V_TARGET_REL`, `{q}`, `{id}`, `{searchUriHost}`.

### Caching layers

- Record JSON: PostgreSQL `lux_data_cache` (`QLMT_USE_PG_DATA_CACHE`) and/or LMDB (`QLMT_LMDB_PATH`, zlib-
  compressed values keyed by raw UUID bytes). Loaded by `files/load-json-to-postgres.py`.
- HAL links: disk (`hal_cache/*.json`, default) or PostgreSQL `hal_data_cache`. Computing them is expensive —
  one query per candidate relation — so a cache miss is a slow request.
- SPARQL responses: in-process `alru_cache`.

### URI rewriting

Data uses `QLMT_DATAURI` (`https://lux.collections.yale.edu/`); responses must use the deployment's own
`mt_uri`, assembled from `replace_proto`/`replace_host`/`replace_port`/`replace_path`. Inbound queries are
rewritten mt→data before translation, outbound records data→mt (currently via a JSON round-trip through
`json.dumps`/`loads`). Any new endpoint needs both directions.

`QLMT_PORTAL` (YPM, YCBA, YUAG, PMC, IPCH) makes the translator inject `?uri lux:source lux:<portal>` into every
pattern, turning the instance into a single-unit portal.

## Notes

- `qleverlux/sparql-orig.py` is an untracked pre-materialized-view snapshot of `sparql.py`, kept for reference.
  Do not edit it or import from it.
- `MTConfig` attributes are overwritten wholesale by `vars(args)`, so the argparse `dest` name wins: the env-var
  fields `mt_queue_size`/`mt_workers`/… are shadowed by `queue_size`/`workers`/…, which is what `serve.py` and
  `print_config` read.
