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

python uvicorn-serve.py                 # dev: HTTP on QLMT_MTPORT, single process
python serve.py                         # hypercorn HTTPS/2 on QLMT_MTPORT (needs files/<cert_name>.pem + -key.pem)
python workers.py                       # hypercorn multi-process
python tester.py [scope]                # REPL: type a query string or JSON, prints the generated SPARQL
python -m pytest tests/ -q              # SPARQL builder conformance tests (W3C + SPARQL 1.1 suites)
```

`requirements.txt` sets minimum versions for dependencies, mostly indirect ones, with published
vulnerabilities below them. They come from `pip-audit` over what the middle tier installs. Run it again on a
deployed environment, and raise or add a minimum when a new advisory names something:
`pip freeze | grep -v -E '^-e|@ ' > /tmp/req.txt && pip-audit --no-deps --disable-pip -r /tmp/req.txt`.

Vector similarity has two more commands, run from this root with their own venv (`requirements-vectors.txt`;
the middle tier never imports them, enforced by `tests/test_import_boundaries.py`):

```bash
PY=../vectors/.venv/bin/python
$PY -m lux_embed {labels,kinds,facets,cards,embed,load,index,query} ...   # build the vector store offline; lux_embed/README.md
HF_HUB_OFFLINE=1 $PY -m sparql_service --store ../vectors/data/mds/vectors.lmdb --port 8081   # the SERVICE sidecar
$PY -m pytest tests/test_lux_embed.py tests/test_facet_dump.py tests/test_labels.py tests/test_sparql_service.py tests/test_lmdb_vector_store.py -q
```

All three servers are three-line shims over `qleverlux/server.py` (`serve_dev` / `serve_https` /
`run_workers`), which is also runnable as `python -m qleverlux.server [--dev|--workers]`. Connection pools are
opened by the app's lifespan handler, so nothing has to remember to do it.

The tests cover `qleverlux/sparql/` (the query object model), the escaping helpers, the import boundary and the
vector code; the request path and translator have none, so treat the compiled output as the contract when changing it — see "Verifying a change" below. `tester.py` is the fastest
feedback loop for query-translation changes: it exercises the parser → JsonReader → SparqlTranslator chain
without needing QLever running.

Configuration is env-var first (all keys prefixed `QLMT_`), overridden by CLI flags. A `.env` in the directory
the process starts in is read by pydantic-settings, and its `QLMT_` keys win over the real environment. It is
never loaded into `os.environ`: it once was, with `override=True`, so any variable in it (`HTTPS_PROXY`,
`SSL_CERT_FILE`, ...) replaced the real one for the whole process. The package is `python-dotenv`; `dotenv` on
PyPI is a different, deprecated project. Each
setting is declared exactly once, as a field on `Settings` in `qleverlux/settings.py`; its env var is `QLMT_` +
the field name unless the field carries an explicit alias, and its flag is the field name with hyphens unless
it is listed in `NO_CLI`. Booleans get `--flag` / `--no-flag`. Passing `--pgpass` deliberately aborts the
process; use `.env`. `parse_known_args` is used, so unknown flags (e.g. uvicorn's) are tolerated and stashed in
`settings.remaining_args`.

QLever setup (indexing, materialized views, the UI) is described in `README.md` and `installing_qlever.md`; the
canonical index/server settings live in `files/Qleverfile`. Its `ACCESS_TOKEN` is read from `QLEVER_ACCESS_TOKEN`
in the environment (`$$(printenv ...)`), so every `qlever` command needs that set: the repository is public and
the token authorises SPARQL UPDATE. Never write one into the file.

## Architecture

### Package layout

```
qleverlux/
  settings.py        Settings: env + CLI. Pure data, no I/O beyond reading .env
  enums.py models.py path/query enums, pydantic response models
  errors.py          QLeverError / BadQuery + the handlers that render them
  app.py             MiddleTier (the object graph) + create_app + module-level `app`
  server.py          serve_dev / serve_https / run_workers
  api/               one module per group of routes; parameters in, service call out
  services/          what each endpoint does: search, facets, related, records, hal, stats, ai_translate, cms
  clients/           qlever, postgres, lmdb_store, record_cache, ai/ (one module per AI backend)
  query/             predicates (vocabulary as data), text, translator, related, catalogue
  sparql/            builder + terms: the SPARQL object model
  presentation/      uris (the only place that knows mt_uri ↔ data_uri), envelopes
  vectors/           lmdb_vector_store: the store lux_embed writes and sparql_service reads (not imported here)
lux_embed/           offline vector build: record LMDB + QLever facets -> cards -> embeddings -> HNSW graph
sparql_service/      the vector SERVICE QLever calls for `similar`; its README is the query contract
```

Dependencies point one way: `api` → `services` → (`query`, `clients`, `presentation`) → `sparql`. Nothing under
`query/` imports the middle tier, so a translator or a related-list query can be built on its own.

### Request path

`qleverlux/app.py` holds `MiddleTier`, which owns one instance of everything — settings, catalogue, clients,
services — and `create_app()`, which builds the FastAPI app. The lifespan handler constructs `MiddleTier`, puts
it on `app.state`, and opens/closes the pools, so construction happens inside the running event loop. Routes
reach it with `mt=Depends(get_middletier)` (`api/deps.py`). Importing `qleverlux.app` is cheap: the catalogue
(~90 query files, compiled to SPARQL) is only built when the server starts.

To add an endpoint: a route in `api/`, included from `api/__init__.py`, delegating to a method on a service.

Every SPARQL round-trip goes through `QLeverClient.query` (`clients/qlever.py`), which dispatches to an httpx
(HTTP/2, default) or aiohttp implementation. Both are wrapped in `@alru_cache(maxsize=500)` keyed on query text,
which caches successes only: a failure (timeout, QLever error, failed SERVICE) is raised out of the cached call
and turned back into an error dict outside it. Caching failures once made a single slow first attempt fail
that query instantly until evicted. Requests are shed with a 504-ish stub when
`open_requests > max_qlever_requests`, decided per call and outside the cache, unless called with
`sheddable=False`, which nothing uses now: HAL generation once did, so every uncached record pushed 6-25 queries
past the limit. `open_requests` is decremented in `finally`, because a cancelled request (async-lru cancels a
shared call once every caller has gone) raises `CancelledError`, which `except Exception` misses, and each miss
left the count one higher until everything was shed. Each query carries `timeout=<QLMT_QLEVER_TIMEOUT>s`, so
QLever stops when the client stops waiting instead of running on to its own `TIMEOUT`; QLever refuses a timeout
above its `TIMEOUT` (60 s in `files/Qleverfile`) without the access token, so `QLMT_QLEVER_TIMEOUT` must not
exceed it. Results are
unwrapped into `{results, total, time, variables}` with IRIs bare and int/decimal literals as numbers, from one of
two formats, `query(..., fmt=)` (also `run_query`; part of the cache key):

- `JSON` (default) asks for `application/sparql-results+json`; `total` and `time` come from QLever's `meta`
  (`result-size-total`, `query-time-ms`, needing its runtime setting `sparql-results-json-with-time`, on by
  default), and `send` is honoured. Facets and HAL counts read the total. It replaced QLever's own
  `application/qlever-results+json`, whose query-plan tree made related-list and HAL responses 24-30 times
  the size for the same rows; `process_qlever_results` is kept as the reference the evaluation checks against.
- `TSV` is for related lists only. On a QLever cache miss their queries (dozens of OPTIONAL count sub-selects)
  cost 360-900 ms as either JSON format and 17-65 ms as TSV, CSV or XML, with identical rows - the two JSON
  formats are the ones that report the total. TSV has no total and ignores `send`, which related lists (own
  LIMIT) do not need, and writes a tab inside a literal as a space, which their IRIs and counts cannot have.

Both are asked for uncompressed (`Accept-Encoding: identity`): both HTTP clients otherwise ask for gzip, and
QLever compressed every result for us to decompress, on what is almost always a local link. QLever answers an
error as JSON with an `exception` and a 400 in either format. `files/qlever_serialization_eval.py` compares
every serialization and encoding QLever offers on the middle tier's own queries, warm, on a cache miss
(`--cold`) and under concurrency. Connections are kept alive (httpx closes one idle for 5 s); `http2=True`
has no effect, since QLever speaks HTTP/1.1.

QLever's result cache is limited by entry count as well as size, and the count is what fills: at qlever's
default of 200 it was full holding 29 MB, while one search plus one uncached record's HAL links is over 400
distinct queries. `files/Qleverfile` sets `CACHE_MAX_NUM_ENTRIES = 10000`; on the same workload throughput
went from 42 to 750 queries a second on one client, 313 to 2,000 on eight.

Services call `run_query(client, sparql)` from `errors.py`, which raises `QLeverError` on failure; one exception
handler renders it, so handlers only deal with results that worked. A failure's detail (QLever's message, an
exception's text) goes to the log, and the client gets a plain message with the same status: QLever's message
can quote the query and internal addresses. `QLMT_DEBUG_ERRORS=true` adds the detail to the response as
`detail`, for development. Validation errors (`BadQuery`, 400) are the caller's own and are shown as they are.

### Query translation

Three layers, outermost first:

1. **luxql** (external package) — `QueryParser` turns a simple string query into an AST; `JsonReader` validates a
   LUX JSON query against `LuxConfig` and returns a tree of `LuxBoolean` / `LuxRelationship` / `LuxLeaf`.
2. `qleverlux/query/translator.py` — `SparqlTranslator` walks that tree and emits a query object. Entry
   points: `translate_search`, `translate_search_count` (wraps in `COUNT(*)`, used for HAL estimates),
   `translate_facet`, `translate_facet_count`, `translate_search_related`. Free-text clauses are delegated to
   `query/text.py`.
3. `qleverlux/sparql/` (`builder.py` + `terms.py`) — a vendored/modified SPARQL Burger. Objects
   (`SelectQuery`, `GraphPattern`/`Pattern`, `Triple`, `Filter`, `Binding`, `BNode`, `Values`, …) render via
   `get_text()`. `BNode` and the `service=` argument on `GraphPattern` exist for QLever's `SERVICE` extensions.
   This is the only part with tests (`tests/`, the W3C and SPARQL 1.1 suites compared by algebra).

Predicate names in `predicates.py` and `config/sorts.json` must match the *index*, not the LUX search-term
name. Three families were wrong and are worth knowing about: "creation" is modelled as the record's
`Beginning` (there is no `startOf<Scope>Creation`); an event's timespan is the event, so it has
`startOfEvent`/`endOfEvent` rather than the `Beginning`/`Ending` pair every other scope uses; and the item
production event is `…OfItemBeginning`, not `…OfItemProduction`. Record type is `a`, not a `lux:` predicate.
When adding a mapping, count the triples first — `SELECT (COUNT(*) AS ?n) WHERE { ?s lux:<name> ?o }` — because
a wrong name fails silently as an empty result rather than an error.

The search vocabulary (`LuxConfig`) is read from the first of: `QLMT_SEARCH_CONFIG` (a path; it must exist,
or startup fails), `<config_path>/advanced-search-config.json`, and only then luxql's own copy
(`resolve_search_config` in `query/catalogue.py`; startup prints which). This repo ships
`config/advanced-search-config.json`, so every instance gets the same vocabulary with the code. Without it,
luxql uses the `advanced-search-config.json` saved in its package directory, and a luxql installed with pip has
none (`setup.py` does not ship it), so it fetches the **live LUX endpoint** and saves that. Live LUX has no
`similar`, which is how an instance ends up answering `similar` with "No possible scope for Leaf component".
On that fallback the vocabulary can change under a deployment with no change to this repo at all: when
related-list relations start being skipped, or facets start being dropped, check which file was used and its
date before looking for a regression here. A term disappearing upstream (`place.setCreatedHere` did,
mid-session) silently removes every relation whose path runs through it. A before/after comparison of compiled
output is only meaningful if that file did not move in between.

`build_lux_config` has to publish its patched `LuxConfig` into *every* luxql module that captured it, not just
`luxql.config`. `luxql.query` does `from .config import _cached_lux_config` at import time and every query node
takes `self.config` from that module global, so rebinding only `luxql.config._cached_lux_config` leaves query
nodes on the unpatched original — the extra terms then resolve in the catalogue but raise "No possible scope"
the moment a query actually uses one.

The predicate vocabulary is data, in `qleverlux/query/predicates.py`: `SCOPE_FIELDS` (relationships) and
`SCOPE_LEAF_FIELDS` (leaf values), keyed by scope then LUX search-term name, mapping to `lux:` predicate names.
A leading `^` means inverse (`^lux:foo`). `get_predicate` / `get_leaf_predicate` special-case classification,
memberOf, dimensions, and the date families (which expand to `startOf…`/`endOf…` pairs). `MISSED` ("missed") is
the sentinel for an unmapped term. Three things read this module — the translator, the related-list builder and
`files/derive_from_upstream.py` — which is why it is not a translator attribute. The translator still exposes
`scope_fields` / `scope_leaf_fields` as attributes pointing at the same dicts.

Transitive terms (`broader+`, `narrower+`, `partOf+`, `contains+`, `memberOf+`, `containingSet+`) map to
`+` paths there, and `SparqlTranslator.get_predicate` bounds them to `{1,QLMT_MAX_HOPS}` (default 8; 0 is
unbounded) through `bound_path`, which QLever supports with the usual distinct-pairs meaning. The hierarchies
have cycles (2,741 concepts reach themselves through `broader`), so the full closure is huge: `broader+` over
all concepts is 136M pairs against 16M for `{1,8}`. Related lists read `SCOPE_FIELDS` directly and are not
bounded, but none of them uses a transitive term.

Scopes are `item work agent place concept set event`; record classes (URL path segments) are the finer set in
`classEnum` (`object digital text visual person group period activity` …) and get folded back to scopes in
`do_hal_links`.

### Multi-scope search

`_scope: "multi"` searches several scopes at once and merges the results into one list. The query is an `OR`
whose branches each name their own `_scope`:

```json
{"_scope": "multi", "OR": [{"_scope": "concept", "text": "exhibition"},
                           {"_scope": "event", "text": "exhibition"}]}
```

luxql knows nothing about `multi` — `JsonReader` rejects the scope, and it ignores `_`-prefixed keys, so a
nested `_scope` would not be honoured anyway. It does not need to: every branch is an ordinary single-scope
query, so `read_multi_branches` (`query/catalogue.py`) reads each one in its own scope and hands the translator
`[(scope, parsed), ...]`. `translate_multi_search` emits one UNION alternative per branch, each carrying its own
`?uri a lux:<Scope>` (and portal filter) instead of a shared one, with the clause counter running across
branches so their per-clause variables cannot collide. `translate_multi_search_count` is the `COUNT(*)` form
used for HAL.

Sorting is optional. There is no `sorts.json` entry for `multi`; instead `SearchService.resolve_sort` looks the
sort key up in each branch's own scope. If every branch maps it to the *same* predicate the sort is used as-is
(`archiveSortId` does, because `lux:sortIdentifier` is unscoped); if they map it to different predicates
(`anySortName` -> `lux:<scope>SortName`) the translator emits a UNION inside the sort `OPTIONAL` so each branch
contributes its own value to one `?sortValue`; and if any branch lacks the key entirely it falls back to
relevance with a log line rather than failing. Relevance itself works across branches because each `?score_N`
is `COALESCE`d.

Note QLever orders strings by its configured ICU collation, not by codepoint, so a sort is verified by comparing
against QLever's own `ORDER BY` over the same values rather than Python's `sorted()`.

`multi` is accepted only by search and search-estimate, through `searchScopeEnum`; facets, related lists and
sorts are defined per scope and reject it with a 422. A malformed multi query — no `OR`, an empty `OR`, a
branch without `_scope`, or a branch that is itself `multi` — is a 400.

Two HAL relations use this (`lux:itemCurrentHierarchyPage`, `lux:setCurrentHierarchyPage`, plus
`lux:objectOrSetMemberOfSet` once that query exists upstream). Their templates also ask for
`sort=archiveSortId:asc` and `pageWith=<uri>`, neither of which exists: `lux:archiveSortId` and its per-scope
forms have no triples in the index, and `pageWith` ("the page containing this record") is not implemented. So
those links currently count 0 and are not emitted; the archive hierarchy needs an archive sort key in the data
before it can work.

### Text search and relevance

Free-text (`text` field) searches go through QLever **materialized views** named `<scope>Words`, one `SERVICE
view:<scope>Words { [ view:column-word "w" ; view:column-uri ?uri ; view:column-score ?tf ] }` block per word.
Those views are defined in `files/Qleverfile` under `MATERIALIZED_VIEWS` and weight primary name 14 / record text
5 / any-name 1 — the weights the relevance ordering depends on. Name-field searches instead use `ql:has-word`
inside a `GRAPH ?tf` pattern against `lux:<scope>Name`. Words are split the way QLever's index splits text: on every character that is not a letter or a digit,
accents kept (`query/text.py`, `tokenize`). So "children's" is `children` + `s`, as in the index; only double quotes
make a phrase (`shlex` once took an apostrophe for a quote and failed). A quoted phrase, and in name searches a
word written with an apostrophe ("o'brien"), is then checked with one RE2 `REGEX` (`match_pattern`): the words in
order and whole, ignoring case, separated by anything but letters and digits, or by at most an apostrophe-like
mark and spaces where the query has an apostrophe - so "o'brien" is not "o. brien". In a name search that runs
on the matched name and costs milliseconds. In a text search a phrase is a UNION of three sub-selects, one per
source (`lux:<scope>PrimaryName`, `lux:<scope>RecordText`, `lux:<scope>Any/lux:primaryName`), each starting from
the text index: joined any other way QLever read all 34M item record texts. A phrase of rare words takes
seconds, of common words tens of seconds, which is why an unquoted apostrophe word is not checked in text
searches. Words and phrases go into the SPARQL through `string_literal()`.

The relevance chain has four links, and all four have to line up: each word binds `?tf_<clause>_<word>`; the
clause sums those into `?score_<clause>`; `translate_leaf` records the clause number in `translator.scored`
(a `similar` clause goes in `translator.vector_scored` instead); and `_relevance_query` (only when sorting by
relevance) binds `?tscore = Σ COALESCE(?score_<clause>, 0)` and ranks on it, as below. The per-word variables must
stay qualified by the clause counter — two text clauses in one query would otherwise both bind `?tf_0` and
join on it — and a clause that produces a score but never reaches `scored` is silently dropped from the
ranking. Stopwords (`config/stopwords.json`) are stripped; an all-stopword query raises `ValueError`,
surfaced as a 400.

Text and vector relevance are each brought to 0..1 per result before they are added, so neither swamps the
other. `?sscore = text_scale × SUM(?tscore) / MAX over the query + vector_scale × (cos − min) / (max − min)`,
where `cos` is the result's `MAX(cosine)` and min/max are taken over the query's vector matches. The scales
are `QLMT_TEXT_SCORE_SCALE` and `QLMT_VECTOR_SCORE_SCALE`, both 1 by default. The text part has to be relative
to the query's own best hit because raw view scores have no fixed range: measured top scores run from 4
(`name: landscape`) to 548 (`text: rembrandt`), so no constant squash fits both. Dividing by one number per
query cannot reorder anything, so text-only rankings are exactly what they were. The vector part is min-max
scaled because raw cosines sit in a narrow band whose position depends on the model (top hits about 0.7–0.9
for EmbeddingGemma 2, 0.5–0.75 for Harrier). Added raw, the model would decide how much a vector match weighs
against text, in opposite directions for `AND` (where the spread matters) and `OR` (where the floor does).
Text-only matches have no cosine and are kept out of the min; a lone vector match counts as 1. The maxima
need the whole grouped result, so the relevance query is
`SELECT … { {grouped} {SELECT (MAX(?ts) AS ?tmax) (MAX(?vs) AS ?vmax) (MIN(…) AS ?vmin) {grouped}} }`. A
query with no text clause skips that subquery and ranks on the raw cosine, which orders identically.
QLever shares the duplicated work for the text views (no measurable cost), but **not** for `SERVICE`: a mixed
query calls the vector sidecar twice, about 40 ms extra. The 14/5/1 weights themselves are fixed when the
index is built, not here.

### Vector similarity (`similar`, `similarRecords`)

Two terms, both in every scope:

- `similar` ("are similar to") ships with luxql, `relation: "text"`, and is always free text, encoded by the
  sidecar, even when the value looks like a URI. Before it was wired up it fell through to keyword search.
- `similarRecords` ("are similar to the records") is the middle tier's own, added in `_EXTRA_TERMS`
  (`query/catalogue.py`) as a relationship to its own scope, so its value is any query in that scope:
  `{"id": <uri>}`, an `OR` of ids, or e.g. `{"name": "quantum physics"}`. The records it finds are the seeds:
  a result scores by its nearest seed, and the seeds are left out of the results. Ids are read straight off
  the query. Any other subquery has to be run before the `SERVICE` call can be written, which the translator
  (no QLever client) cannot do, so the services do it first: `services/seeds.py` runs each one as a search
  in that scope, takes the top 100 by relevance (the sidecar's seed limit; a subquery with no text clause
  has no ranking, so it is an arbitrary 100), and leaves the URIs on the node as `seed_records`. Search,
  estimate and facets all call it between parsing and translating; a new route that translates user queries
  must too, or the translator refuses the subquery. No seeds means no results (`FILTER (false)`), and seeds
  the vector store has no vector for are skipped by the sidecar, which only errors when none are left.

`SparqlTranslator.translate_similar` / `translate_similar_records` send them to the `sparql_service` sidecar
(see its README for the contract) as one `SERVICE` call:

```sparql
SERVICE <http://127.0.0.1:8081/sparql> {
   ?uri vec:similarTo <seed-uri> .      # similarRecords: one triple per seed; similar: an escaped literal
   ?uri vec:score ?score_N .            # cosine, 0..1; ranked as described above
   ?uri vec:topK 100 .
   ?uri vec:index "item" .              # the leaf's scope, so vector-first search stays in scope
}
```

- `QLMT_VECTOR_SERVICE` is the URL *as QLever sees it* (loopback means QLever's host). It is stored bare and
  bracketed by `iri()`; the builder only adds a `:` to a service name with none, so `<…>` passes through. Empty
  (the default) disables both terms: they are then a 400, not a silent keyword search.
- `sort=similarity:desc` (or `:asc`) ranks on the cosine alone. Text clauses still filter, and their raw score
  breaks ties, which orders the text-only matches of a `similar OR text` after the similar ones. Like
  `relevance` it is computed by the translator (`RANKED_SORTS`), not looked up in `sorts.json`, so it works for
  `multi` too. It needs no per-query max, so the sidecar is called once. With no `similar` clause it falls back
  to relevance.
- `vec:topK` depends on where the clause sits. At the top of the query (booleans do not count) it is
  `QLMT_VECTOR_TOP_K` (default 100), which also caps the results and the estimate (≤ K). Under a
  relationship (`_under_relationship`), as in works `aboutConcept` {similar}, it is
  `QLMT_VECTOR_NESTED_TOP_K` (default 5): there each similar record pulls in its related records unranked,
  so the weak matches near 100 would swamp the close ones. Choosing K per query needs luxql to carry an
  option on the leaf; it only passes `_options`, `_weight` and `_complete` through today.
- Cosine cutoffs apply after top-K, with the same top/nested split: `QLMT_VECTOR_MIN_SCORE` /
  `QLMT_VECTOR_MAX_GAP` and `QLMT_VECTOR_NESTED_MIN_SCORE` / `QLMT_VECTOR_NESTED_MAX_GAP`, sent as the
  sidecar's `vec:minScore` (absolute floor) and `vec:maxGap` (distance below the best hit). 0, the default,
  leaves one off. To compare a cutoff with the small nested K, raise `QLMT_VECTOR_NESTED_TOP_K` so the
  cutoff is what bites. `files/similarity_cutoffs.py` prints the score curve, what each value keeps and
  the names at each rank.
- `vec:mode "constrained"` is added when the leaf is one of several children of an `AND`
  (`QLMT_VECTOR_CONSTRAIN_IN_AND`), so a constraint too big for QLever to push down (`service-max-value-rows`)
  is an error instead of a near-empty result. A lone `similar` uses the default `auto`, because its only
  constraint is `?uri a lux:<Scope>`, which is far too big to push.
- The sidecar (`sparql_service/`, contract in its README), the offline build (`lux_embed/`) and the store
  they share (`qleverlux/vectors/lmdb_vector_store.py`) are all in this repo, but are separate commands in a
  separate venv: the middle tier never rebuilds or imports any of it. The built data stays outside the repo
  (`../vectors/data/`), and so does the USearch checkout. Serving in place needs USearch `840374af` with
  `files/usearch-view-ram-buffer.patch` applied; drop the patch once upstream has `view_ram_buffer`. Their
  tests skip without numpy, lmdb and usearch.
  A scope's graph can be built narrower than its vectors, on the leading dimensions (`index --graph-dims`)
  or on a linear map stored with the graph (`index --projection`, from `files/learn_projection.py`; `--epochs
  0` there is query-aware PCA); either way the rerank uses the full vectors, over a 10× pool. Keep both:
  agents' full-width graph is over LMDB's 4 GiB value limit, and under heavy load, with the store mostly
  out of memory, a smaller graph is fewer pages to fault in per walk.
  Three things to keep: slice HNSW results to `counts[row]` *before* `keys_to_integers` (the rest of the buffer
  is garbage, and the walk returns at most `expansion_factor` hits), and never hold a `usearch.Executor`
  past the call that needs it — an undrained one spins about two cores even when idle. And don't read
  vectors with `txn.get`: py-lmdb touches every page of a value it returns, so scoring a few hundred hits
  pulled in each whole chunk they sat in (60,070 page faults and 2 s for 300 rows, against 412 and 7 ms).
  `_RowReader` finds a chunk from its key's leaf node and reads just the rows with `os.pread`; it checks the
  LMDB layout against the file first and falls back to `txn.get` where it does not fit.
- User values go through `iri()` / `string_literal()` (`sparql/terms.py`), which reject or escape anything
  that could break out of the term; a `similarRecords` id must also start `http://`/`https://`.

Pagination is deliberately coarse: SPARQL is issued at offsets rounded down to multiples of 60, for a 60-row
window (`LIMIT`; wider only when a page would cross its end), so the alru cache hits across nearby pages, and the
exact page is sliced out of the window in Python. The total is a separate `COUNT(*)` over the same pattern and
(`?uri`, `?type`) grouping (`translate_search_total`; HAL's `translate_search_count` counts bare URIs and can
differ), run alongside the window and shared by every page, sort order and search-estimate. It cannot come from
the window: QLever counts the rows after `OFFSET` and, with a `LIMIT`, only the rows it returns. Before the window
had a `LIMIT`, every page fetched every row from its offset to the end: 40 MB for concepts `broader+` "art", and
26 s for items `text: "sculpture"` sorted by name, against 1.7 s for the window.

### AI query translation

`/api/ai-translate` turns a natural-language question into candidate LUX JSON queries, and with `prevQuery` set
turns an existing query plus a requested change into revised ones. The upstream of this is
`../lux-ai-query-builder` (`query-cli.py`); `config/system-prompt-translate.txt` and
`config/system-prompt-improve.txt` are byte-identical copies of the prompts there, and
`services/ai_translate.py` reproduces its post-processing — verified fixture by fixture, including the scope
traversal, which only accepts a relation that names one of the seven scopes.

The model answers with an `options` array; each option has a `natural` description, a `scope` and a compact
query tree (`f` field, `v` value, `c` comparator, `p` boolean children, `r` relationship child) that
`post_process` expands, coercing the dimension fields to float, `hasDigitalImage` to int and `recordType` to
lower case. Unlike the reference CLI, which returns only the first option, the endpoint returns all of them.

The response shape is fixed by the front end (`lux-frontend`, `IAiDisambiguation` plus `SearchBox.tsx` and
`Disambiguation.tsx`), which reads `data[0].query`, `queryData.query._scope` and `queryData.natural`:

```json
[{"natural": "17th century Italian paintings", "parsed": "", "query": {"AND": [...], "_scope": "item"}}]
```

So `_scope` goes *inside* the query — the client reads it there and deletes it before running the search, and
`JsonReader` accepts the query either way. `natural` comes from both prompts; `parsed` only from the improve
prompt, so it is present but empty on a build. The front end requests `api/ai-translate/{scope}`, so both that
and the bare `/api/ai-translate` are routed; the scope in the path is accepted and ignored, because the model
chooses a scope per option.

`clients/ai/` holds one module per backend behind `AiBackend`, chosen with `QLMT_AI_TRANSLATE_BACKEND`:

- `gemini` — Gemini on Vertex AI. Generation settings match the reference exactly (temperature 0.8, top_p 0.95,
  36000 tokens, thinking budget 2000, all four safety categories OFF), because the prompts were tuned against
  them.
- `lmstudio` — a local model through the LM Studio SDK. `QLMT_AI_TRANSLATE_API_ENDPOINT` is `host:port`, and
  note the handle comes from `client.llm.model(...)`; `client.llm` is a session namespace, not a factory.
- `openai` — any OpenAI-compatible `/v1/chat/completions` endpoint (MTPLX, vLLM, llama.cpp, Ollama, LM Studio's
  own OpenAI endpoint, hosted OpenAI). `QLMT_AI_TRANSLATE_API_ENDPOINT` is the `/v1` base URL.

Left empty, the backend is inferred so existing deployments keep working: a `gemini` model name means Gemini,
otherwise an endpoint means `openai`, otherwise translation is off. `lmstudio` is never inferred. All three
imports are guarded, so only the backend in use has to be installed.

Two things every backend has to cope with. JSON mode is requested where supported but plenty of servers reject
the parameter, so a rejection is retried once without it and then left off. And models wrap their JSON in
markdown fences or prose whatever the prompt says, so `parse_model_json` strips fences and falls back to the
outermost `{...}`; it returns None rather than raising.

A model at temperature 0.8 also sometimes answers with an empty `options` array or with something that will not
parse, so `translate()` allows one retry — the same single retry the reference proxy did — before returning 502.
An empty list is deliberately not returned: the client indexes `[0]` when there is only one option, so it would
fail on the caller's side rather than here. A backend *exception* is not retried, because a refused connection
or a timeout will not come good on a second try and retrying a timeout just doubles the wait.

A local 27B model takes about a minute per query, so `QLMT_AI_TRANSLATE_TIMEOUT` (120 s) is generous. It is one
deadline for the whole translation, retry included (0 is none): the retry gets only what the first attempt
left, and is skipped when nothing is left. Running out is a 504; any other backend failure a 502. A thread
cannot be stopped from outside, so every backend stops itself (`generate(..., deadline=)`, helper `time_left`):

- `openai`: the time left as the request's `timeout`, and the SDK's own retries off (`max_retries=0`); they
  once tripled a timeout before the service's retry doubled it again.
- `gemini`: the time left as `http_options.timeout` (which also sends Vertex `X-Server-Timeout`), plus a check
  between streamed chunks, since on a stream that timeout bounds each read, not the response. It had none.
- `lmstudio`: `respond_stream` instead of `respond`, checked at each fragment and each prompt-processing update;
  past the deadline the prediction is cancelled, which stops the model. The SDK's only limit is 60 s with no
  message at all, so a model still generating had none. The check is a callback that cancels rather than
  raises, because the SDK swallows callback exceptions.

A translation can therefore overrun its deadline by at most one SDK wait: one read for openai and gemini, and
the SDK's 60 s silence limit for lmstudio.

Every backend's SDK call blocks, so `translate()` runs it (both attempts) on a worker thread; on the event loop
one request stopped the process serving anything else for that minute. At most
`QLMT_AI_TRANSLATE_MAX_CONCURRENT` (2) run at once per process and a request beyond that is a 503, not a queue.
The slot is held until the thread finishes rather than until the caller leaves, since a thread cannot be
cancelled; the deadline above is what keeps a stalled backend from holding every slot. `q` and `prevQuery` longer than `QLMT_AI_TRANSLATE_MAX_QUERY_LENGTH` (1000) /
`QLMT_AI_TRANSLATE_MAX_PREV_QUERY_LENGTH` (20000) characters are a 400: the endpoint is public and a hosted
model bills by the token.

### CMS stub (`/jsonapi`)

The front end takes its editorial content (landing page, hero images, featured collections, FAQs, content pages,
results-page overlays) from a Drupal JSON:API at `REACT_APP_CMS_API_BASE_URL`. `services/cms.py` stands in for
it at `<mt_uri>jsonapi/`, so pointing the front end there is a host change only. It serves exactly the two shapes
`lux-frontend/client/src/redux/api/cmsApi.ts` asks for — `node/<type>` (with `page[limit]`/`page[offset]`) and
`node/<type>/<uuid>` — from `QLMT_CMS_PATH/node/<type>.json`, a snapshot taken by `files/snapshot_cms.py`.
The snapshot drops `links` (the front end never reads them); `data` is otherwise identical to the live CMS.
Edits on disk take effect without a restart: each request compares the files' mtimes with the last load and
rereads on any change, which also works per worker under `workers.py` (a reload endpoint would reach only one).
A file that fails to parse leaves the previous content in service.
The UUIDs the front end requests are hard-coded in its `src/config/cms.ts`, so a page recreated in Drupal
needs a fresh snapshot *and* a front-end change. `searchTips` is one of those and is unpublished upstream, so
it is a 404 here as it is a 403 there.

### Config-driven query catalogue (`config/` + `queries/`)

`QueryCatalogue` (`query/catalogue.py`) loads these at startup and precompiles everything to SPARQL text once,
so runtime work is string substitution:

- `queries/*.json` — ~90 named LUX JSON queries with `URI-HERE` placeholders. Generated from the Node.js middle
  tier's `lib/build-query/` by `files/derive_from_upstream.py`.
- `config/hal_links.json` — HAL relation → `{queryName, template, scope}`. Each becomes a count query;
  a non-zero count means the link is emitted on the record.
- `config/related_list_scopes.json` — for each scope/related-list, `from-to` field-path → result scope. A
  relation naming a term with no QLever predicate is skipped — from both the SPARQL and the JSON side, which
  have to stay in step — and reported once at startup rather than stopping the server; that is a safety net for
  a config generated without validation, not a substitute for regenerating it.
  `RelatedListBuilder.query_stub` (`query/related.py`) turns each into an `OPTIONAL { SELECT … COUNT }` fragment
  plus a cheap `LIMIT 1` HAL existence test; both are ordered by a hand-tuned heuristic score (classification
  and `workAbout` boost, multi-hop paths and Set/Event penalise) so HAL checks can short-circuit on the first
  hit. `json_stub` builds the equivalent user-facing JSON query using `config/terms_inverse.json`.
  The count query ranks related records by `DESC(?total) ?uri` (the `?uri` tie-break keeps the order the
  same at every LIMIT) and ends `LIMIT V_LIMIT`. As on the MarkLogic backend, a page counts *entries*, one
  per record and relation (`pageLength` default `QLMT_RELATED_PAGELENGTH`=25, max 1000). Each record
  gives at least one entry, so `RelatedListService` asks for `offset + pageLength + 1` records, rounded up to
  60 for the cache, expands them and slices the page out. `next` appears only when there is another
  entry: the front end shows its Next button exactly when `next` is present.
- `config/facets.json` — facet name → `searchTermName`; entries whose term isn't in `LuxConfig` are dropped at
  startup with a warning. `config/sorts.json` — sort key → predicate, per scope.

Placeholders used across the templates: `URI-HERE`, `V_TARGET_URI`, `V_FROM_URI`, `V_TO_URI`,
`V_NAME_REL`/`V_URI_REL`/`V_TARGET_REL`, `V_LIMIT`, `{q}`, `{id}`, `{searchUriHost}`.

### Caching layers

- PostgreSQL connections take keyword arguments (`PostgresCache.connect_args`), never a pasted conninfo string,
  and over TCP (`QLMT_PGHOST` set) apply `QLMT_PGSSLMODE`, `require` by default; empty leaves libpq's own
  default, `prefer`, which silently falls back to an unencrypted connection. With no host it is a Unix socket,
  with no password or SSL, as before.
- Record JSON: PostgreSQL `lux_data_cache` (`QLMT_USE_PG_DATA_CACHE`) and/or LMDB (`QLMT_LMDB_PATH`, zlib-
  compressed values). Loaded by `files/load-json-to-postgres.py`. `RecordCache` (`clients/record_cache.py`)
  picks between them; with both PostgreSQL tables on, one joined query fetches the record and its links
  together. LMDB keys follow `QLMT_LMDB_KEY_FORMAT`: `uuid` (raw UUID bytes, LUX), `text`, or `qid` for the
  Wikidata store written by `../data-pipeline/make_wikidata_lmdb.py` — a 4-byte Q number plus a character for
  the record's type, because one Q-id is cached once per type (Q90 is a place and a concept). The type comes
  from the class in the URL, and the character from the store's own `types` table. The same format decides
  what the record route accepts as an identifier (`normalise_identifier`): a non-UUID on a `uuid` instance is
  a 422.
- HAL links: disk (`hal_cache/*.json`, default) or PostgreSQL `hal_data_cache`, behind `HalCache`. Computing
  them is expensive — one query per candidate relation — so a cache miss is a slow request. A block is cached
  only if every query it needed answered: a probe that timed out or was shed says nothing about the link, so
  that block is served but not kept, and the next request tries again. Caching it once made a moment of
  overload permanent for those records.
- SPARQL responses: in-process `alru_cache` on `QLeverClient`.

### URI rewriting

Data uses `QLMT_DATAURI` (`https://lux.collections.yale.edu/`); responses must use the deployment's own
`mt_uri`, assembled from `replace_proto`/`replace_host`/`replace_port`/`replace_path`. All of it goes through
`UriRewriter` (`presentation/uris.py`): `inbound()` mt→data before translation, `outbound()` /
`outbound_record()` / `outbound_json()` data→mt on the way out (the last still via a `json.dumps`/`loads` round
trip). Any new endpoint needs both directions — use the rewriter rather than `.replace()` in a handler.

Records are always served at `<mt_uri>data/<class>/<id>`; where they sit in the data is `QLMT_DATAURI` +
`QLMT_RECORD_PATH` (default `data/{class}/{id}`). With the class in the data URI, outbound is the old string
prefix swap, byte-identical to before. Wikidata's URIs have none (`QLMT_RECORD_PATH={id}`), so outbound needs
the record's Linked Art type: `outbound_json` walks the document and rewrites each node with both `id` and
`type`, and search passes the `?type` its query returns. A URI with no known type — related-list entries,
facet values — is left as the data URI, which `inbound()` passes through unchanged, so links still work. In
that mode one identifier is several records, so the HAL cache key is `<id>-<class>` rather than `<id>`.

Note `mt_uri` has no trailing slash unless `QLMT_EXTERNAL_PATH` provides one, and every URI is built by
concatenation (`f"{mt_uri}data/…"`), so a deployment must set `QLMT_EXTERNAL_PATH=/`.

`QLMT_PORTAL` (YPM, YCBA, YUAG, PMC, IPCH) makes the translator inject `?uri lux:source lux:<portal>` into every
pattern, turning the instance into a single-unit portal. The catalogue wires it onto the translator.

## Regenerating the derived config

`files/derive_from_upstream.py` re-derives `queries/*.json` and most of `config/` from two upstream checkouts —
`../lux-middletier` (queries, `hal_links.json`) and `../lux-marklogic` (`facets.json`, `related_lists.json`,
`terms_inverse.json`, `related_list_scopes.json`); `stopwords.json` comes from the instance's advanced search config
(resolved as the middle tier does, falling back to luxql's). It needs `node` on PATH but no npm install. It defaults to a dry run that reports the drift;
`--write` applies, `--check` exits 1 when out of date, `--prune` deletes queries dropped upstream.

`-b`/`--base-dir` (or `QLMT_BASE_DIR`) picks which middle tier gets updated — any directory holding `config/` and
`queries/`, so a deployed instance can be refreshed without going through this checkout; `--config-path` and
`--queries-path` cover instances that split them. Only the written files move with `--base-dir`: the predicate
vocabulary still comes from whichever `qleverlux` is importable, else this checkout. Upstream checkouts are
looked for beside this checkout, beside the installed `qleverlux`, then beside the instance, so a copy of the
script dropped into an instance directory still works with no arguments.

Two things it deliberately does not do. `config/sorts.json` maps LUX sort keys onto QLever predicate paths that
have no upstream counterpart, so it is only audited, not written — which also means a fix to it in this checkout
does **not** reach a deployed instance the way a derived file does. Copy it, or apply the same edit there.

Corrections to a *derived* file go in `derive_overrides.json` instead, and do survive. `facets.json` needs
several: upstream names some search terms differently from luxql (`curationAgent` for `curatedBy`,
`creationOrPublicationDate` for the scope-prefixed form), and without the patch `_trim_facets` deletes the facet
at startup with a `Could not find search term` warning.

That file also carries what this middle tier deliberately does **not** support, so upstream keeps publishing it
and we stop reporting it as a gap: `facets.json` -> `delete` drops the facet at regeneration, and
`sorts.json` -> `ignore` keeps `audit_sorts` quiet about the sort key. `setLastModifiedById` /
`setLastModifiedDate` are there because sets were once user-editable collections and those queries found the
recently modified ones; that usage is retired. And generated related-list relations whose hops
have no predicate in `query/predicates.py` are dropped with a message naming the missing predicate — add it there
to let the relation through.

**A copy of this script inside an instance goes stale.** It finds the predicate vocabulary through the
`qleverlux` it can import, so a copy that predates a move of that module silently finds nothing — and an
unvalidated `related_list_scopes.json` names predicates QLever has no term for. It now preflights that lookup
and exits before writing anything rather than producing a config the middle tier has to work around; refresh the
instance's copy from this checkout when it fails that way. `--no-validate` is the deliberate override.

`queries/*.json` are validated against `LuxConfig` too, because upstream describes MarkLogic's full vocabulary
and can emit a query luxql rejects. The report splits them by whether it matters: a query some HAL relation
references is read at startup, and when it will not parse that link is simply absent from records, so those are
listed with the relations they cost; the rest are unused files and get one line. The prediction is exact — the
relations named are the ones startup logs `Error parsing query for <hal>` about. Validation uses the *patched*
`LuxConfig` from `query/catalogue.build_lux_config`, so the middle tier's extra search terms do not read as
errors. Queries are still written either way; unlike `related_list_scopes.json` an unparseable query costs one
HAL link rather than breaking startup, so it is a warning, not a refusal.

Three of these are long-standing: `currentItemAndSiblings`, `currentSetAndSiblings` and `itemsOrSetsMemberOfSet`
use `_scope: "multi"`, which luxql has no support for. `setCreatedPublishedInfluencedByAgent` is the other one —
`set` has no `creationInfluencedBy` predicate, and `lux:agentOfSetBeginning` has no triples in the index either,
so there is nothing to map it onto. Hand corrections that must survive regeneration go in
`<config-path>/derive_overrides.json` as per-file `patch`/`delete` rules; an instance without its own file falls
back to this checkout's.

The script runs the upstream JavaScript (it imports `lib/build-query/` through `node`, and evaluates the
MarkLogic config literals), so only point it at checkouts you trust. Its regex-based predecessor,
`files/translate_query.py`, `eval`ed the upstream query files in Python and has been removed.

## Verifying a change

Translation and the catalogue have no tests, so the practical check on a change to either is that the compiled output is what you expect. Build a `QueryCatalogue` and dump everything it
produces — `sparql_hal_queries`, `related_list_sparql`, `related_list_json`, `hal_related_list_tests`, plus
`translate_search` / `translate_facet` output for a spread of scopes and query shapes — to sorted JSON before
and after, and diff. A pure refactor should come out byte-identical; a behaviour change should show up only
where you meant it to.

```python
from qleverlux.settings import load_settings
from qleverlux.query.catalogue import QueryCatalogue
cat = QueryCatalogue(load_settings([]))
print(cat.translator.translate_search(cat.json_reader.read(jq, "item"), scope="item").get_text())
```

For the request path, `create_app(middletier=mt)` takes a `MiddleTier` whose `qlever` attribute you have
replaced with a stub, so every route can be exercised through `fastapi.testclient.TestClient` with no QLever,
PostgreSQL or network.

## Notes

- `qleverlux/sparql-orig.py` is an untracked pre-materialized-view snapshot of the old flat `sparql.py`, kept
  for reference. It is not importable (the hyphen) and is not part of the package. Do not edit it or import
  from it.
- A search term mapped to `""` in `SCOPE_FIELDS` (currently only `item.usedForEvent`) means LUX has the term
  but QLever has no single predicate for it. `get_predicate` resolves it to `MISSED`, `RelatedListBuilder`
  skips any relation containing it, and `derive_from_upstream.py` drops such relations — treat `""` as "no
  predicate", never as a predicate.
