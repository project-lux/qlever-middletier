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

python uvicorn-serve.py                 # dev: HTTP on :5001, single process
python serve.py                         # hypercorn HTTPS/2 on QLMT_MTPORT (needs files/<cert_name>.pem + -key.pem)
python workers.py                       # hypercorn multi-process
python tester.py [scope]                # REPL: type a query string or JSON, prints the generated SPARQL
python -m pytest tests/ -q              # SPARQL builder conformance tests (W3C + SPARQL 1.1 suites)
```

All three servers are three-line shims over `qleverlux/server.py` (`serve_dev` / `serve_https` /
`run_workers`), which is also runnable as `python -m qleverlux.server [--dev|--workers]`. Connection pools are
opened by the app's lifespan handler, so nothing has to remember to do it.

The only tests are for `qleverlux/sparql/` (the query object model); the middle tier itself has none, so treat
the compiled output as the contract when changing it — see "Verifying a change" below. `tester.py` is the fastest
feedback loop for query-translation changes: it exercises the parser → JsonReader → SparqlTranslator chain
without needing QLever running.

Configuration is env-var first (`.env` via `dotenv`, all keys prefixed `QLMT_`), overridden by CLI flags. Each
setting is declared exactly once, as a field on `Settings` in `qleverlux/settings.py`; its env var is `QLMT_` +
the field name unless the field carries an explicit alias, and its flag is the field name with hyphens unless
it is listed in `NO_CLI`. Booleans get `--flag` / `--no-flag`. Passing `--pgpass` deliberately aborts the
process; use `.env`. `parse_known_args` is used, so unknown flags (e.g. uvicorn's) are tolerated and stashed in
`settings.remaining_args`.

QLever setup (indexing, materialized views, the UI) is described in `README.md` and `installing_qlever.md`; the
canonical index/server settings live in `files/Qleverfile`.

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
and both bail out with a 504-ish stub when `open_requests > max_qlever_requests` unless called with
`sheddable=False` (HAL generation does, so it can't be shed halfway through a links block). Results come back in
QLever's own `application/qlever-results+json` and are unwrapped by `process_qlever_results` into
`{results, total, time, variables}` with `<uri>` brackets stripped and typed literals coerced.

Services call `run_query(client, sparql)` from `errors.py`, which raises `QLeverError` on failure; one exception
handler renders it, so handlers only deal with results that worked.

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

`LuxConfig` is not a fixed input. luxql fetches the advanced search config from the **live LUX endpoint** and
caches it to `advanced-search-config.json` inside its own package (`cache_remote_config=True`), so the search
vocabulary can change under a deployment with no change to this repo at all. When related-list relations start
being skipped, or facets start being dropped, check that file's date before looking for a regression here: a
term disappearing upstream (`place.setCreatedHere` did, mid-session) silently removes every relation whose path
runs through it. It also means a before/after comparison of compiled output is only meaningful if that file did
not move in between.

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
inside a `GRAPH ?tf` pattern against `lux:<scope>Name`. Quoted phrases become `CONTAINS()` filters over a UNION of
the three text sources.

The relevance chain has four links, and all four have to line up: each word binds `?tf_<clause>_<word>`; the
clause sums those into `?score_<clause>`; `translate_leaf` records the clause number in `translator.scored`;
and `translate_search` (only when `sort == "relevance"`) binds
`?score = Σ COALESCE(?score_<clause>, 0)` and orders by `SUM(?score) AS ?sscore`. The per-word variables must
stay qualified by the clause counter — two text clauses in one query would otherwise both bind `?tf_0` and
join on it — and a clause that produces a score but never reaches `scored` is silently dropped from the
ranking. Stopwords (`config/stopwords.json`) are stripped; an all-stopword query raises `ValueError`,
surfaced as a 400.

Pagination is deliberately coarse: SPARQL is issued at offsets rounded down to multiples of 60 (so the alru cache
hits across nearby pages) and the exact page is sliced out of the result list in Python.

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

A local 27B model takes about a minute per query, so `QLMT_AI_TRANSLATE_TIMEOUT` is generous by default.

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
- `config/facets.json` — facet name → `searchTermName`; entries whose term isn't in `LuxConfig` are dropped at
  startup with a warning. `config/sorts.json` — sort key → predicate, per scope.

Placeholders used across the templates: `URI-HERE`, `V_TARGET_URI`, `V_FROM_URI`, `V_TO_URI`,
`V_NAME_REL`/`V_URI_REL`/`V_TARGET_REL`, `{q}`, `{id}`, `{searchUriHost}`.

### Caching layers

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
  them is expensive — one query per candidate relation — so a cache miss is a slow request.
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
`terms_inverse.json`, `related_list_scopes.json`); `stopwords.json` comes from the advanced search config that
`luxql` ships. It needs `node` on PATH but no npm install. It defaults to a dry run that reports the drift;
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

`files/translate_query.py` is the earlier, regex-based version of the same idea and is superseded by it.

## Verifying a change

Nothing outside `qleverlux/sparql/` has tests, so the practical check on a change to translation or the
catalogue is that the compiled output is what you expect. Build a `QueryCatalogue` and dump everything it
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
