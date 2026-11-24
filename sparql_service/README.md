# sparql_service — the LUX vector SERVICE for QLever

This is a loopback sidecar that QLever calls through `SERVICE <http://127.0.0.1:8081/sparql> { … }`. It answers with `application/sparql-results+json` from the LMDB vector store built by `lux_embed`.

```bash
cd ~/Development/lux/qlever-middletier      # this repo's root; a separate process from the middle tier
HF_HUB_OFFLINE=1 ../vectors/.venv/bin/python -m sparql_service --store ../vectors/data/mds/vectors.lmdb --port 8081
curl -s http://127.0.0.1:8081/health      # scopes, sizes, generations, in-place serving
```

## Query contract

```sparql
PREFIX lux: <https://lux.collections.yale.edu/ns/>
PREFIX vec: <http://lux.collections.yale.edu/ns/vec#>

SELECT ?set ?name ?score WHERE {
  ?set a lux:Set ; lux:setPrimaryName ?name .                 # graph constraint, evaluated by QLever
  SERVICE <http://127.0.0.1:8081/sparql> {
    ?set vec:similarTo "Roman coins" ;                         # text, or a LUX IRI: <https://…/data/…>
         vec:score ?score ;                                    # cosine similarity (xsd:double)
         vec:topK 5 .
  }
} ORDER BY DESC(?score)
```

| Predicate | Object | Meaning |
| --- | --- | --- |
| `vec:similarTo` | IRI(s) and/or literal(s) | **Required.** An IRI uses that record's stored vector; IRIs with no stored vector are skipped, and it is an error only when no seed is left. A literal is encoded with the model the store was built with (recorded in its meta; older stores are EmbeddingGemma 2), using that model's query prompt (`SearchQuery`, or Harrier's `web_search_query`). Every scope must share one model. For several seeds, give a list (`vec:similarTo <a>, <b>, <c>`) or repeat the triple; at most 100. |
| `vec:combine` | `"max"`, `"each"`, `"mean"` | How several seeds combine (see below). Default `"max"`. |
| `vec:includeSeeds` | `true` / `false` | Default `false`: seed IRIs are left out of the results, and their slots are refilled so `topK` rows still come back. |
| `vec:score` | variable | Cosine similarity. |
| `vec:rank` | variable | 1-based rank. |
| `vec:topK` | number | Results to return. Default 20, maximum 10,000. |
| `vec:minScore` | number | Drops hits below this cosine, after `topK`. |
| `vec:ef` | number | HNSW expansion factor for this query (at most 50,000), overriding `--expansion-factor` (default 10,000; at least `topK` × 3 is always used). A walk can be trapped by a dense cluster of near-duplicates and miss the true neighbours; a wider walk escapes it, at a roughly linear cost. `files/similarity_cutoffs.py --recall` measures what a value finds. Exact scoring of pushed-down candidates ignores it. |
| `vec:rewalk` | number | Walks again from the stored vectors of the best N hits (0–50) and reranks the pool, overriding `--rewalk` (default 10). A text query sits away from every record, where a walk can pass by records the graph does link; a walk from a record starts where the graph is reliable. |
| `vec:rewalkEf` | number | Expansion factor of the second walks, overriding `--rewalk-ef` (default 300; at least `topK` × 3 is always used). Narrow ones are cheap: on dev ten at ef 300 added 20–30 ms and found a top hit the first walk had missed. Wide ones (ef 10,000) recovered two more of ten on some place queries but cost about 200 ms. |
| `vec:exact` | `true` / `false` | `true` scores every vector of the scope with no graph: perfect recall, cost linear in the scope (about 77 ms for 327k 768d concepts on a laptop; 1–3 s a query on dev). `false` forces the graph. By default a scope of at most `--exact-below` records (default 0, so none) is scored exactly. |
| `vec:maxGap` | number | Drops hits more than this below the best hit (of each seed's ranking, for `"each"`), after `topK`. Raw cosines sit in a band that moves with the query and the model, so this follows the query where `minScore` cannot. `files/similarity_cutoffs.py` shows the curve and what each value keeps. |
| `vec:index` | `"concept"`, `"work,item"`, … | Restricts or chooses scopes. Optional: by default the scope is inferred from the IRIs' path segments (`object`/`digital` → item, `visual`/`text` → work, `person`/`group` → agent, `activity` → event, …). |
| `vec:mode` | `"auto"`, `"constrained"`, `"global"` | `"constrained"` refuses to run without pushed-down candidates; see below. |

## Several seeds: "similar to any of these records"

```sparql
SERVICE <http://127.0.0.1:8081/sparql> {
  ?uri vec:similarTo <…/activity/A>, <…/activity/B>, <…/activity/C> ;
       vec:score ?score ; vec:rank ?rank ; vec:topK 20 ; vec:combine "each" .
}
```

One SERVICE call does it; there's no need for a UNION of single-seed calls.

| `vec:combine` | Ranking | Order by | Use it for |
| --- | --- | --- | --- |
| `"max"` (default) | One ranking: each record scored by its best-matching seed. | `DESC(?score)` | "more like these" when the seeds are alike. The seed with the tightest neighbourhood can take every slot; with a war and two football tournaments as seeds, the top 6 were all tournaments. |
| `"each"` | Each seed's own top hits, interleaved round-robin. | `?rank` (scores come from different seeds) | "similar to any of these" with every seed represented. |
| `"mean"` | One search against the seeds' normalized centroid. | `DESC(?score)` | "similar to the group as a whole". |

The SERVICE body may contain only these `vec:` triples about one subject variable. Anything else gets an HTTP 400 that says why, and QLever shows that message.

## Modes

- **Graph-first.** QLever pushes the join candidates as `VALUES` (measured on QLever 0.6.0: a POST with an `application/sparql-query` body, `VALUES (?x) { (<iri>) … }` appended at the end).
  - Candidates are grouped by scope and scored **exactly**.
  - Measured through QLever: 2,355 pushed sets took about 0.1 s; 50,687 pushed concepts (a `broader+` closure plus a text query) took about 1 s.
- **Vector-first.** There's no `VALUES` (no constraint, or a join side QLever can't push).
  - HNSW top-K per scope, then an exact rerank, merged by score.
- **`vec:mode "constrained"`.** QLever pushes nothing when the sibling result exceeds `service-max-value-rows` (default 10,000; set at start with `qlever start service-max-value-rows=100000`), and it gives no warning.
  - A global top-K joined against a large constraint then returns almost nothing.
  - With this mode the service returns an error that names the cause. Use it on every graph-constrained query.

## Behaviour notes

- **`head.vars`** always equals the SERVICE's `SELECT` variables, which QLever requires.
- **Cross-scope queries.** A wider query vector (e.g. a 768d concept) is sliced Matryoshka-style to a narrower index (512d item) and re-normalized. A narrower one can't query a wider index; use a text query.
- **Deadline.** The default is 30 s (`--timeout`); past it the service returns 503. QLever has no SERVICE timeout of its own.
- **Startup.** The port opens only once the service is ready. A child process (`sparql_service/warm.py`) reads the live store into the page cache in the order searches need it — every scope's graph, then the IRI maps, then the vector chunks — and stops at `--warm-gb` (default: half of RAM). It never reads free pages such as old graph generations. On a small machine the vectors are what goes cold, smallest scopes first. Meanwhile the encoder loads. Then every scope's graph is opened and one query is encoded and searched per scope. Each step is logged with its time. If the encoder cannot load, startup fails rather than serving a sidecar that errors on every text query; if warming fails, it logs a warning and carries on. `--no-warm` skips all of it, and the first queries then pay for it: loading the model alone can outlast the deadline.
- **Caching.** One search sends the same SERVICE call several times (each facet query repeats the search's pattern), so results are cached in one LRU bounded by `--cache-mb` (default 256). It caches three things:
  - the response, keyed by a hash of the request body;
  - the ranked hits, keyed by seeds, options and the pushed-down candidates taken as a sorted set, so the same candidates in another order, or under other variable names, still hit;
  - encoded query text.

  Identical requests in flight share one computation, and one that outlives the deadline still finishes and fills the cache for the retry. Entries are tagged with every scope's graph generation, so a new graph invalidates them. `X-Vector-Cache` on each response says `response`, `hits`, `shared` or `miss`; `GET /cache` gives entries, bytes, hits and misses.
- **Rerank pool.** A walk's candidates are rescored exactly from the stored vectors, `topK` × 3 of them, or × 10 in a scope whose graph was built on fewer dimensions than its vectors (`lux_embed index --graph-dims`), which orders its pool less well. `--rerank-factor` sets one factor for every scope. Each extra candidate is a row read, about 0.15–0.3 ms when the vectors are cold.
- **New graphs.** New graph generations written by `lux_embed index` are picked up within 10 s. A scope whose graph isn't built yet shows as unavailable in `/health`.
- **Concurrency.** One process. Searches run on FastAPI's worker threads without a USearch executor pool, because an idle pool busy-spins its workers. Index pages are served in place from the LMDB mmap (`view_ram_buffer`).
- **Names in joins.** LUX has one primary name per language, so join a single English/sort name, or aggregate with `SAMPLE`, to avoid one row per language.
