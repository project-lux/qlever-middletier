"""The vector SERVICE endpoint QLever calls: ``POST /sparql`` -> ``application/sparql-results+json``.

Execution follows the plan :mod:`sparql_service.sparql` parses:

* **graph-first** - QLever pushed the graph-constrained candidates as ``VALUES``: they are grouped
  by record class (scope) and scored exactly against the query vector;
* **vector-first** - no ``VALUES``: HNSW top-K per scope, exact rerank, merged by score;
* ``vec:mode "constrained"`` refuses vector-first, because a missing ``VALUES`` block can mean
  QLever's ``service-max-value-rows`` cut the pushdown off - and a global top-K joined against a
  large constraint silently returns almost nothing.

The query vector is an IRI's stored vector (from its own scope, sliced Matryoshka-style when the
target index is narrower) or free text encoded with the model the store was built with (its meta
records it; stores from before that are EmbeddingGemma 2), using that model's query prompt. Every
scope must share one model, because one text query vector serves them all.

Nothing is loaded lazily: before the port opens, :meth:`VectorService.warm` reads the live
store into the page cache (graphs, then IRI maps, then vectors, within a budget), opens every
scope's graph, loads the encoder and runs one encode and one search per scope. Done on the first request instead, the model load alone outlasts the deadline, so the
first few queries after a restart fail.

Results are cached, because one search runs the same SERVICE call several times: the facet
queries all repeat the search's pattern, so QLever sends the same body once per facet, mostly at
the same moment. Three levels share one byte budget (:class:`~sparql_service.cache.ByteLRU`):
the rendered response by a hash of the body; the ranked hits by the plan, with the pushed-down
candidates hashed as a sorted set so the same candidates in another order still hit; and encoded
query text. Identical requests in flight share one computation rather than all missing at once,
and one that outlives its deadline still finishes and fills the cache for the retry. Every entry
is tagged with the scopes' graph generations, so a new graph invalidates it.

QLever has no SERVICE timeout, so the service enforces its own deadline. Searches run on worker
threads with no USearch executor pool (an idle pool busy-spins its workers).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from contextlib import asynccontextmanager

import lmdb
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from qleverlux.vectors.lmdb_vector_store import MAX_DBS, LmdbVectorStore, _open_dbs, scopes_in

from .cache import ByteLRU
from .sparql import XSD, Plan, SparqlError, parse

DATA = "https://lux.collections.yale.edu/data/"
# IRI path segment -> scope; LUX mints no /data/event/ (events and periods are activities).
SCOPE_OF_SEGMENT = {
    "object": "item", "digital": "item", "visual": "work", "text": "work", "set": "set",
    "person": "agent", "group": "agent", "place": "place", "concept": "concept",
    "activity": "event", "event": "event", "period": "event",
}
RESULTS_JSON = "application/sparql-results+json"
LEGACY_MODEL = "google/embeddinggemma-2"  # what stores built before the meta recorded a model used
log = logging.getLogger("uvicorn.error")


def physical_memory() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 1 << 62


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def plan_key(plan: Plan) -> tuple:
    """Everything about a plan that decides its hits - not the variable names, which only shape
    the response - with the candidates as a sorted set."""
    candidates = None if plan.candidates is None else (len(plan.candidates), digest("\n".join(sorted(plan.candidates))))
    return ("hits", tuple((seed.kind, seed.value) for seed in plan.seeds), plan.top_k,
            tuple(plan.scopes) if plan.scopes is not None else None, plan.mode, plan.combine,
            plan.include_seeds, plan.min_score, plan.max_gap, candidates)


def scope_of(iri: str) -> str | None:
    if not iri.startswith(DATA):
        return None
    return SCOPE_OF_SEGMENT.get(iri[len(DATA):].split("/", 1)[0])


class VectorService:
    def __init__(self, store_path: str, *, device: str = "auto", refresh_seconds: float = 10.0,
                 exact_threshold: int = 200_000, expansion_factor: int = 128, cache_bytes: int = 256 << 20):
        self.path = store_path
        self.cache = ByteLRU(cache_bytes)
        self.env = lmdb.open(store_path, readonly=True, max_dbs=MAX_DBS, readahead=False, subdir=True, map_size=1 << 40)
        self.device, self.refresh_seconds = device, refresh_seconds
        self.exact_threshold, self.expansion_factor = exact_threshold, expansion_factor
        self._stores: dict[str, LmdbVectorStore] = {}
        self._checked: dict[str, float] = {}
        self._lock = threading.Lock()
        self._encoder = None
        self._encoder_lock = threading.Lock()

    # -- stores ---------------------------------------------------------------------------------

    def scopes(self) -> list[str]:
        return scopes_in(self.env)

    def store(self, scope: str) -> LmdbVectorStore:
        """The scope's reader, opened on first use and refreshed (new graph generation) at most
        every ``refresh_seconds``."""
        now = time.monotonic()
        with self._lock:
            store = self._stores.get(scope)
            if store is None:
                if scope not in self.scopes():
                    raise SparqlError(f"no index for scope {scope!r}; have {', '.join(self.scopes()) or 'none'}")
                try:
                    store = self._stores[scope] = LmdbVectorStore(self.path, scope, env=self.env)
                except LookupError as error:  # vectors written, graph not built yet
                    raise SparqlError(str(error)) from None
                self._checked[scope] = now
            elif now - self._checked.get(scope, 0) > self.refresh_seconds:
                store.refresh()
                self._checked[scope] = now
            return store

    def generations(self) -> tuple:
        """The graph generation of every scope: the tag cached results are checked against."""
        out = []
        for scope in self.scopes():
            try:
                out.append((scope, self.store(scope).meta["generation"]))
            except SparqlError:
                out.append((scope, None))
        return tuple(out)

    def describe(self) -> dict:
        out = {}
        for scope in self.scopes():
            try:
                out[scope] = self.store(scope).describe()
            except SparqlError as error:
                out[scope] = {"unavailable": str(error)}
        return out

    def warm(self, budget: int | None = None) -> None:
        """Pays every first-query cost up front: the store's pages, each scope's graph, the
        encoder (load plus the first forward pass, which compiles kernels on MPS/CUDA) and one
        search per scope. Raises if the encoder cannot load - a sidecar that cannot encode
        text answers every `similar` with an error, so it should not come up looking healthy.

        The pages are read by a child process (:mod:`sparql_service.warm` says why) while the
        encoder loads here; ``budget`` bytes at most, half of physical memory by default."""
        if budget is None:
            budget = physical_memory() // 2
        log.info("vector store: warming up to %.1f GB, graphs first", budget / 1e9)
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        child = subprocess.Popen(
            [sys.executable, "-m", "sparql_service.warm", self.path, "--budget", str(budget)],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [package_root, os.environ.get("PYTHONPATH")]))},
        )
        try:
            started = time.perf_counter()
            log.info("loading encoder %s (device %s)", self.model(), self.device)
            encoder = self.encoder()
            log.info("encoder loaded on %s in %.1fs", getattr(encoder, "device", self.device), time.perf_counter() - started)
            if child.wait():
                log.warning("vector store: warming failed (exit %d); searches will read from disk", child.returncode)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

        stores = {}
        for scope in self.scopes():
            try:
                stores[scope] = self.store(scope)
            except SparqlError as error:
                log.warning("vector store: %s unavailable: %s", scope, error)
        if not stores:
            log.warning("vector store: no scope has a graph; nothing to search")
            return

        started = time.perf_counter()
        widest = max(store.meta["dimensions"] for store in stores.values())
        with self._encoder_lock:
            vector = encoder.encode_query("warm-up", widest)
        for store in stores.values():
            store.search_global(vector, 10, expansion_factor=self.expansion_factor)
        log.info("first encode and search on %s took %.2fs", ", ".join(stores), time.perf_counter() - started)

    # -- query vectors --------------------------------------------------------------------------

    def models(self) -> dict[str, str]:
        """The encoder each scope's vectors were built with, from its meta (graph or not)."""
        out = {}
        for scope in self.scopes():
            dbs = _open_dbs(self.env, scope, create=False)
            with self.env.begin() as txn:
                raw = txn.get(b"meta", db=dbs["meta"])
            out[scope] = (json.loads(bytes(raw)).get("model") if raw else None) or LEGACY_MODEL
        return out

    def model(self) -> str:
        """The one encoder text queries need: every scope's, which must be the same."""
        models = self.models()
        distinct = sorted(set(models.values()))
        if len(distinct) > 1:
            raise SparqlError("this store mixes encoders ("
                              + ", ".join(f"{scope}: {model}" for scope, model in sorted(models.items()))
                              + "); a text query is encoded once for every scope, so rebuild them with one model")
        return distinct[0] if distinct else LEGACY_MODEL

    def encoder(self):
        with self._encoder_lock:
            if self._encoder is None:
                from lux_embed.encoder import Encoder

                self._encoder = Encoder(device=self.device, model=self.model())
            return self._encoder

    def query_vectors(self, plan: Plan, widest: int) -> np.ndarray:
        """One row per seed at the widest target dimension (IRIs from their own scope's stored
        vectors, sliced Matryoshka-style; text through the encoder), or their centroid for
        ``vec:combine "mean"``."""
        rows = []
        texts = list(dict.fromkeys(seed.value for seed in plan.seeds if seed.kind == "literal"))
        if texts:
            model = self.model()
            encoded = {text: self.cache.get(("text", model, widest, text)) for text in texts}
            todo = [text for text, vector in encoded.items() if vector is None]
            if todo:
                encoder = self.encoder()
                with self._encoder_lock:  # one forward pass at a time on the device
                    for text in todo:
                        vector = encoded[text] = np.asarray(encoder.encode_query(text, widest), dtype=np.float32)
                        vector.setflags(write=False)  # shared through the cache
                        self.cache.put(("text", model, widest, text), vector, vector.nbytes + len(text) + 200)
        missing = []
        for seed in plan.seeds:
            if seed.kind == "literal":
                rows.append(encoded[seed.value])
                continue
            scope = scope_of(seed.value)
            if scope is None:
                raise SparqlError(f"vec:similarTo <{seed.value}> is not a LUX record IRI")
            vector = self.store(scope).get_vector(seed.value)
            if vector is None:
                # Seeds the store has no vector for (records newer than the vectors, or
                # none of their own) are skipped, so one gap does not sink a query
                # seeded by many records. It is only an error when none are left.
                missing.append(f"<{seed.value}> ({scope})")
                continue
            if len(vector) < widest:
                raise SparqlError(f"<{seed.value}> has a {len(vector)}d vector, narrower than the {widest}d "
                                  f"index it would query; use a text query or an equally wide scope")
            vector = vector[:widest]
            rows.append(vector / np.linalg.norm(vector))
        if not rows:
            raise SparqlError(f"no vector in the index for {', '.join(missing[:3])}"
                              + (f" or {len(missing) - 3} more seeds" if len(missing) > 3 else ""))
        matrix = np.vstack(rows).astype(np.float32)
        if plan.combine == "mean" and len(matrix) > 1:
            centroid = matrix.mean(axis=0)
            matrix = (centroid / np.linalg.norm(centroid))[None, :]
        return matrix

    # -- execution ------------------------------------------------------------------------------

    def run(self, plan: Plan) -> list[tuple[str, float]]:
        if plan.candidates is None and plan.mode == "constrained":
            raise SparqlError(
                "vec:mode \"constrained\" but no VALUES arrived: the graph constraint was probably larger than "
                "QLever's service-max-value-rows, so it was not pushed down. Narrow the constraint, raise "
                "service-max-value-rows, or rank the candidates in the middle tier.")
        by_scope: dict[str, list[str]] = {}
        if plan.candidates is not None:
            for iri in plan.candidates:
                scope = scope_of(iri)
                if scope and (plan.scopes is None or scope in plan.scopes):
                    by_scope.setdefault(scope, []).append(iri)
            if not by_scope:
                return []
        else:
            seed_scopes = list(dict.fromkeys(scope_of(seed.value) for seed in plan.seeds if seed.kind == "iri"))
            scopes = plan.scopes or [scope for scope in seed_scopes if scope]
            if not scopes:
                raise SparqlError("vec:index is required for a text query without pushed-down candidates")
            by_scope = {scope: None for scope in scopes}
        stores = {scope: self.store(scope) for scope in by_scope}
        query = self.query_vectors(plan, max(s.meta["dimensions"] for s in stores.values()))
        excluded = set() if plan.include_seeds else {seed.value for seed in plan.seeds if seed.kind == "iri"}
        wanted = plan.top_k + len(excluded)  # seeds would otherwise use up result slots

        def search(queries: np.ndarray) -> list[tuple[str, float]]:
            hits: list[tuple[str, float]] = []
            for scope, candidates in by_scope.items():
                store = stores[scope]
                if candidates is None:
                    hits += store.search_global(queries, wanted, expansion_factor=self.expansion_factor)
                else:
                    hits += store.search_candidates(queries, candidates, wanted, exact_threshold=self.exact_threshold)
            hits.sort(key=lambda hit: -hit[1])
            return [hit for hit in hits if hit[0] not in excluded]

        def cut(hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
            """vec:minScore and vec:maxGap, on a ranking already cut to top-K."""
            floor = -2.0 if plan.min_score is None else plan.min_score
            if plan.max_gap is not None and hits:
                floor = max(floor, hits[0][1] - plan.max_gap)
            return [hit for hit in hits if hit[1] >= floor]

        if plan.combine != "each" or len(query) == 1:
            return cut(search(query)[: plan.top_k])
        # "each": every seed's ranking, interleaved round-robin, so no seed is crowded out.
        rankings = [cut(search(query[row : row + 1])) for row in range(len(query))]
        merged, seen = [], set()
        for position in range(max(map(len, rankings), default=0)):
            for ranking in rankings:
                if position < len(ranking) and ranking[position][0] not in seen:
                    seen.add(ranking[position][0])
                    merged.append(ranking[position])
        return merged[: plan.top_k]

    def run_cached(self, plan: Plan, tag: tuple) -> tuple[list[tuple[str, float]], str]:
        """:meth:`run`, through the cache; also says whether it was a hit."""
        key = plan_key(plan)
        hits = self.cache.get(key, tag)
        if hits is not None:
            return hits, "hits"
        hits = self.run(plan)
        self.cache.put(key, hits, sum(len(iri) + 120 for iri, _ in hits) + 200, tag)
        return hits, "miss"

    @staticmethod
    def results(plan: Plan, hits: list[tuple[str, float]]) -> dict:
        bindings = []
        for rank, (iri, score) in enumerate(hits, 1):
            row = {plan.subject: {"type": "uri", "value": iri}}
            if plan.score:
                row[plan.score] = {"type": "literal", "datatype": XSD + "double", "value": repr(float(score))}
            if plan.rank:
                row[plan.rank] = {"type": "literal", "datatype": XSD + "integer", "value": str(rank)}
            bindings.append({var: value for var, value in row.items() if var in plan.select})
        # QLever requires head.vars to equal the SERVICE's visible variables exactly.
        return {"head": {"vars": plan.select}, "results": {"bindings": bindings}}


def create_app(store_path: str, *, device: str = "auto", timeout: float = 30.0, warm: bool = True,
               warm_bytes: int | None = None, **options) -> FastAPI:
    service = VectorService(store_path, device=device, **options)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # uvicorn opens the port only after this returns, so QLever is refused rather than
        # timed out while the model loads.
        if warm:
            await run_in_threadpool(service.warm, warm_bytes)
        yield

    app = FastAPI(title="LUX vector SERVICE", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.service = service

    def answer(query: str, key: bytes) -> tuple[bytes, dict]:
        started = time.perf_counter()
        tag = service.generations()
        cached = service.cache.get(("response", key), tag)
        if cached is not None:
            (body, candidates), source = cached, "response"
        else:
            plan = parse(query)
            hits, source = service.run_cached(plan, tag)
            body = json.dumps(service.results(plan, hits)).encode("utf-8")
            candidates = len(plan.candidates) if plan.candidates is not None else None
            service.cache.put(("response", key), (body, candidates), len(body) + 200, tag)
        return body, {"X-Vector-Candidates": str(candidates), "X-Vector-Cache": source,
                      "X-Vector-Ms": str(round((time.perf_counter() - started) * 1e3, 1))}

    inflight: dict[bytes, asyncio.Future] = {}

    def settled(key: bytes, future: asyncio.Future) -> None:
        inflight.pop(key, None)
        if not future.cancelled():
            future.exception()  # retrieved, so an error nobody waited for is not logged as lost

    async def shared_answer(query: str) -> tuple[bytes, dict]:
        """One computation per distinct body in flight. It runs to completion even when every
        waiter has timed out, so the retry finds it cached."""
        key = digest(query)
        future = inflight.get(key)
        if future is not None:
            body, headers = await asyncio.wait_for(asyncio.shield(future), timeout)
            return body, {**headers, "X-Vector-Cache": "shared"}
        future = inflight[key] = asyncio.ensure_future(run_in_threadpool(answer, query, key))
        future.add_done_callback(lambda done: settled(key, done))
        return await asyncio.wait_for(asyncio.shield(future), timeout)

    async def read_query(request: Request) -> str:
        if request.method == "GET":
            return request.query_params.get("query", "")
        body = (await request.body()).decode("utf-8")
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype == "application/x-www-form-urlencoded":
            return urllib.parse.parse_qs(body).get("query", [""])[0]
        return body  # application/sparql-query, which QLever sends

    @app.api_route("/sparql", methods=["GET", "POST"])
    async def sparql(request: Request) -> Response:
        query = await read_query(request)
        if not query.strip():
            return JSONResponse({"error": "no query"}, status_code=400)
        try:
            body, headers = await shared_answer(query)
        except SparqlError as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        except asyncio.TimeoutError:
            return JSONResponse({"error": f"vector search exceeded {timeout:g}s"}, status_code=503)
        return Response(body, media_type=RESULTS_JSON, headers=headers)

    @app.get("/health")
    async def health() -> dict:
        return await run_in_threadpool(service.describe)

    @app.get("/cache")
    async def cache() -> dict:
        return service.cache.describe()

    return app
