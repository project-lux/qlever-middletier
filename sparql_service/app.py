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
scope must share one model, because one text query vector serves them all. The encoder loads on
first use, so IRI-only deployments never pay for it.

QLever has no SERVICE timeout, so the service enforces its own deadline. Searches run on worker
threads with no USearch executor pool (an idle pool busy-spins its workers).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.parse

import lmdb
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from qleverlux.vectors.lmdb_vector_store import MAX_DBS, LmdbVectorStore, _open_dbs, scopes_in

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


def scope_of(iri: str) -> str | None:
    if not iri.startswith(DATA):
        return None
    return SCOPE_OF_SEGMENT.get(iri[len(DATA):].split("/", 1)[0])


class VectorService:
    def __init__(self, store_path: str, *, device: str = "auto", refresh_seconds: float = 10.0,
                 exact_threshold: int = 200_000, expansion_factor: int = 128):
        self.path = store_path
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

    def describe(self) -> dict:
        out = {}
        for scope in self.scopes():
            try:
                out[scope] = self.store(scope).describe()
            except SparqlError as error:
                out[scope] = {"unavailable": str(error)}
        return out

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
        texts = [seed.value for seed in plan.seeds if seed.kind == "literal"]
        if texts:
            encoder = self.encoder()
            with self._encoder_lock:  # one forward pass at a time on the device
                encoded = {text: encoder.encode_query(text, widest) for text in texts}
        for seed in plan.seeds:
            if seed.kind == "literal":
                rows.append(encoded[seed.value])
                continue
            scope = scope_of(seed.value)
            if scope is None:
                raise SparqlError(f"vec:similarTo <{seed.value}> is not a LUX record IRI")
            vector = self.store(scope).get_vector(seed.value)
            if vector is None:
                raise SparqlError(f"<{seed.value}> has no vector in the {scope!r} index")
            if len(vector) < widest:
                raise SparqlError(f"<{seed.value}> has a {len(vector)}d vector, narrower than the {widest}d "
                                  f"index it would query; use a text query or an equally wide scope")
            vector = vector[:widest]
            rows.append(vector / np.linalg.norm(vector))
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

        if plan.combine != "each" or len(query) == 1:
            return search(query)[: plan.top_k]
        # "each": every seed's ranking, interleaved round-robin, so no seed is crowded out.
        rankings = [search(query[row : row + 1]) for row in range(len(query))]
        merged, seen = [], set()
        for position in range(max(map(len, rankings))):
            for ranking in rankings:
                if position < len(ranking) and ranking[position][0] not in seen:
                    seen.add(ranking[position][0])
                    merged.append(ranking[position])
        return merged[: plan.top_k]

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


def create_app(store_path: str, *, device: str = "auto", timeout: float = 30.0, **options) -> FastAPI:
    service = VectorService(store_path, device=device, **options)
    app = FastAPI(title="LUX vector SERVICE", docs_url=None, redoc_url=None)
    app.state.service = service

    def answer(query: str) -> tuple[dict, dict]:
        started = time.perf_counter()
        plan = parse(query)
        hits = service.run(plan)
        timing = {"candidates": len(plan.candidates) if plan.candidates is not None else None,
                  "ms": round((time.perf_counter() - started) * 1e3, 1)}
        return service.results(plan, hits), timing

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
            results, timing = await asyncio.wait_for(run_in_threadpool(answer, query), timeout)
        except SparqlError as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        except asyncio.TimeoutError:
            return JSONResponse({"error": f"vector search exceeded {timeout:g}s"}, status_code=503)
        return Response(json.dumps(results), media_type=RESULTS_JSON,
                        headers={"X-Vector-Candidates": str(timing["candidates"]), "X-Vector-Ms": str(timing["ms"])})

    @app.get("/health")
    async def health() -> dict:
        return await run_in_threadpool(service.describe)

    return app
