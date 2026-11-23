"""Tests for the vector SERVICE: the parser on QLever's exact request shape, and the endpoint.

Run with the vectors venv: ../vectors/.venv/bin/python -m pytest tests/test_sparql_service.py -q
"""

import json
import time

import numpy as np
import pytest

pytest.importorskip("usearch")
pytest.importorskip("lmdb")

from qleverlux.vectors.lmdb_vector_store import build_store  # noqa: E402
from sparql_service.sparql import SparqlError, parse  # noqa: E402

D = "https://lux.collections.yale.edu/data/"
PROLOGUE = "PREFIX lux: <https://lux.collections.yale.edu/ns/>\nPREFIX vec: <http://lux.collections.yale.edu/ns/vec#>\n"


def concept(i):
    return f"{D}concept/{i:08x}-0000-4000-8000-000000000000"


def item(i):
    return f"{D}object/{i:08x}-0000-4000-8000-000000000000"


def qlever_request(body, values=None):
    """What QLever 0.6.0 sends: prologue, `SELECT vars {`, the body, VALUES appended before '}'."""
    pushed = ""
    if values is not None:
        pushed = "\nVALUES (?c) { " + " ".join(f"(<{v}>)" for v in values) + " } "
    return f"{PROLOGUE}SELECT ?c ?score {{\n {body} {pushed}\n}}"


# -- parser ----------------------------------------------------------------------------------------


def test_parses_qlevers_request_with_pushed_values():
    plan = parse(qlever_request(f"?c vec:similarTo <{concept(1)}> ; vec:score ?score ; vec:topK 5 ", [concept(2), concept(3)]))
    assert plan.select == ["c", "score"] and plan.subject == "c" and plan.score == "score"
    assert [(t.kind, t.value) for t in plan.seeds] == [("iri", concept(1))] and plan.combine == "max"
    assert plan.top_k == 5 and plan.candidates == [concept(2), concept(3)]


def test_parses_text_queries_options_and_no_values():
    plan = parse(PROLOGUE + 'SELECT * WHERE { ?x vec:similarTo "dutch \\"golden age\\" painting"@en ; '
                 'vec:score ?s ; vec:rank ?r ; vec:index "work,item" ; vec:mode "global" . }')
    assert [(t.kind, t.value) for t in plan.seeds] == [("literal", 'dutch "golden age" painting')]
    assert plan.select == ["x", "s", "r"] and plan.scopes == ["work", "item"] and plan.mode == "global"
    assert plan.candidates is None


def test_several_seeds_by_list_or_repetition_and_combine():
    plan = parse(qlever_request(f'?c vec:similarTo <{concept(1)}>, <{concept(2)}>, "glaze" ; vec:similarTo <{concept(1)}> ; '
                                f'vec:score ?score ; vec:combine "mean"'))
    assert [t.value for t in plan.seeds] == [concept(1), concept(2), "glaze"] and plan.combine == "mean"
    with pytest.raises(SparqlError, match="vec:combine"):
        parse(qlever_request(f'?c vec:similarTo <{concept(1)}> ; vec:combine "min"'))


def test_large_values_block_parses_fast():
    values = [concept(i) for i in range(100_000)]
    query = qlever_request(f"?c vec:similarTo <{concept(1)}> ; vec:score ?score", values)
    started = time.perf_counter()
    plan = parse(query)
    assert len(plan.candidates) == 100_000 and plan.candidates[-1] == values[-1]
    assert time.perf_counter() - started < 1.0, "100k pushed IRIs must not take a general SPARQL parse"


def test_user_written_single_variable_values_and_undef():
    plan = parse(PROLOGUE + f"SELECT ?c ?score {{ VALUES ?c {{ <{concept(1)}> UNDEF <{concept(2)}> }} "
                 f"?c vec:similarTo <{concept(9)}> ; vec:score ?score }}")
    assert plan.candidates == [concept(1), concept(2)]


@pytest.mark.parametrize("body, message", [
    ("?c lux:broader ?p ; vec:similarTo <x:y>", "only vec: predicates"),
    ("?c vec:score ?score", "vec:similarTo is required"),
    ("?c vec:similarTo ?q", "IRIs or text literals"),
    ("?c vec:similarTo <x:y> . OPTIONAL { ?c vec:score ?s }", "only vec: triples"),
    ('?c vec:similarTo <x:y> ; vec:mode "sometimes"', "vec:mode"),
])
def test_refuses_what_it_cannot_evaluate(body, message):
    with pytest.raises(SparqlError, match=message):
        parse(qlever_request(body))


# -- endpoint --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from sparql_service.app import create_app

    path = str(tmp_path_factory.mktemp("store") / "vectors.lmdb")
    rng = np.random.default_rng(0)
    concepts = rng.standard_normal((200, 8)).astype(np.float32)
    items = concepts[:, :4] + 0.01 * rng.standard_normal((200, 4)).astype(np.float32)  # "Matryoshka" prefix
    build_store(path, [concept(i) for i in range(200)], concepts, scope="concept")
    build_store(path, [item(i) for i in range(200)], items, scope="item")
    app = create_app(path)
    app.state.concepts = concepts

    class FakeEncoder:  # stands in for EmbeddingGemma: the "text" is concept 7's vector
        def encode_query(self, text, dims):
            v = concepts[7][:dims]
            return v / np.linalg.norm(v)

    app.state.service._encoder = FakeEncoder()
    with TestClient(app) as test_client:
        yield test_client


def test_startup_opens_every_scope_before_serving(client):
    # The fixture's TestClient ran the lifespan: graphs are open and the encoder was used.
    assert set(client.app.state.service._stores) == {"concept", "item"}


def test_warming_reads_graphs_first_and_keeps_to_its_budget(tmp_path):
    from sparql_service.warm import warm

    path = str(tmp_path / "vectors.lmdb")
    rng = np.random.default_rng(1)
    build_store(path, [concept(i) for i in range(5000)], rng.standard_normal((5000, 64)).astype(np.float32),
                scope="concept")
    everything = warm(path, 1 << 40, threads=4)
    assert all(everything.values()), everything
    graphs_only = warm(path, everything["graphs"], threads=4)
    assert graphs_only == {"graphs": everything["graphs"], "IRI maps": 0, "vectors": 0}


def post(client, query):
    return client.post("/sparql", content=query, headers={"Content-Type": "application/sparql-query"})


def test_graph_first_scores_exactly_the_pushed_candidates(client):
    pushed = [concept(i) for i in (3, 7, 11, 50)]
    response = post(client, qlever_request(f"?c vec:similarTo <{concept(7)}> ; vec:score ?score ; vec:topK 3 ; vec:includeSeeds true", pushed))
    assert response.status_code == 200 and response.headers["content-type"].startswith("application/sparql-results+json")
    body = response.json()
    assert body["head"]["vars"] == ["c", "score"]
    rows = body["results"]["bindings"]
    assert len(rows) == 3 and rows[0]["c"]["value"] == concept(7)
    assert {r["c"]["value"] for r in rows} <= set(pushed)
    assert rows[0]["score"]["datatype"].endswith("#double") and float(rows[0]["score"]["value"]) == pytest.approx(1.0, abs=1e-4)
    assert float(rows[0]["score"]["value"]) >= float(rows[1]["score"]["value"]) >= float(rows[2]["score"]["value"])


def test_vector_first_uses_the_index(client):
    response = post(client, qlever_request(f"?c vec:similarTo <{concept(42)}> ; vec:score ?score ; vec:topK 1 ; vec:includeSeeds true"))
    assert response.json()["results"]["bindings"][0]["c"]["value"] == concept(42)
    rows = post(client, qlever_request(f"?c vec:similarTo <{concept(42)}> ; vec:score ?score ; vec:topK 5")).json()["results"]["bindings"]
    assert len(rows) == 5 and concept(42) not in {r["c"]["value"] for r in rows}  # seeds excluded, slots kept


def test_constrained_mode_refuses_a_missing_pushdown(client):
    response = post(client, qlever_request(f'?c vec:similarTo <{concept(1)}> ; vec:score ?score ; vec:mode "constrained"'))
    assert response.status_code == 400 and "service-max-value-rows" in response.json()["error"]


def test_cross_scope_query_slices_the_wider_vector(client):
    # A concept's 8d vector queries the 4d item index; item i was built from concept i's prefix.
    query = PROLOGUE + f'SELECT ?i ?score {{ ?i vec:similarTo <{concept(9)}> ; vec:score ?score ; vec:index "item" ; vec:topK 1 }}'
    assert post(client, query).json()["results"]["bindings"][0]["i"]["value"] == item(9)


def test_mixed_scope_candidates_are_grouped_and_merged(client):
    pushed = [concept(5), item(5), item(6)]
    values = " ".join(f"<{v}>" for v in pushed)
    query = PROLOGUE + (f"SELECT ?x ?score ?rank {{ ?x vec:similarTo <{concept(5)}> ; vec:score ?score ; "
                        f"vec:rank ?rank ; vec:includeSeeds true . VALUES ?x {{ {values} }} }}")
    rows = post(client, query).json()["results"]["bindings"]
    assert {r["x"]["value"] for r in rows} == set(pushed)
    assert [r["rank"]["value"] for r in rows] == ["1", "2", "3"]


def test_text_query_goes_through_the_encoder(client):
    query = PROLOGUE + 'SELECT ?c ?score { ?c vec:similarTo "anything" ; vec:score ?score ; vec:index "concept" ; vec:topK 1 }'
    assert post(client, query).json()["results"]["bindings"][0]["c"]["value"] == concept(7)


def test_repeated_requests_are_answered_from_the_cache(client):
    pushed = [concept(i) for i in range(40, 60)]
    query = qlever_request(f"?c vec:similarTo <{concept(41)}> ; vec:score ?score ; vec:topK 4", pushed)
    first, again = post(client, query), post(client, query)
    assert first.headers["X-Vector-Cache"] == "miss" and again.headers["X-Vector-Cache"] == "response"
    assert again.content == first.content
    # The same candidates in another order, under other variable names: same hits, own response.
    renamed = qlever_request(f"?x vec:similarTo <{concept(41)}> ; vec:score ?s ; vec:topK 4", pushed[::-1])
    renamed = renamed.replace("?c", "?x")
    reordered = post(client, renamed)
    assert reordered.headers["X-Vector-Cache"] == "hits"
    assert [row["x"]["value"] for row in reordered.json()["results"]["bindings"]] == \
        [row["c"]["value"] for row in first.json()["results"]["bindings"]]
    assert client.get("/cache").json()["hits"] >= 2


def test_text_is_encoded_once(client):
    service = client.app.state.service
    calls = []
    real = service._encoder
    class Counting:
        def encode_query(self, text, dims):
            calls.append(text)
            return real.encode_query(text, dims)
    service._encoder = Counting()
    try:
        for top_k in (2, 3):  # different plans, same text
            query = PROLOGUE + ('SELECT ?c ?score { ?c vec:similarTo "encoded once" ; vec:score ?score ; '
                                f'vec:index "concept" ; vec:topK {top_k} }}')
            assert post(client, query).headers["X-Vector-Cache"] == "miss"
    finally:
        service._encoder = real
    assert calls == ["encoded once"]


def test_concurrent_identical_requests_share_one_search(client):
    from concurrent.futures import ThreadPoolExecutor

    service = client.app.state.service
    runs, real_run = [], service.run
    def slow_run(plan):
        runs.append(plan)
        time.sleep(0.3)
        return real_run(plan)
    service.run = slow_run
    try:
        query = qlever_request(f"?c vec:similarTo <{concept(77)}> ; vec:score ?score ; vec:topK 5")
        with ThreadPoolExecutor(4) as pool:
            responses = list(pool.map(lambda _: post(client, query), range(4)))
    finally:
        del service.run
    assert len(runs) == 1 and {r.status_code for r in responses} == {200}
    assert len({r.content for r in responses}) == 1
    assert sorted(r.headers["X-Vector-Cache"] for r in responses) == ["miss", "shared", "shared", "shared"]


def test_a_new_generation_invalidates(tmp_path):
    from sparql_service.cache import ByteLRU

    cache = ByteLRU(100)
    cache.put("k", "v", 10, tag=("concept", 1))
    assert cache.get("k", ("concept", 1)) == "v"
    assert cache.get("k", ("concept", 2)) is None and cache.size == 0
    for i in range(12):
        cache.put(i, i, 10)
    assert cache.size == 100 and cache.get(0) is None and cache.get(11) == 11
    cache.put("huge", "x", 101)
    assert cache.get("huge") is None


def test_form_and_get_requests_and_errors(client):
    query = qlever_request(f"?c vec:similarTo <{concept(2)}> ; vec:score ?score ; vec:topK 1")
    assert client.post("/sparql", data={"query": query}).status_code == 200
    assert client.get("/sparql", params={"query": query}).status_code == 200
    bad = client.get("/sparql", params={"query": qlever_request(f"?c vec:similarTo <{D}concept/missing> ; vec:score ?s")})
    assert bad.status_code == 400 and "error" in bad.json()
    health = client.get("/health").json()
    assert set(health) == {"concept", "item"} and health["concept"]["entries"] == 200


def test_seeds_without_a_vector_are_skipped_unless_none_are_left(client):
    # similarRecords seeds come from a QLever query, which can name records newer than the vectors.
    missing = f"{D}concept/ffffffff-0000-4000-8000-000000000000"
    query = qlever_request(f"?c vec:similarTo <{missing}>, <{concept(4)}> ; vec:score ?score ; vec:topK 1 ; vec:includeSeeds true")
    response = post(client, query)
    assert response.status_code == 200
    assert response.json()["results"]["bindings"][0]["c"]["value"] == concept(4)
    alone = post(client, qlever_request(f"?c vec:similarTo <{missing}> ; vec:score ?score"))
    assert alone.status_code == 400 and "no vector" in alone.json()["error"]


def ranked(client, query):
    return [(r["c"]["value"], float(r["score"]["value"])) for r in post(client, query).json()["results"]["bindings"]]


def test_similar_to_any_of_several_records(client):
    seeds = f"<{concept(3)}>, <{concept(50)}>, <{concept(120)}>"
    hits = ranked(client, qlever_request(f"?c vec:similarTo {seeds} ; vec:score ?score ; vec:topK 3 ; vec:includeSeeds true"))
    assert {iri for iri, _ in hits} == {concept(3), concept(50), concept(120)}
    assert all(score == pytest.approx(1.0, abs=1e-4) for _, score in hits)


def test_combine_max_and_mean_match_brute_force(client):
    concepts = client.app.state.concepts
    unit = concepts / np.linalg.norm(concepts, axis=1, keepdims=True)
    seeds = [3, 4]
    everything = [concept(i) for i in range(200)]
    for combine, expected_scores in (
        ("max", (unit @ unit[seeds].T).max(axis=1)),
        ("mean", unit @ (unit[seeds].mean(axis=0) / np.linalg.norm(unit[seeds].mean(axis=0)))),
    ):
        body = (f'?c vec:similarTo <{concept(3)}>, <{concept(4)}> ; vec:score ?score ; vec:topK 5 ; '
                f'vec:combine "{combine}" ; vec:includeSeeds true')
        hits = ranked(client, qlever_request(body, everything))  # graph-first: exact over all 200
        order = np.argsort(-expected_scores)[:5]
        assert [iri for iri, _ in hits] == [concept(i) for i in order], combine
        assert [s for _, s in hits] == pytest.approx(expected_scores[order].tolist(), abs=1e-5)


def test_combine_each_interleaves_every_seeds_neighbours(client):
    concepts = client.app.state.concepts
    unit = concepts / np.linalg.norm(concepts, axis=1, keepdims=True)
    seeds, everything = [10, 20], [concept(i) for i in range(200)]
    body = f'?c vec:similarTo <{concept(10)}>, <{concept(20)}> ; vec:score ?score ; vec:rank ?rank ; vec:topK 6 ; vec:combine "each"'
    query = qlever_request(body, everything).replace("SELECT ?c ?score {", "SELECT ?c ?score ?rank {")
    rows = post(client, query).json()["results"]["bindings"]
    expected = []
    per_seed = [[i for i in np.argsort(-(unit @ unit[s])) if i not in seeds] for s in seeds]
    for a, b in zip(*per_seed):  # round-robin, first occurrence wins
        for i in (a, b):
            if i not in expected:
                expected.append(i)
    assert [r["c"]["value"] for r in rows] == [concept(i) for i in expected[:6]]
    assert [r["rank"]["value"] for r in rows] == [str(n) for n in range(1, 7)]


def test_min_score_and_max_gap_cut_the_top_k(client):
    everything = [concept(i) for i in range(200)]
    base = f"?c vec:similarTo <{concept(5)}> ; vec:score ?score ; vec:topK 20"
    full = ranked(client, qlever_request(base, everything))
    assert len(full) == 20
    floor = full[9][1]
    above = ranked(client, qlever_request(f"{base} ; vec:minScore {floor!r}", everything))
    assert above == [hit for hit in full if hit[1] >= floor] and len(above) >= 10
    gap = full[0][1] - full[4][1]
    close = ranked(client, qlever_request(f"{base} ; vec:maxGap {gap!r}", everything))
    assert close == [hit for hit in full if hit[1] >= full[0][1] - gap] and len(close) >= 5
    both = ranked(client, qlever_request(f"{base} ; vec:minScore {full[2][1]!r} ; vec:maxGap {gap!r}", everything))
    assert both == full[: len(both)] and len(both) == len([h for h in full if h[1] >= full[2][1]])
    nothing = ranked(client, qlever_request(f"{base} ; vec:minScore 1", everything))
    assert nothing == []


def test_cutoffs_apply_to_each_seeds_ranking_under_combine_each(client):
    everything = [concept(i) for i in range(200)]
    body = (f'?c vec:similarTo <{concept(10)}>, <{concept(20)}> ; vec:score ?score ; vec:topK 50 ; '
            f'vec:combine "each" ; vec:maxGap 0')
    hits = ranked(client, qlever_request(body, everything))
    assert len(hits) == 2  # each seed's best only


@pytest.mark.parametrize("option, message", [
    ("vec:minScore 2", "between -1 and 1"),
    ('vec:minScore "high"', "is a number"),
    ("vec:maxGap -0.1", "not negative"),
])
def test_refuses_bad_cutoffs(option, message):
    with pytest.raises(SparqlError, match=message):
        parse(qlever_request(f"?c vec:similarTo <{concept(1)}> ; {option}"))


def test_text_queries_use_the_model_the_store_was_built_with(tmp_path):
    from qleverlux.vectors.lmdb_vector_store import VectorWriter
    from sparql_service.app import LEGACY_MODEL, VectorService

    def scope(path, name, model):
        writer = VectorWriter(path, name, 4, model=model)
        writer.append([concept(1)], np.ones((1, 4), dtype=np.float32) / 2)
        writer.finish()
        writer.close()

    harrier = "microsoft/harrier-oss-v1-0.6b"
    path = str(tmp_path / "one.lmdb")
    scope(path, "concept", harrier)
    scope(path, "place", harrier)
    assert VectorService(path).model() == harrier

    old = str(tmp_path / "old.lmdb")  # built before the meta recorded a model: EmbeddingGemma 2
    build_store(old, [concept(i) for i in range(3)], np.eye(3, 4, dtype=np.float32), scope="concept")
    assert VectorService(old).model() == LEGACY_MODEL

    mixed = str(tmp_path / "mixed.lmdb")
    scope(mixed, "concept", harrier)
    scope(mixed, "place", LEGACY_MODEL)
    with pytest.raises(SparqlError, match="mixes encoders"):
        VectorService(mixed).model()
