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


def test_form_and_get_requests_and_errors(client):
    query = qlever_request(f"?c vec:similarTo <{concept(2)}> ; vec:score ?score ; vec:topK 1")
    assert client.post("/sparql", data={"query": query}).status_code == 200
    assert client.get("/sparql", params={"query": query}).status_code == 200
    bad = client.get("/sparql", params={"query": qlever_request(f"?c vec:similarTo <{D}concept/missing> ; vec:score ?s")})
    assert bad.status_code == 400 and "error" in bad.json()
    health = client.get("/health").json()
    assert set(health) == {"concept", "item"} and health["concept"]["entries"] == 200


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
