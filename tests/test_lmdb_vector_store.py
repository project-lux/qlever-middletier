"""
Regression tests for qleverlux.vectors.lmdb_vector_store.

The HNSW walk returns at most ``expansion_factor`` hits and leaves the rest of
its key buffer uninitialized. Converting the whole buffer made any request for
more hits than that raise ("an identity using its high eight bytes has no 64-bit
integer form"); the sidecar turned that into a 500 for topK >= 43 on concepts.

Needs numpy, lmdb and USearch 3; skipped where they are not installed (the
middle tier's own environment does not need them). Run with the vectors venv:

    ../vectors/.venv/bin/python -m pytest tests/test_lmdb_vector_store.py
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("lmdb")
pytest.importorskip("usearch")

from qleverlux.vectors import lmdb_vector_store as lvs  # noqa: E402

N, DIM = 3000, 32


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    rng = np.random.default_rng(7)
    data = rng.standard_normal((N, DIM)).astype(np.float32)
    iris = [f"https://example.org/data/concept/{i}" for i in range(N)]
    path = str(tmp_path_factory.mktemp("vec") / "store.lmdb")
    lvs.build_store(path, iris, data, scope="concept", chunk_rows=1024)
    s = lvs.LmdbVectorStore(path, "concept")
    yield s, data, iris
    s.close()


def _exact_top(data, q, k):
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    return np.argsort(-(unit @ (q / np.linalg.norm(q))))[:k]


@pytest.mark.parametrize("top_k, expansion", [(10, 128), (43, 128), (100, 128), (400, 16)])
def test_search_global_returns_full_top_k(store, top_k, expansion):
    s, data, iris = store
    hits = s.search_global(data[0], top_k, expansion_factor=expansion)
    assert len(hits) == top_k
    assert hits[0][0] == iris[0]  # the seed's own vector is its best match
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_global_several_queries(store):
    s, data, _ = store
    assert len(s.search_global(data[:3], 200, expansion_factor=16)) == 200


def test_search_candidates_filtered_walk_past_expansion(store):
    """Above exact_threshold the candidates go through the filtered HNSW walk."""
    s, data, iris = store
    allowed = iris[::2]
    hits = s.search_candidates(data[0], allowed, 100, exact_threshold=10, expansion_factor=16)
    assert len(hits) == 100
    assert {iri for iri, _ in hits} <= set(allowed)
    assert hits[0][0] == iris[0]


def test_search_candidates_exact_matches_brute_force(store):
    s, data, iris = store
    hits = s.search_candidates(data[5], iris, 20)
    assert [iri for iri, _ in hits] == [iris[i] for i in _exact_top(data, data[5], 20)]


@pytest.mark.parametrize("in_place", [True, False])
def test_open_store_does_not_spin_when_idle(store, monkeypatch, in_place):
    """An undrained usearch Executor busy-spins its workers (about two cores
    each), so a store must not keep one alive. The sidecar holds four stores
    open and was burning ~9 cores with no traffic."""
    import resource
    import time

    s, _, _ = store
    if in_place and not lvs.CAN_VIEW:
        pytest.skip("this USearch build has no view_ram_buffer")
    monkeypatch.setattr(lvs, "CAN_VIEW", in_place)
    opened = lvs.LmdbVectorStore(None, "concept", env=s.env)

    def cpu():
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime

    before = cpu()
    time.sleep(1.0)
    assert cpu() - before < 0.3
    assert opened.describe()["in_place"] is in_place
