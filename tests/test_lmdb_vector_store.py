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


def test_search_exact_matches_brute_force_across_chunks(store):
    s, data, iris = store  # 3000 rows in 1024-row chunks, scanned in blocks smaller than a chunk
    q = np.random.default_rng(3).standard_normal(DIM).astype(np.float32)
    hits = s.search_exact(q, 25, block_rows=500)
    assert [iri for iri, _ in hits] == [iris[i] for i in _exact_top(data, q, 25)]
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    assert [score for _, score in hits] == pytest.approx(
        (unit @ (q / np.linalg.norm(q)))[_exact_top(data, q, 25)].tolist(), abs=1e-5)


def test_rewalk_never_loses_what_one_walk_found(store):
    """The second walks add to the pool and the pool is reranked exactly, so the result is at
    least as good at every rank - measured on queries away from every record, like text."""
    s, data, iris = store
    rng = np.random.default_rng(11)
    exact_hits = 0
    for q in rng.standard_normal((20, DIM)).astype(np.float32):
        truth = {iris[i] for i in _exact_top(data, q, 20)}
        once = s.search_global(q, 20, expansion_factor=16)
        twice = s.search_global(q, 20, expansion_factor=16, rewalk=5)
        narrow = s.search_global(q, 20, expansion_factor=16, rewalk=5, rewalk_expansion_factor=1)
        assert all(b >= a - 1e-6 for (_, a), (_, b) in zip(once, twice))
        assert all(b >= a - 1e-6 for (_, a), (_, b) in zip(once, narrow))
        exact_hits += len(truth & {iri for iri, _ in twice}) - len(truth & {iri for iri, _ in once})
    assert exact_hits >= 0


def _through_lmdb(s, ids):
    """The rows as py-lmdb returns them, whole chunks at a time: what _vectors must equal."""
    reader, s._rows = s._rows, None
    try:
        with s.env.begin(buffers=True) as txn:
            return s._vectors(txn, ids)
    finally:
        s._rows = reader


def test_rows_are_read_straight_from_the_file(store):
    s, data, _ = store
    assert s._rows is not None  # 1024-row chunks are overflow values: the layout check passed
    rng = np.random.default_rng(5)
    for ids in (
        [0], [N - 1], [5, 6, 7, 8, 1030, 1031, 2999],     # runs within and across chunks
        rng.choice(N, 40, replace=False),                  # scattered: one read per row
        rng.choice(N, 900, replace=False),                 # most of each chunk: whole chunks
        [17, 17, 3, 17],                                   # repeats, out of order
    ):
        ids = np.asarray(ids, dtype=np.uint64)
        with s.env.begin(buffers=True) as txn:
            got = s._vectors(txn, ids)
        np.testing.assert_array_equal(got, _through_lmdb(s, ids))
        np.testing.assert_array_equal(got, data[ids.astype(np.int64)])


def test_finding_a_chunk_does_not_read_it(store, monkeypatch):
    """Only the rows asked for are read, by pread: py-lmdb's get (which reads a whole chunk) is
    not called for the vectors."""
    s, data, iris = store
    spans = []
    real = s._rows.read
    monkeypatch.setattr(s._rows, "read", lambda sp: spans.extend(sp) or real(sp))
    vector = s.get_vector(iris[1500])
    np.testing.assert_array_equal(vector, data[1500])
    assert spans == [(spans[0][0], DIM * 4)]  # one row of float32


def test_a_layout_that_fails_the_check_is_read_as_before(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    data = rng.standard_normal((300, DIM)).astype(np.float32)
    iris = [f"https://example.org/data/concept/{i}" for i in range(300)]
    path = str(tmp_path / "store.lmdb")
    lvs.build_store(path, iris, data, scope="concept", chunk_rows=128)
    monkeypatch.setattr(lvs._RowReader, "verify", lambda self, cursor, key: False)
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s._rows is None
        np.testing.assert_array_equal(s.get_vector(iris[200]), data[200])
        assert s.search_global(data[3], 5)[0][0] == iris[3]
    finally:
        s.close()


def test_small_inline_chunks_are_read_through_lmdb(tmp_path):
    rng = np.random.default_rng(10)
    data = rng.standard_normal((50, 4)).astype(np.float32)  # 3-row chunks of 4 floats: stored inline
    iris = [f"https://example.org/data/concept/{i}" for i in range(50)]
    path = str(tmp_path / "small.lmdb")
    lvs.build_store(path, iris, data, scope="concept", chunk_rows=3)
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s._rows is None  # nothing to check the layout against
        with s.env.begin(buffers=True) as txn:
            np.testing.assert_array_equal(s._vectors(txn, np.array([0, 10, 49], dtype=np.uint64)), data[[0, 10, 49]])
    finally:
        s.close()
