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

import json

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


@pytest.mark.parametrize("kind", ["I8", "U8", "U1X8", "I4X2"])
def test_build_index_refuses_kinds_it_cannot_feed(tmp_path, kind):
    data = np.random.default_rng(2).standard_normal((50, 8)).astype(np.float32)
    path = str(tmp_path / "store.lmdb")
    iris = [f"https://example.org/data/concept/{i}" for i in range(50)]
    with pytest.raises(ValueError, match="cannot build"):
        lvs.build_store(path, iris, data, scope="concept", quantization=getattr(lvs.Scalar, kind))
    lvs.build_store(path, iris, data, scope="concept")
    with pytest.raises(ValueError, match="cannot build"):
        lvs.build_index(path, "concept", quantization=getattr(lvs.Scalar, kind))


def test_build_index_builds_an_e4m3_graph_the_store_serves(tmp_path):
    rng = np.random.default_rng(4)
    data = rng.standard_normal((2000, 32)).astype(np.float32)
    iris = [f"https://example.org/data/concept/{i}" for i in range(2000)]
    path = str(tmp_path / "store.lmdb")
    lvs.build_store(path, iris, data, scope="concept")
    assert lvs.build_index(path, "concept", quantization=lvs.Scalar.E4M3)["quantization"] == "E4M3"
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s.search_global(data[7], 5, expansion_factor=200)[0][0] == iris[7]
    finally:
        s.close()


def test_a_graph_on_fewer_dimensions_walks_narrow_and_scores_full(tmp_path):
    rng = np.random.default_rng(5)
    data = rng.standard_normal((2000, 32)).astype(np.float32)
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    iris = [f"https://example.org/data/concept/{i}" for i in range(2000)]
    path = str(tmp_path / "store.lmdb")
    lvs.build_store(path, iris, data, scope="concept")
    meta = lvs.build_index(path, "concept", quantization=lvs.Scalar.E4M3, graph_dimensions=16)
    assert (meta["graph_dimensions"], meta["dimensions"]) == (16, 32)
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s.describe()["graph_dimensions"] == 16
        # Even with no rerank asked for, the scores are full-width cosines, not the graph's.
        hits = s.search_global(data[7], 5, rerank_factor=1, expansion_factor=200)
        assert hits[0][0] == iris[7]
        for iri, score in hits:
            assert score == pytest.approx(float(unit[iris.index(iri)] @ unit[7]), abs=1e-5)
        # A narrow graph reranks a bigger pool by default.
        wanted = []
        walk = s._walk
        s._walk = lambda graph, queries, n, ef: wanted.append(n) or walk(graph, queries, n, ef)
        s.search_global(data[7], 5, expansion_factor=200)
        s.search_global(data[7], 5, rerank_factor=2, expansion_factor=200)
        del s._walk
        assert wanted == [5 * lvs.NARROW_RERANK_FACTOR, 10]
        # The filtered walk takes the same narrow query.
        candidates = iris[:1500]
        found = s.search_candidates(data[3], candidates, 5, exact_threshold=10, expansion_factor=400)
        assert found[0][0] == iris[3]
    finally:
        s.close()
    # Rebuilding without it goes back to the full width.
    assert "graph_dimensions" not in lvs.build_index(path, "concept")


def _pca(data, width):
    mean = data.mean(axis=0)
    basis = np.linalg.svd(data - mean, full_matrices=False)[2][:width].T
    return mean.astype(np.float32), np.ascontiguousarray(basis, dtype=np.float32)


def test_a_projected_graph_walks_projected_and_scores_full(tmp_path):
    rng = np.random.default_rng(11)
    # Clustered, so the leading principal directions carry the neighbourhoods.
    centres = rng.standard_normal((40, 32)).astype(np.float32) * 3
    data = centres[rng.integers(0, 40, 2000)] + rng.standard_normal((2000, 32)).astype(np.float32)
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    iris = [f"https://example.org/data/concept/{i}" for i in range(2000)]
    path = str(tmp_path / "store.lmdb")
    lvs.build_store(path, iris, data, scope="concept")
    mean, basis = _pca(data, 12)
    meta = lvs.build_index(path, "concept", projection=(mean, basis), projection_info={"file": "pca12.npz"})
    assert (meta["graph_dimensions"], meta["dimensions"]) == (12, 32)
    assert meta["graph_projection"] == {"file": "pca12.npz", "dimensions": 12}
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s.describe()["projection"]["file"] == "pca12.npz"
        # The graph sees projected rows: a query walks as (q - mean) @ basis, renormalised.
        seen = []
        rows = s._graph.rows
        s._graph.rows = lambda v: seen.append(v) or rows(v)
        hits = s.search_global(data[7], 10, rerank_factor=1, expansion_factor=200)
        del s._graph.rows
        projected = (data[7] - mean) @ basis
        assert np.allclose(rows(seen[0][:1]), projected / np.linalg.norm(projected), atol=1e-5)
        # Scores are full-width cosines, and the top hit is the record itself.
        assert hits[0][0] == iris[7]
        for iri, score in hits:
            assert score == pytest.approx(float(unit[iris.index(iri)] @ unit[7]), abs=1e-5)
        # With the default (narrow) rerank pool, recall against exact search is high.
        recall = []
        for row in rng.choice(2000, 50, replace=False):
            truth = set(np.argsort(-(unit @ unit[row]))[:10].tolist())
            found = {iris.index(iri) for iri, _ in s.search_global(data[row], 10, expansion_factor=200)}
            recall.append(len(truth & found) / 10)
        assert np.mean(recall) >= 0.95
        # Rewalk seeds and filtered walks go through the projection too.
        assert s.search_global(data[3], 5, expansion_factor=200, rewalk=3)[0][0] == iris[3]
        assert s.search_candidates(data[3], iris[:1500], 5, exact_threshold=10, expansion_factor=400)[0][0] == iris[3]
    finally:
        s.close()

    # Rebuilding without it goes back to the full width and drops the stored projection. (A writer
    # is another process in practice; py-lmdb opens a path once per process, so not concurrently here.)
    meta = lvs.build_index(path, "concept")
    assert "graph_dimensions" not in meta and "graph_projection" not in meta
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s.describe()["projection"] is None and s._graph.projection is None
        with s.env.begin() as txn:
            assert txn.get(lvs.PROJECTION_KEY, db=s.dbs["graph"]) is None
        assert s.search_global(data[7], 1, expansion_factor=200)[0][0] == iris[7]
    finally:
        s.close()


def test_a_projection_must_fit_the_vectors_and_their_model(tmp_path):
    data = np.random.default_rng(12).standard_normal((200, 8)).astype(np.float32)
    iris = [f"https://example.org/data/concept/{i}" for i in range(200)]
    path = str(tmp_path / "store.lmdb")
    lvs.build_store(path, iris, data, scope="concept")
    mean, basis = _pca(data, 4)
    with pytest.raises(ValueError, match="does not fit"):
        lvs.build_index(path, "concept", projection=(mean[:7], basis))
    with pytest.raises(ValueError, match="does not fit"):
        lvs.build_index(path, "concept", projection=(mean, np.ones((8, 9), np.float32)))
    with pytest.raises(ValueError, match="not both"):
        lvs.build_index(path, "concept", projection=(mean, basis), graph_dimensions=4)
    # A map fitted to one model's vectors is refused for another's (meta["model"] is the writer's).
    with lvs.lmdb.open(path, max_dbs=lvs.MAX_DBS, subdir=True) as env:
        dbs = lvs._open_dbs(env, "concept", create=False)
        with env.begin(write=True) as txn:
            meta = json.loads(txn.get(b"meta", db=dbs["meta"]))
            txn.put(b"meta", json.dumps({**meta, "model": "microsoft/harrier-oss-v1-0.6b"}).encode(), db=dbs["meta"])
    with pytest.raises(ValueError, match="fitted to google/embeddinggemma-2"):
        lvs.build_index(path, "concept", projection=(mean, basis), projection_info={"model": "google/embeddinggemma-2"})
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s.meta["generation"] == 1 and "graph_projection" not in s.meta  # nothing was published
    finally:
        s.close()


def test_a_saved_map_loads_with_where_it_came_from(tmp_path):
    mean, basis = np.zeros(8, np.float32), np.eye(8, 4, dtype=np.float32)
    np.savez(tmp_path / "harrier320.npz", mean=mean, basis=basis, model="microsoft/harrier-oss-v1-0.6b",
             scope="agent", step=50)
    (m, b), info = lvs.load_projection(str(tmp_path / "harrier320.npz"))
    assert np.array_equal(m, mean) and np.array_equal(b, basis)
    assert info == {"file": "harrier320.npz", "model": "microsoft/harrier-oss-v1-0.6b", "scope": "agent", "step": 50}
    # learn_projection.py saves model=None for a store with no model: not unpickled, just left out
    np.savez(tmp_path / "anon.npz", mean=mean, basis=basis, model=None)
    assert lvs.load_projection(str(tmp_path / "anon.npz"))[1] == {"file": "anon.npz"}


def test_graph_dimensions_must_fit_the_vectors(tmp_path):
    data = np.random.default_rng(6).standard_normal((50, 8)).astype(np.float32)
    path = str(tmp_path / "store.lmdb")
    iris = [f"https://example.org/data/concept/{i}" for i in range(50)]
    with pytest.raises(ValueError, match="does not fit"):
        lvs.build_store(path, iris, data, scope="concept", graph_dimensions=9)
    lvs.build_store(path, iris, data, scope="concept", graph_dimensions=4)
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s.meta["graph_dimensions"] == 4
        generation = s.meta["generation"]
    finally:
        s.close()
    with pytest.raises(ValueError, match="does not fit"):
        lvs.build_index(path, "concept", graph_dimensions=9)
    s = lvs.LmdbVectorStore(path, "concept")
    try:
        assert s.meta["generation"] == generation  # nothing was published
    finally:
        s.close()


def test_graph_bytes_matches_the_sizes_measured_on_real_graphs():
    # e4m3 at connectivity 16: 1112, 600 and 344 bytes a vector at 1024, 512 and 256 dimensions
    for dims, per_vector in ((1024, 1112), (512, 600), (256, 344)):
        assert lvs.graph_bytes(1_000_000, dims, lvs.Scalar.E4M3) == 1_000_000 * per_vector
    # full LUX's 4.85M agents at full width do not fit in one LMDB value; on 512 dimensions they do
    assert lvs.graph_bytes(4_849_897, 1024, lvs.Scalar.E4M3) > lvs.LMDB_MAX_VALUE
    assert lvs.graph_bytes(4_849_897, 512, lvs.Scalar.E4M3) < lvs.LMDB_MAX_VALUE


def test_a_graph_too_big_for_lmdb_is_refused_before_it_is_built(tmp_path, monkeypatch):
    data = np.random.default_rng(7).standard_normal((2000, 64)).astype(np.float32)
    path = str(tmp_path / "store.lmdb")
    iris = [f"https://example.org/data/concept/{i}" for i in range(2000)]
    lvs.build_store(path, iris, data, scope="concept")
    built = []
    monkeypatch.setattr(lvs, "LMDB_MAX_VALUE", 300_000)  # 2000 x (64 + 88) bytes is 304,000
    from types import SimpleNamespace  # DenseIndex is a C type: swap the module's name for it instead
    monkeypatch.setattr(lvs, "DenseIndex", SimpleNamespace(init_ram=lambda *a, **k: built.append(1) or 1 / 0))
    with pytest.raises(ValueError, match=r"over LMDB's .* limit .*--graph-dims 32 fits"):
        lvs.build_index(path, "concept")
    assert built == []  # refused up front, not after building


def test_a_graph_bigger_than_guessed_is_still_refused_at_write(tmp_path, monkeypatch):
    data = np.random.default_rng(8).standard_normal((500, 16)).astype(np.float32)
    path = str(tmp_path / "store.lmdb")
    iris = [f"https://example.org/data/concept/{i}" for i in range(500)]
    lvs.build_store(path, iris, data, scope="concept")
    monkeypatch.setattr(lvs, "graph_bytes", lambda *a, **k: 0)  # the estimate guesses low
    monkeypatch.setattr(lvs, "LMDB_MAX_VALUE", 10_000)
    with pytest.raises(ValueError, match="the graph is .* over LMDB's 4 GiB limit"):
        lvs.build_index(path, "concept")
