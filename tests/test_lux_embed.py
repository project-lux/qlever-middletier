"""Tests for the embedding builder's Linked Art helpers and the resumable vector writer.

Run with the vectors venv: ../vectors/.venv/bin/python -m pytest tests/test_lux_embed.py -q
"""


import numpy as np
import pytest

pytest.importorskip("usearch")
pytest.importorskip("lmdb")

from qleverlux.vectors.lmdb_vector_store import LmdbVectorStore, VectorWriter, build_index  # noqa: E402
from lux_embed.linkedart import alternate_names, date_phrase, primary_name, statements  # noqa: E402

AAT = "http://vocab.getty.edu/aat/"


def ref(aat):
    return {"id": f"https://lux.collections.yale.edu/data/concept/{aat}", "type": "Type", "_label": "IGNORED",
            "equivalent": [{"id": AAT + aat, "type": "Type"}]}


def lang(aat, notation=None):
    node = {"id": "https://lux.collections.yale.edu/data/concept/l", "type": "Language",
            "equivalent": [{"id": AAT + aat}]}
    if notation:
        node["notation"] = notation
    return node


EN, FR = "300388277", "300388306"
PRIMARY, SORT, ALT = "300404670", "300451544", "300264273"


def name(content, classes=(), langs=()):
    return {"type": "Name", "content": content, "classified_as": [ref(c) for c in classes],
            "language": [lang(code) for code in langs]}


def test_primary_name_prefers_english_primary_and_ignores_label():
    record = {"_label": "Wrong", "identified_by": [
        name("Nom français", [PRIMARY], [FR]),
        name("English Name", [PRIMARY], [EN]),
        name("Untagged Primary", [PRIMARY]),
    ]}
    assert primary_name(record) == "English Name"


def test_primary_name_falls_back_in_order():
    assert primary_name({"identified_by": [name("Nom", [PRIMARY], [FR]), name("Untagged", [PRIMARY])]}) == "Untagged"
    assert primary_name({"identified_by": [name("Nom", [PRIMARY], [FR]), name("Plain")]}) == "Nom"
    assert primary_name({"identified_by": [name("Nom", [], [FR]), name("Plain")]}) == "Plain"
    assert primary_name({"identified_by": []}) is None


def test_english_detected_by_notation_alone():
    record = {"identified_by": [name("Nom", [PRIMARY], [FR]),
                                {"type": "Name", "content": "By notation", "classified_as": [ref(PRIMARY)],
                                 "language": [{"id": "x", "notation": "en"}]}]}
    assert primary_name(record) == "By notation"


def test_alternate_names_skip_sort_names_other_languages_and_duplicates():
    record = {"identified_by": [
        name("Smith, John", [SORT], [EN]),
        name("John Smith", [PRIMARY], [EN]),
        name("J. Smith", [ALT], [EN]),
        name("john smith", [ALT]),
        name("Jean Smith", [ALT], [FR]),
    ]}
    assert alternate_names(record, primary_name(record)) == ["J. Smith"]


def test_statements_keep_english_strip_html_skip_boilerplate_and_cap():
    record = {"referred_to_by": [
        {"content": "<p>A <b>fine</b> painting &amp; frame.</p>", "classified_as": [ref("300435416")], "language": [lang(EN)]},
        {"content": "Une peinture.", "language": [lang(FR)]},
        {"content": "Gift of someone, 1950", "classified_as": [ref("300026687")]},
        {"content": " ".join(["word"] * 50)},
    ]}
    out = statements(record, max_words=10)
    assert out[0] == "A fine painting & frame."
    assert all("Une" not in s and "Gift" not in s for s in out)
    assert sum(len(s.replace(" …", "").split()) for s in out) <= 10


def test_date_phrase_prefers_the_timespan_name():
    assert date_phrase({"identified_by": [{"content": "late 1926"}], "begin_of_the_begin": "1926-06-01"}) == "late 1926"
    assert date_phrase({"begin_of_the_begin": "1880-01-01T00:00:00Z", "end_of_the_end": "1890-12-31T23:59:59Z"}) == "1880–1890"
    assert date_phrase({"begin_of_the_begin": "1888-01-01", "end_of_the_end": "1888-12-31"}) == "1888"
    assert date_phrase(None) is None


def test_writer_commits_whole_chunks_and_resumes(tmp_path):
    path = str(tmp_path / "store.lmdb")
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((25, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    iris = [f"https://lux.collections.yale.edu/data/concept/{i:08x}-0000-4000-8000-000000000000" for i in range(25)]

    writer = VectorWriter(path, "concept", 8, chunk_rows=10)
    writer.append(iris[:13], vectors[:13])
    assert writer.count == 10 and writer.cursor == iris[9]  # only the full chunk committed
    writer.close()  # crash: the 3 pending rows are lost

    writer = VectorWriter(path, "concept", 8, chunk_rows=10)
    assert writer.count == 10 and writer.cursor == iris[9]
    writer.append(iris[10:], vectors[10:])
    writer.finish()
    assert writer.complete and writer.count == 25
    writer.close()

    with pytest.raises(ValueError):
        VectorWriter(path, "concept", 16, chunk_rows=10)  # a different geometry needs restart=True

    meta = build_index(path, "concept")
    assert meta["generation"] == 1 and meta["count"] == 25
    store = LmdbVectorStore(path, "concept")
    assert np.allclose(store.get_vector(iris[17]), vectors[17])
    assert store.search_global(vectors[3], 1)[0][0] == iris[3]
    store.close()


def test_shards_load_into_the_store_embed_would_have_written(tmp_path):
    import lmdb

    from lux_embed.vector_shards import ShardWriter, load_shards
    from qleverlux.vectors.lmdb_vector_store import MAX_DBS, _open_dbs

    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((25, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    iris = [f"https://lux.collections.yale.edu/data/concept/{i:08x}-0000-4000-8000-000000000000" for i in range(25)]
    common = dict(vector_dtype="float16", model="m")

    direct = VectorWriter(str(tmp_path / "direct.lmdb"), "concept", 8, chunk_rows=10, **common)
    direct.append(iris, vectors)
    direct.finish()
    direct.close()

    first = ShardWriter(str(tmp_path / "s0"), "concept", 8, shard=(0, 2), **common)
    first.append(iris[:6], vectors[:6])
    first.close()
    with open(tmp_path / "s0" / "vectors.bin", "ab") as f:  # a crash mid-batch: bytes past the last commit
        f.write(b"\1" * 40)
    first = ShardWriter(str(tmp_path / "s0"), "concept", 8, shard=(0, 2), **common)
    assert first.count == 6 and first.cursor == iris[5]
    first.append(iris[6:12], vectors[6:12])
    first.finish()
    first.close()
    second = ShardWriter(str(tmp_path / "s1"), "concept", 8, shard=(1, 2), **common)
    second.append(iris[12:], vectors[12:])
    second.close()

    with pytest.raises(SystemExit):  # shard 1 isn't finished
        load_shards(str(tmp_path / "loaded.lmdb"), "concept", [str(tmp_path / "s1"), str(tmp_path / "s0")])
    second = ShardWriter(str(tmp_path / "s1"), "concept", 8, shard=(1, 2), **common)
    second.finish()
    second.close()
    loaded = load_shards(str(tmp_path / "loaded.lmdb"), "concept", [str(tmp_path / "s1"), str(tmp_path / "s0")],
                         restart=True, chunk_rows=10)
    assert loaded == 25

    def dump(path):
        env = lmdb.open(path, readonly=True, lock=False, max_dbs=MAX_DBS, subdir=True)
        dbs = _open_dbs(env, "concept", create=False)
        with env.begin() as txn:
            out = {name: list(txn.cursor(db=db)) for name, db in dbs.items()}
        env.close()
        return out

    loaded_dump, direct_dump = dump(str(tmp_path / "loaded.lmdb")), dump(str(tmp_path / "direct.lmdb"))
    assert loaded_dump.keys() == direct_dump.keys()
    for name in loaded_dump:  # chunks, IRIs and meta: byte for byte what embed --store writes
        assert loaded_dump[name] == direct_dump[name], name


def test_token_budget_batches_one_length_each_and_keeps_the_order():
    """--token-budget: cards of one token length per batch (no padding, so no attention mask), at
    most budget // length cards each, and the vectors back in the cards' own order."""
    from lux_embed.encoder import Encoder

    seen = []

    class Model:
        max_seq_length = 6

        def tokenizer(self, texts, add_special_tokens=True):
            return {"input_ids": [text.split() for text in texts]}

        def encode(self, texts, batch_size, truncate_dim, normalize_embeddings, convert_to_numpy):
            seen.append([min(len(t.split()), self.max_seq_length) for t in texts])
            return np.array([[float(t.split()[1]), 1.0] for t in texts])  # the card's number, to check order

    encoder = Encoder.__new__(Encoder)  # no model download: only the batching is under test
    encoder.model, encoder.profile, encoder.native_dims, encoder.name = Model(), {"document": "{title} {text}",
                                                                              "matryoshka": True}, 2, "fake"
    words = [3, 1, 3, 9, 2, 3, 1, 7, 3, 3]  # two cards over the cap of 6: those are one length, 6
    cards = [(f"card {n}", " ".join(["w"] * (k - 2))) if k > 2 else (f"card {n}", "") for n, k in enumerate(words)]
    texts = [encoder.document(t, x) for t, x in cards]
    lengths = [min(len(t.split()), 6) for t in texts]

    vectors = encoder.encode_documents(cards, 2, token_budget=9)
    np.testing.assert_allclose(vectors[:, 0] / vectors[:, 1], range(len(cards)))  # order kept
    assert all(len(set(batch)) == 1 for batch in seen)  # one length a batch
    assert all(len(batch) <= max(1, 9 // batch[0]) for batch in seen)  # within the budget
    assert sorted(n for batch in seen for n in batch) == sorted(lengths)  # every card exactly once
    assert len(encoder.length_batches(texts, 10_000, most=2)) == sum(-(-lengths.count(n) // 2) for n in set(lengths))
