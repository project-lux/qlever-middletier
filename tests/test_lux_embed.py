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
