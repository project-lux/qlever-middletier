"""Pass 1 (labels) must cover every record exactly once in both record-LMDB layouts, and say so as it goes;
`kinds` must sample evenly without walking the whole store; and `embed`'s card pool must match one thread.

Run with the vectors venv: ../vectors/.venv/bin/python -m pytest tests/test_labels.py -q
"""

import json
import uuid
import zlib

import pytest

lmdb = pytest.importorskip("lmdb")

from lux_embed.labels import LabelStore, build_labels  # noqa: E402
from lux_embed.records import RecordStore  # noqa: E402

PRIMARY = "http://vocab.getty.edu/aat/300404670"


def record(n):
    return {"type": "Person", "identified_by": [{"type": "Name", "content": f"Person {n}",
                                                 "classified_as": [{"id": "x", "equivalent": [{"id": PRIMARY}]}]}]}


def make_store(path, binary, count=500):
    """``binary``: data-pipeline's 16-byte keys and bare records; else the datacache's text keys."""
    ids = [uuid.UUID(int=(n * 0x9E3779B97F4A7C15 << 64) % (1 << 128) | n) for n in range(count)]
    env = lmdb.open(path, map_size=1 << 26, max_dbs=2, subdir=True)
    db = env.open_db(b"data")
    with env.begin(write=True, db=db) as txn:
        for n, uid in enumerate(ids):
            body = {"id": f"https://lux.collections.yale.edu/data/person/{uid}", **record(n)}
            doc = body if binary else {"json": body}
            txn.put(uid.bytes if binary else str(uid).encode(), zlib.compress(json.dumps(doc).encode()))
    env.close()
    return ids


@pytest.mark.parametrize("binary", [True, False])
def test_partitions_cover_every_key_once(tmp_path, binary):
    ids = make_store(str(tmp_path / "r.lmdb"), binary)
    store = RecordStore(str(tmp_path / "r.lmdb"))
    ranges = store.partitions(4096)
    assert len(ranges) == 4096  # binary keys used to get only 256, so progress never showed
    seen = [uid for first, bound in ranges for uid, _ in store.scan(first, bound)]
    assert sorted(seen) == sorted(str(uid) for uid in ids)


@pytest.mark.parametrize("binary", [True, False])
def test_labels_reports_progress_and_labels_everything(tmp_path, binary, capsys):
    ids = make_store(str(tmp_path / "r.lmdb"), binary)
    total = build_labels(str(tmp_path / "r.lmdb"), str(tmp_path / "labels.lmdb"), workers=2)
    assert total == len(ids)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("labels: scanning 500 records in 4,096 partitions")
    assert len(lines) == 17  # the opening line, then one per sixteenth
    assert lines[-1].startswith("labels: 500/500 records, 4,096/4,096 partitions")
    labels = LabelStore(str(tmp_path / "labels.lmdb"))
    assert labels.get(str(ids[7])) == ("Person", "Person 7")


@pytest.mark.parametrize("binary", [True, False])
def test_sample_seeks_spread_records_without_repeats(tmp_path, binary):
    make_store(str(tmp_path / "r.lmdb"), binary, count=2000)
    store = RecordStore(str(tmp_path / "r.lmdb"))
    names = [r["identified_by"][0]["content"] for r in store.sample(200)]
    assert 150 < len(names) <= 200 and len(set(names)) == len(names)  # seeks that collide are dropped
    numbers = sorted(int(n.split()[1]) for n in names)
    assert numbers[0] < 200 and numbers[-1] > 1800  # spread over the store, not its first stretch
    assert len(list(store.sample(5000))) == 2000  # more than the store holds: every record once


@pytest.mark.parametrize("binary", [True, False])
def test_card_pool_builds_the_threads_cards_in_order(tmp_path, binary):
    from types import SimpleNamespace

    from lux_embed.produce import card_batches, open_inputs

    make_store(str(tmp_path / "r.lmdb"), binary, count=3000)
    build_labels(str(tmp_path / "r.lmdb"), str(tmp_path / "labels.lmdb"), workers=2)
    args = SimpleNamespace(records=str(tmp_path / "r.lmdb"), labels=str(tmp_path / "labels.lmdb"), facets=None,
                           qlever=None, scope="agent", note_words=300, batch=700)

    def build(workers):
        inputs = open_inputs(args, need_facets=True)
        members = (uuid for uuid, _ in inputs[1].scope_members("agent"))
        return list(card_batches(args, inputs, members, workers))

    threaded, pooled = build(0), build(3)
    flat = [card for batch in pooled for card in batch]
    assert flat == [card for batch in threaded for card in batch]  # same cards, same order
    assert len(flat) == 3000 and [iri for iri, _, _ in flat] == sorted(iri for iri, _, _ in flat)
    assert all(len(batch) >= args.batch for batch in pooled[:-1])  # whole batches for the encoder


def test_card_store_export_matches_the_thread_and_resumes(tmp_path):
    import gc
    from types import SimpleNamespace

    from lux_embed.__main__ import main
    from lux_embed.card_store import CardStore, CardWriter, pack
    from lux_embed.produce import card_batches, open_inputs

    make_store(str(tmp_path / "r.lmdb"), True, count=3000)
    build_labels(str(tmp_path / "r.lmdb"), str(tmp_path / "labels.lmdb"), workers=2)
    args = SimpleNamespace(records=str(tmp_path / "r.lmdb"), labels=str(tmp_path / "labels.lmdb"), facets=None,
                           qlever=None, scope="agent", note_words=300, batch=700)
    def threaded():  # its store handles close on return: py-lmdb opens a path once per process
        inputs = open_inputs(args, need_facets=True)
        return [c for b in card_batches(args, inputs, (u for u, _ in inputs[1].scope_members("agent")), 0) for c in b]

    expected = threaded()
    gc.collect()
    common = ["--records", args.records, "--labels", args.labels, "--scope", "agent", "--batch", "700"]
    main(["cards", *common, "--out", str(tmp_path / "cards.lmdb"), "--card-workers", "2"])
    store = CardStore(str(tmp_path / "cards.lmdb"), "agent")
    assert store.meta["complete"] and store.meta["count"] == 3000
    assert [c for b in store.batches(None, 1000) for c in b] == expected
    after = expected[999][0].rsplit("/", 1)[-1]  # what embed resumes from: the last IRI's UUID
    assert [c for b in store.batches(after, 1000) for c in b] == expected[1000:]
    from lux_embed.vector_shards import shard_bounds  # four key-range shards cover the scope exactly once
    sharded = [c for i in range(4) for b in store.batches(None, 1000, *shard_bounds((i, 4))) for c in b]
    assert sharded == expected
    assert [c for b in store.batches(after, 1000, *shard_bounds((0, 4))) for c in b] == \
        [c for c in expected[1000:] if c[0].rsplit("/", 1)[-1] < "40000000"]  # a shard resumed mid-range

    # as if a build died after 1,000 cards: the rerun picks up there and ends identical
    writer = CardWriter(str(tmp_path / "partial.lmdb"), "agent")
    writer.append([pack(*card) for card in expected[:1000]])
    writer.close()
    main(["cards", *common, "--out", str(tmp_path / "partial.lmdb"), "--card-workers", "0"])
    partial = CardStore(str(tmp_path / "partial.lmdb"), "agent")
    assert partial.meta["count"] == 3000 and [c for b in partial.batches(None, 512) for c in b] == expected


def test_bare_holdings_get_no_card_and_works_name_their_carrier(tmp_path):
    from lux_embed.cards import CardBuilder
    from lux_embed.labels import LabelStore
    from lux_embed.records import RecordStore

    D = "https://lux.collections.yale.edu/data/"
    ids = {name: str(uuid.UUID(int=i + 1)) for i, name in enumerate(
        ["books", "item_class", "work", "library", "museum", "unit", "artist"])}

    def ref(kind, key):
        return {"id": f"{D}{kind}/{ids[key]}", "type": "Type"}

    def named(doc, label):
        doc["identified_by"] = [{"type": "Name", "content": label,
                                 "classified_as": [{"id": "x", "equivalent": [{"id": PRIMARY}]}]}]
        return doc

    def concept(key, label, aat):
        return named({"id": f"{D}concept/{ids[key]}", "type": "Type",
                      "equivalent": [{"id": f"http://vocab.getty.edu/aat/{aat}"}]}, label)

    docs = [
        concept("books", "Books", "300028051"),
        concept("item_class", "Collection Item", "300404024"),
        named({"id": f"{D}text/{ids['work']}", "type": "LinguisticObject"}, "The physics of quantum information"),
        named({"id": f"{D}set/{ids['unit']}", "type": "Set"}, "Marx Library"),
        named({"id": f"{D}person/{ids['artist']}", "type": "Person"}, "A. Printmaker"),
        named({"id": f"{D}object/{ids['library']}", "type": "HumanMadeObject", "carries": [ref("text", "work")],
               "classified_as": [ref("concept", "books"), ref("concept", "item_class")],
               "member_of": [ref("set", "unit")]}, "The physics of quantum information"),
        named({"id": f"{D}object/{ids['museum']}", "type": "HumanMadeObject", "carries": [ref("text", "work")],
               "classified_as": [ref("concept", "books")],
               "produced_by": {"type": "Production", "carried_out_by": [ref("person", "artist")]}}, "An artist's book"),
    ]
    env = lmdb.open(str(tmp_path / "r.lmdb"), map_size=1 << 26, max_dbs=2, subdir=True)
    with env.begin(write=True, db=env.open_db(b"data")) as txn:
        for doc in docs:
            txn.put(uuid.UUID(doc["id"].rsplit("/", 1)[-1]).bytes, zlib.compress(json.dumps(doc).encode()))
    env.close()
    build_labels(str(tmp_path / "r.lmdb"), str(tmp_path / "labels.lmdb"), workers=1)
    records = RecordStore(str(tmp_path / "r.lmdb"))
    builder = CardBuilder(records, LabelStore(str(tmp_path / "labels.lmdb")))

    assert builder.card(records.get(ids["library"])) is None  # Books, Collection Item and a library: nothing of its own
    museum = builder.card(records.get(ids["museum"]))
    assert museum and "Made by: A. Printmaker" in museum[1] and "Type: Books." in museum[1]
    # the work names its carrier, without the blanket class, even though Books is on 77% of LUX objects
    facets = {"object type": [ids["item_class"], ids["books"]]}
    assert "Realized as: Books." in builder.card(records.get(ids["work"]), facets)[1]


def test_skip_name_only_drops_cards_with_nothing_but_names(tmp_path):
    """Half of full LUX's agents have only a name (and maybe other names): their vectors could only
    match on how a name looks, so ``--skip-name-only`` gives them no card. Anything else - a field
    from the record, a facet, a note - keeps the card."""
    from lux_embed.cards import CardBuilder

    D = "https://lux.collections.yale.edu/data/"
    ids = {name: str(uuid.UUID(int=i + 1)) for i, name in enumerate(
        ["bare", "aliases", "national", "noted", "faceted", "french", "physics"])}

    def named(doc, label, *others):
        doc["identified_by"] = [{"type": "Name", "content": label,
                                 "classified_as": [{"id": "x", "equivalent": [{"id": PRIMARY}]}]}]
        doc["identified_by"] += [{"type": "Name", "content": other} for other in others]
        return doc

    def person(key, *others, **fields):
        return named({"id": f"{D}person/{ids[key]}", "type": "Person", **fields}, f"Person {key}", *others)

    nationality = {"id": f"{D}concept/{ids['french']}", "type": "Type",
                   "classified_as": [{"id": "x", "equivalent": [{"id": "http://vocab.getty.edu/aat/300379842"}]}]}
    docs = [
        named({"id": f"{D}concept/{ids['french']}", "type": "Type"}, "French"),
        named({"id": f"{D}concept/{ids['physics']}", "type": "Type"}, "Physics"),
        person("bare"),
        person("aliases", "P. Aliases", "Aliases, P."),
        person("national", classified_as=[nationality]),
        person("noted", referred_to_by=[{"type": "LinguisticObject", "content": "A physicist who taught in Lyon."}]),
        person("faceted"),
    ]
    env = lmdb.open(str(tmp_path / "r.lmdb"), map_size=1 << 26, max_dbs=2, subdir=True)
    with env.begin(write=True, db=env.open_db(b"data")) as txn:
        for doc in docs:
            txn.put(uuid.UUID(doc["id"].rsplit("/", 1)[-1]).bytes, zlib.compress(json.dumps(doc).encode()))
    env.close()
    build_labels(str(tmp_path / "r.lmdb"), str(tmp_path / "labels.lmdb"), workers=1)
    records, labels = RecordStore(str(tmp_path / "r.lmdb")), LabelStore(str(tmp_path / "labels.lmdb"))
    facets = {"faceted": {"subject": [ids["physics"]]}}

    def cards(skip):
        builder = CardBuilder(records, labels, skip_name_only=skip)
        return {key: builder.card(records.get(ids[key]), facets.get(key))
                for key in ("bare", "aliases", "national", "noted", "faceted")}

    kept = cards(skip=False)
    assert all(kept.values())  # off by default: every record still gets its card
    assert kept["bare"] == ("Person bare", "Person bare")
    assert kept["aliases"][1].startswith("Also known as: ")

    skipped = cards(skip=True)
    assert skipped["bare"] is None and skipped["aliases"] is None
    assert "Nationality: French." in skipped["national"][1]
    assert "taught in Lyon" in skipped["noted"][1]
    assert "Subjects: Physics." in skipped["faceted"][1]


def test_agent_cards_name_the_relation_of_each_subject_and_both_date_spans(tmp_path):
    """An author on Edo prints and an Edo print designer must not read alike: subjects of the agent's
    works and subjects their objects depict are named apart. Publication is its own span, since an
    old author's works mostly come out in modern editions. Facets extracted before the split carry a
    single "subject", which still reads as Subjects."""
    from lux_embed.cards import CardBuilder

    D = "https://lux.collections.yale.edu/data/"
    ids = {name: str(uuid.UUID(int=i + 1)) for i, name in enumerate(["hadith", "landscape", "agent"])}

    def named(doc, label):
        doc["identified_by"] = [{"type": "Name", "content": label,
                                 "classified_as": [{"id": "x", "equivalent": [{"id": PRIMARY}]}]}]
        return doc

    docs = [named({"id": f"{D}concept/{ids['hadith']}", "type": "Type"}, "Hadith"),
            named({"id": f"{D}concept/{ids['landscape']}", "type": "Type"}, "Landscapes"),
            named({"id": f"{D}person/{ids['agent']}", "type": "Person"}, "An Agent")]
    env = lmdb.open(str(tmp_path / "r.lmdb"), map_size=1 << 26, max_dbs=2, subdir=True)
    with env.begin(write=True, db=env.open_db(b"data")) as txn:
        for doc in docs:
            txn.put(uuid.UUID(doc["id"].rsplit("/", 1)[-1]).bytes, zlib.compress(json.dumps(doc).encode()))
    env.close()
    build_labels(str(tmp_path / "r.lmdb"), str(tmp_path / "labels.lmdb"), workers=1)
    records = RecordStore(str(tmp_path / "r.lmdb"))
    builder = CardBuilder(records, LabelStore(str(tmp_path / "labels.lmdb")))
    agent = records.get(ids["agent"])

    _, text = builder.card(agent, {"work subject": [ids["hadith"]], "depicted subject": [ids["landscape"]],
                                   "dates": ["1650", "1702"], "published": ["2022", "2023"]})
    assert "Works about: Hadith." in text and "Depicted: Landscapes." in text
    assert "Active: 1650–1702." in text and "Works published: 2022–2023." in text
    assert "Subjects" not in text

    _, legacy = builder.card(agent, {"subject": [ids["hadith"]], "published": ["2022", "2022"]})
    assert "Subjects: Hadith." in legacy and "Works published: 2022." in legacy
