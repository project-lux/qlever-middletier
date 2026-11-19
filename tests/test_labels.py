"""Pass 1 (labels) must cover every record exactly once in both record-LMDB layouts, and say so as it goes.

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
            doc = record(n) if binary else {"json": record(n)}
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
