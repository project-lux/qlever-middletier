"""Stage 1 facets extracted to an LMDB must read back exactly as the live QLever lookup answers them.

Run with the vectors venv: ../vectors/.venv/bin/python -m pytest tests/test_facet_dump.py -q
"""

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("lmdb")

from lux_embed.facet_dump import FacetDump, dump_scope  # noqa: E402
from lux_embed.facets import Facets, QLever  # noqa: E402

D = "https://lux.collections.yale.edu/data/"
NS = "https://lux.collections.yale.edu/ns/"
A1, A2, A3 = (f"{D}person/00000000-0000-0000-0000-00000000000{i}" for i in (1, 2, 3))
PARIS, ROME = f"{D}place/10000000-0000-0000-0000-000000000001", f"{D}place/10000000-0000-0000-0000-000000000002"
PRINT = f"{D}concept/20000000-0000-0000-0000-000000000001"

PREDICATES = ["agentOfItemBeginning", "placeOfItemBeginning", "itemClassification",
              "startOfItemBeginning", "endOfItemBeginning"]
# (target, kind, value, count) - A3 has no incoming objects at all
GROUPED = [(A1, "place", PARIS, 3), (A1, "place", ROME, 1), (A1, "object type", PRINT, 4), (A2, "place", ROME, 2)]
DATES = [(A1, "1650-01-01T00:00:00", "1702-12-31T23:59:59"), (A2, "0950-01-01T00:00:00", "0960-01-01T00:00:00")]


def tsv(header, rows):
    def cell(value):
        return f"<{value}>" if str(value).startswith("http") else (
            str(value) if isinstance(value, int) else f'"{value}"')
    return "\n".join(["\t".join(header), *("\t".join(cell(v) for v in row) for row in rows)]) + "\n"


class FakeQLever(BaseHTTPRequestHandler):
    queries: list[str] = []

    def do_POST(self):
        query = urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())["query"][0]
        FakeQLever.queries.append(query)
        if "SELECT DISTINCT ?p" in query:
            body = tsv(["?p"], [[NS + p] for p in PREDICATES])
        elif "SELECT DISTINCT ?target" in query:
            body = tsv(["?target"], [[A1], [A2], [A3]])
        elif "COUNT(?x)" in query:
            body = tsv(["?n"], [[100]])
        elif "VALUES ?target" in query and "?k" in query:
            wanted = {t for t in (A1, A2, A3) if f"<{t}>" in query}
            body = tsv(["?target", "?k", "?v", "?n"], [r for r in GROUPED if r[0] in wanted])
        elif "VALUES ?target" in query:
            wanted = {t for t in (A1, A2, A3) if f"<{t}>" in query}
            body = tsv(["?target", "?start", "?end"], [r for r in DATES if r[0] in wanted])
        else:  # document frequencies, one GROUP BY per kind
            body = tsv(["?v", "?n"], [[PARIS, 10], [ROME, 40], [PRINT, 5]])
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), FakeQLever)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeQLever.queries = []
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


def test_dump_reads_back_as_the_live_lookup(endpoint, tmp_path):
    out = str(tmp_path / "facets.lmdb")
    dump_scope(QLever(endpoint), "agent", out, batch=2, workers=2)

    live = Facets(QLever(endpoint), "agent", cache_dir=str(tmp_path / "live")).fetch([A1, A2, A3])
    dumped = FacetDump(out, "agent").fetch([A1, A2, A3])

    assert set(dumped) == {A1, A2}  # A3 has nothing, so no entry
    assert dumped[A1]["object type"] == [PRINT.rsplit("/", 1)[-1]]
    assert dumped[A1]["dates"] == ["1650", "1702"]
    assert dumped[A2]["dates"] == ["950", "960"]
    for target in (A1, A2):  # same ranking as live; values are stored as the UUIDs the label store keys on
        assert dumped[target]["dates"] == list(live[target]["dates"])
        for kind in ("place", "object type"):
            assert dumped[target].get(kind, []) == [v.rsplit("/", 1)[-1] for v in live[target].get(kind, [])]
    assert dumped[A1]["place"] == [PARIS.rsplit("/", 1)[-1], ROME.rsplit("/", 1)[-1]]  # tf-idf order


def test_interrupted_scope_resumes_after_the_committed_prefix(endpoint, tmp_path):
    import json

    import lmdb

    out = str(tmp_path / "facets.lmdb")
    dump_scope(QLever(endpoint), "agent", out, batch=1, workers=1)
    env = lmdb.open(out, max_dbs=8, subdir=True)
    db, meta_db = env.open_db(b"facets:agent"), env.open_db(b"meta")
    with env.begin(write=True) as txn:  # as if it died after the first batch
        txn.drop(db, delete=False)
        txn.put(b"agent", json.dumps({"done": 1, "total": 3, "complete": False}).encode(), db=meta_db)
    env.close()
    FakeQLever.queries = []
    dump_scope(QLever(endpoint), "agent", out, batch=1, workers=1)
    batches = [q for q in FakeQLever.queries if "VALUES ?target" in q]
    assert not any(f"<{A1}>" in q for q in batches) and any(f"<{A2}>" in q for q in batches)
    assert set(FacetDump(out, "agent").fetch([A1, A2, A3])) == {A2}


def test_complete_scope_is_not_redone_and_missing_scope_is_reported(endpoint, tmp_path):
    out = str(tmp_path / "facets.lmdb")
    dump_scope(QLever(endpoint), "agent", out, batch=2, workers=1)
    asked = len(FakeQLever.queries)
    dump_scope(QLever(endpoint), "agent", out, batch=2, workers=1)
    # only the start-up predicate list is asked again; no targets or batches
    assert all("SELECT DISTINCT ?p" in q for q in FakeQLever.queries[asked:])
    assert FacetDump(out, "agent").meta["complete"]
    assert not FacetDump(out, "work").enabled
