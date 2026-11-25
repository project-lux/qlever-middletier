#!/usr/bin/env python3
"""Copy a LUX record LMDB without its objects and works, for a smaller download.

Agent, place, concept, event and set cards never read object or work records: what objects and
works say about an agent arrives through the Stage 1 facets, extracted from QLever. So for those
builds the ~60M object and work records are dead weight.

    python strip_records.py /path/to/records.lmdb /path/to/records-no-objects.lmdb

Reads either record layout lux_embed reads (sub-DB ``data``): the middle tier's datacache
(``{"json": record, ...}``) or a distribution store (the bare record), values zlib JSON. Keys and
values are copied unchanged, in key order, so the copy is the same layout and loads fast.

A record's class comes from its URI, ``.../data/<class>/<uuid>``: object and digital are items,
text and visual are works. Only the first few hundred bytes are decompressed to find it; a record
whose top-level ``id`` is not at the start is decompressed whole and parsed (counted as
``parsed``). Needs only Python 3 and ``lmdb``.
"""

import argparse
import json
import re
import sys
import time
import zlib
from collections import Counter

import lmdb

DROP = {"object", "digital", "text", "visual"}  # items and works
# The top-level id at the very start of the document, with or without the datacache wrapper.
HEAD = re.compile(rb'^\s*\{\s*(?:"json"\s*:\s*\{\s*)?"id"\s*:\s*"[^"]*/data/([a-z]+)/')
BATCH = 50_000  # records per write transaction


def record_class(value: bytes) -> tuple[str, bool]:
    """(class from the record's URI, whether it needed a full parse)."""
    head = zlib.decompressobj().decompress(value, 512)
    found = HEAD.match(head)
    if found:
        return found.group(1).decode(), False
    doc = json.loads(zlib.decompress(value))
    record = doc["json"] if isinstance(doc, dict) and "json" in doc and "type" not in doc else doc
    found = re.search(r"/data/([a-z]+)/", record.get("id", ""))
    return (found.group(1) if found else "unknown"), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--limit", type=int, default=0, help="stop after N source records (a trial)")
    args = parser.parse_args()

    src = lmdb.open(args.source, readonly=True, lock=False, max_dbs=8, subdir=True, readahead=True)
    size = src.info()["map_size"]
    dst = lmdb.open(args.target, map_size=max(size, 1 << 40), max_dbs=8, subdir=True,
                    metasync=False, sync=False)  # no writemap: it would size the file to the whole map
    src_db, dst_db = src.open_db(b"data", create=False), dst.open_db(b"data")

    kept, dropped, parsed = Counter(), Counter(), 0
    started, seen = time.time(), 0
    with src.begin(db=src_db, buffers=True) as rtxn:
        cursor = rtxn.cursor()
        wtxn = dst.begin(db=dst_db, write=True)
        for key, value in cursor:
            cls, slow = record_class(value)
            parsed += slow
            if cls in DROP:
                dropped[cls] += 1
            else:
                kept[cls] += 1
                wtxn.put(bytes(key), bytes(value), append=True)  # source order is key order
            seen += 1
            if seen % BATCH == 0:
                wtxn.commit()
                wtxn = dst.begin(db=dst_db, write=True)
                if seen % 1_000_000 == 0:
                    rate = seen / (time.time() - started)
                    print(f"{seen:,} read, {sum(kept.values()):,} kept ({rate:,.0f}/s)", flush=True)
            if args.limit and seen >= args.limit:
                break
        wtxn.commit()
    dst.sync(True)

    print(f"done in {time.time() - started:,.0f} s: {seen:,} read, {parsed:,} needed a full parse")
    print("kept:   ", dict(kept.most_common()))
    print("dropped:", dict(dropped.most_common()))
    if kept.get("unknown"):
        print("note: records with no /data/<class>/ in their id were kept", file=sys.stderr)


if __name__ == "__main__":
    main()
