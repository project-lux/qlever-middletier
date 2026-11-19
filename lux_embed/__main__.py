"""Embedding builder for LUX record classes.

    python -m lux_embed labels  --records R.lmdb --labels L.lmdb            # Pass 1 (once per dataset)
    python -m lux_embed kinds   --records R.lmdb                            # statement kinds, for the skip list
    python -m lux_embed facets  --qlever URL --out F.lmdb [--scope agent]   # Stage 1 offline, beside QLever
    python -m lux_embed cards   --records R.lmdb --labels L.lmdb --scope agent --limit 5 [--qlever URL | --facets F.lmdb]
    python -m lux_embed embed   --records R.lmdb --labels L.lmdb --store V.lmdb --scope agent [--qlever URL | --facets F.lmdb]
    python -m lux_embed index   --store V.lmdb --scope agent
    python -m lux_embed query   --store V.lmdb --labels L.lmdb --scope agent "dutch landscape painters"

``embed`` resumes where it stopped (each committed chunk records its last IRI); pass ``--restart``
to rebuild a scope. Cards and Stage 1 facets are built by a pool of worker processes
(``--card-workers``, default: the CPUs this job may use, less two) while the encoder works.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import os
import time

from .facets import FACET_BATCH
from .produce import FACET_SCOPES, card_batches, cards_for, facets_for, read_records
from .produce import open_inputs as _open
from .records import available_cpus

DEFAULT_DIMS = {"item": 512, "work": 512, "set": 768, "agent": 768, "place": 768, "concept": 768, "event": 768}


def cmd_labels(args):
    from .labels import build_labels

    started = time.time()
    total = build_labels(args.records, args.labels, workers=args.workers)
    print(f"labelled {total:,} records in {time.time() - started:,.0f}s -> {args.labels}")


def cmd_facets(args):
    from .facet_dump import FACET_SCOPES as DUMP_SCOPES, dump_scope
    from .facets import QLever

    qlever = QLever(args.qlever, timeout=args.timeout)
    for scope in [args.scope] if args.scope else DUMP_SCOPES:
        dump_scope(qlever, scope, args.out, batch=args.batch, workers=args.workers, restart=args.restart)


def cmd_kinds(args):
    from .linkedart import aat_ids, as_list
    from .records import RecordStore

    records = RecordStore(args.records)
    total = len(records)
    print(f"kinds: sampling {min(args.sample, total):,} of {total:,} records", flush=True)
    kinds, scanned, started, reported = collections.Counter(), 0, time.time(), time.time()
    for record in records.sample(args.sample):
        scanned += 1
        for statement in as_list(record.get("referred_to_by")):
            for cx in as_list(statement.get("classified_as")):
                for aat in aat_ids(cx):
                    kinds[(aat, cx.get("_label") or "")] += 1  # _label: only to name the kind here
        if time.time() - reported >= 10:
            reported = time.time()
            print(f"kinds: {scanned:,} sampled ({scanned / (reported - started):,.0f}/s)", flush=True)
    print(f"statement kinds in {scanned:,} sampled records:")
    for (aat, label), n in kinds.most_common(40):
        print(f"  {aat}  {n:>8,}  {label}")


def cmd_cards(args):
    records, labels, builder, facets = _open(args, need_facets=True)
    uuids = [uuid for uuid, _ in itertools.islice(labels.scope_members(args.scope, args.after), args.limit)]
    batch = read_records(records, uuids)
    for iri, title, text in cards_for(builder, batch, facets_for(facets, batch)):
        print(f"{iri}\n  title: {title} | text: {text}\n")


def cmd_embed(args):
    import numpy as np

    from qleverlux.vectors.lmdb_vector_store import VectorWriter

    from .encoder import Encoder

    records, labels, builder, facets = _open(args, need_facets=True)
    encoder = Encoder(device=args.device, max_seq_length=args.max_seq_length, model=args.model)
    dims = args.dims or (DEFAULT_DIMS[args.scope] if encoder.profile["matryoshka"] else encoder.native_dims)
    encoder.check_dims(dims)
    writer = VectorWriter(args.store, args.scope, dims, chunk_rows=args.chunk_rows, vector_dtype=args.vector_dtype,
                          model=encoder.name, restart=args.restart)
    if writer.meta.get("model") not in (None, encoder.name):
        raise SystemExit(f"{args.store} {args.scope} was started with {writer.meta['model']}; "
                         f"use another --store, or --restart to rebuild it with {encoder.name}")
    if writer.complete:
        trial = f", a --limit {writer.meta['limit']:,} trial" if writer.meta.get("limit") else ""
        print(f"{args.scope}: already complete ({writer.count:,} vectors{trial}); --restart to rebuild")
        return
    resume = writer.cursor.rsplit("/", 1)[-1] if writer.cursor else None
    print(f"{args.scope}: {dims}d on {encoder.device}/{encoder.dtype}, "
          f"{'resuming after ' + resume if resume else 'from the start'} ({writer.count:,} done)", flush=True)

    members = (uuid for uuid, _ in labels.scope_members(args.scope, resume))
    if args.limit:
        members = itertools.islice(members, args.limit)

    workers = available_cpus() - 2 if args.card_workers < 0 else args.card_workers
    workers = max(0, min(workers, 64))
    print(f"{args.scope}: building cards in {f'{workers} worker processes' if workers else 'one thread'}", flush=True)
    done, started = 0, time.time()
    for cards in card_batches(args, (records, labels, builder, facets), members, workers):
        if not cards:
            continue
        vectors = encoder.encode_documents([(title, text) for _, title, text in cards], dims, args.encode_batch)
        writer.append([iri for iri, _, _ in cards], vectors)
        done += len(cards)
        rate = done / (time.time() - started)
        print(f"{args.scope}: {writer.count + writer._pending:,} embedded ({rate:,.0f} cards/s)", flush=True)
    if args.limit:  # a trial: its partial last chunk ends the scope until a --restart
        writer.meta["limit"] = args.limit
    writer.finish()
    print(f"{args.scope}: complete, {writer.count:,} vectors in {time.time() - started:,.0f}s"
          + (f" (--limit {args.limit:,} trial; --restart for the full scope)" if args.limit else ""))
    writer.close()
    if not args.no_index:
        cmd_index(args)


def cmd_index(args):
    from usearch import Scalar

    from qleverlux.vectors.lmdb_vector_store import build_index

    started = time.time()
    meta = build_index(args.store, args.scope, quantization=getattr(Scalar, args.quantization.upper()),
                       connectivity=args.connectivity)
    print(f"{args.scope}: graph generation {meta['generation']} over {meta['count']:,} vectors "
          f"({meta['quantization']}, M={meta['connectivity']}) in {time.time() - started:,.0f}s")


def cmd_query(args):
    from qleverlux.vectors.lmdb_vector_store import LmdbVectorStore

    from .encoder import MODEL, Encoder
    from .labels import LabelStore

    store, labels = LmdbVectorStore(args.store, args.scope), LabelStore(args.labels)
    encoder = Encoder(device=args.device, model=store.meta.get("model") or MODEL)  # the model that built it
    vector = encoder.encode_query(args.text, store.meta["dimensions"])
    for iri, score in store.search_global(vector, args.top):
        print(f"{score:.3f}  {labels.label(iri) or '?'}  <{iri}>")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lux_embed", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, *needs):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        if "records" in needs:
            p.add_argument("--records", required=True, help="record LMDB (e.g. middletier/datacache/nrich_lux.lmdb)")
        if "labels" in needs:
            p.add_argument("--labels", required=True, help="label store written by `labels`")
        if "store" in needs:
            p.add_argument("--store", required=True, help="vector store LMDB")
        if "scope" in needs:
            p.add_argument("--scope", required=True, choices=sorted(DEFAULT_DIMS))
        if "cards" in needs:
            p.add_argument("--qlever", default=None, help="QLever endpoint for Stage 1 facets, e.g. http://localhost:7011")
            p.add_argument("--facets", default=None, help="Stage 1 facets extracted by `facets`, instead of --qlever")
            p.add_argument("--note-words", type=int, default=300)
        if "device" in needs:
            p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
        return p

    p = add("labels", cmd_labels, "records", "labels")
    p.add_argument("--workers", type=int, default=0, help="default: the CPUs this job may use, less two")
    p = add("facets", cmd_facets)
    p.add_argument("--qlever", required=True, help="QLever endpoint serving the dataset, e.g. http://localhost:7011")
    p.add_argument("--out", required=True, help="facet LMDB directory to write (copy it to the embedding machine)")
    p.add_argument("--scope", default=None, choices=sorted(FACET_SCOPES), help="one scope (default: all three)")
    p.add_argument("--batch", type=int, default=FACET_BATCH, help="targets per QLever query (default %(default)s)")
    p.add_argument("--workers", type=int, default=4, help="concurrent QLever queries")
    p.add_argument("--timeout", type=float, default=120, help="seconds per QLever query before the batch is split")
    p.add_argument("--restart", action="store_true", help="re-extract a scope already done")
    p = add("kinds", cmd_kinds, "records")
    p.add_argument("--sample", type=int, default=200_000)
    p = add("cards", cmd_cards, "records", "labels", "scope", "cards")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--after", default=None, help="start after this UUID")
    p = add("embed", cmd_embed, "records", "labels", "store", "scope", "cards", "device")
    p.add_argument("--model", default="google/embeddinggemma-2",
                   help="sentence encoder; one with a profile in encoder.py (e.g. microsoft/harrier-oss-v1-0.6b)")
    p.add_argument("--dims", type=int, default=0, help="Matryoshka dimension (default per scope; full width "
                   "for models without Matryoshka training)")
    p.add_argument("--limit", type=int, default=0, help="embed at most this many records (for trials)")
    p.add_argument("--batch", type=int, default=2048, help="records per card/facet batch")
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--card-workers", type=int, default=-1,
                   help="processes building cards (default: the CPUs this job may use, less two; 0 = one thread)")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--chunk-rows", type=int, default=4096)
    p.add_argument("--vector-dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--restart", action="store_true")
    p.add_argument("--no-index", action="store_true", help="skip building the graph at the end")
    p.add_argument("--quantization", default="bf16")
    p.add_argument("--connectivity", type=int, default=16)
    p = add("index", cmd_index, "store", "scope")
    p.add_argument("--quantization", default="bf16")
    p.add_argument("--connectivity", type=int, default=16)
    p = add("query", cmd_query, "store", "labels", "scope", "device")
    p.add_argument("text")
    p.add_argument("--top", type=int, default=10)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
