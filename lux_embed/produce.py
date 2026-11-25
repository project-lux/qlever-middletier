"""Card production for ``cards`` and ``embed``: records + labels + Stage 1 facets -> ``(iri, title, text)``.

Building a card is pure Python: decompress and parse the record, then look up a label for every
reference it makes and every facet value. One thread manages a few hundred cards a second, while
an H200 encodes thousands, so ``embed`` builds them in a pool of worker processes
(``--card-workers``). Each worker opens its own read-only handles on the stores; the parent only
encodes and writes. Pieces are submitted in member order and collected in the same order, so the
vector store still receives records in key order and a resumed run starts after the last one.

This module must not import torch: spawned workers import it, and should stay small.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import queue
import signal
import threading
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

from .card_store import pack
from .facets import FACET_BATCH, SCOPE_FACETS

FACET_SCOPES = set(SCOPE_FACETS)
PIECE = 512  # members per task handed to a worker


def read_records(records, uuids) -> list[dict]:
    """The records of ``uuids`` (missing ones dropped), decompressed once per batch."""
    with records.env.begin(db=records.db) as txn:
        return [r for r in (records.get(uuid, txn) for uuid in uuids) if r]


def facets_for(facets, batch: list[dict]) -> dict:
    if facets is None:
        return {}
    iris = [record["id"] for record in batch]
    found = {}
    for start in range(0, len(iris), FACET_BATCH):
        found.update(facets.fetch(iris[start : start + FACET_BATCH]))
    return found


def cards_for(builder, batch: list[dict], facets: dict) -> list[tuple[str, str, str]]:
    """``(iri, title, text)`` for every record of the batch that yields a card."""
    out = []
    for record in batch:
        card = builder.card(record, facets.get(record["id"], {}))
        if card:
            out.append((record["id"], *card))
    return out


def open_inputs(args, need_facets: bool):
    """``(records, labels, builder, facets)`` for ``args.records`` / ``labels`` / ``facets`` / ``qlever``."""
    from .cards import CardBuilder
    from .facets import Facets, QLever
    from .labels import LabelStore
    from .records import RecordStore

    records, labels = RecordStore(args.records), LabelStore(args.labels)
    facets = None
    if need_facets and args.qlever and args.facets:
        raise SystemExit("--qlever and --facets are alternatives; give one")
    if need_facets and args.facets and args.scope in FACET_SCOPES:
        from .facet_dump import FacetDump

        facets = FacetDump(args.facets, args.scope)
        if not facets.enabled:
            raise SystemExit(f"{args.facets} has no {args.scope} facets; run `facets --scope {args.scope}` "
                             f"beside QLever, or drop --facets to build cards without them")
    elif need_facets and args.qlever and args.scope in FACET_SCOPES:
        facets = Facets(QLever(args.qlever), args.scope, cache_dir=os.path.dirname(os.path.abspath(args.labels)))
        if not facets.enabled:
            print(f"facets: the endpoint has none of the {args.scope} facet predicates; cards go without", flush=True)
            facets = None
    builder = CardBuilder(records, labels, max_note_words=args.note_words,
                          skip_name_only=bool(getattr(args, "skip_name_only", False)))
    return records, labels, builder, facets


# -- the worker pool -------------------------------------------------------------------------------

_inputs = None


def _init_worker(settings: dict) -> None:
    global _inputs
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is the parent's to handle
    _inputs = open_inputs(SimpleNamespace(**settings), need_facets=True)


def _build(inputs, uuids: list[str], packed: bool) -> list:
    records, _, builder, facets = inputs
    batch = read_records(records, uuids)
    cards = cards_for(builder, batch, facets_for(facets, batch))
    return [pack(*card) for card in cards] if packed else cards


def _piece(uuids: list[str], packed: bool) -> list:
    return _build(_inputs, uuids, packed)


def card_batches(args, inputs, members: Iterator[str], workers: int, packed: bool = False) -> Iterator[list]:
    """Yields lists of about ``args.batch`` cards for ``members``, in member order.

    Cards are ``(iri, title, text)``, or with ``packed`` the card store's ``(key, value)`` rows,
    compressed in the workers so the writing parent only has to put them.

    ``workers`` 0 builds them in one background thread with ``inputs`` (the parent's own handles);
    more starts that many processes. The parent must have opened ``inputs`` first either way: with
    ``--qlever`` that computes and caches the facet document frequencies once, rather than once per
    worker.
    """
    if workers <= 0:
        yield from prefetch(_build(inputs, uuids, packed)
                            for uuids in iter(lambda: list(itertools.islice(members, args.batch)), []))
        return
    settings = {key: getattr(args, key, None)
                for key in ("records", "labels", "facets", "qlever", "scope", "note_words", "skip_name_only")}
    pieces = iter(lambda: list(itertools.islice(members, PIECE)), [])
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(settings,))
    pending: deque = deque()

    def fill() -> None:  # keep every worker busy while the parent encodes, without listing the whole scope
        while len(pending) < 4 * workers and (piece := next(pieces, None)) is not None:
            pending.append(pool.submit(_piece, piece, packed))

    try:
        fill()
        out: list = []
        while pending:
            out.extend(pending.popleft().result())
            fill()
            if len(out) >= args.batch:
                yield out
                out = []
        if out:
            yield out
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def prefetch(batches: Iterator[list], depth: int = 2) -> Iterator[list]:
    """Runs ``batches`` one or two ahead in a background thread while the caller works on this one."""
    ready: queue.Queue = queue.Queue(maxsize=depth)

    def produce():
        try:
            for batch in batches:
                ready.put(batch)
        except BaseException as error:  # surface producer failures in the caller
            ready.put(error)
        ready.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while (batch := ready.get()) is not None:
        if isinstance(batch, BaseException):
            raise batch
        yield batch


def card_workers(requested: int) -> int:
    """``--card-workers``: negative means the CPUs this job may use, less two; 0 means one thread."""
    from .records import available_cpus

    return max(0, min(available_cpus() - 2 if requested < 0 else requested, 64))
