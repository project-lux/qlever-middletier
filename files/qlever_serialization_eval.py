"""Which result serialization should the middle tier ask QLever for?

The middle tier asks for ``application/qlever-results+json`` and unwraps it with
``process_qlever_results``. QLever can also answer a SELECT as SPARQL JSON, SPARQL XML, TSV, CSV
or raw 64-bit IDs, each with or without gzip. This measures, for the queries the middle tier
actually sends, what each choice costs end to end, and whether it could be a drop-in replacement:

    /Users/rs2668/Development/ENV/bin/python files/qlever_serialization_eval.py \\
        --endpoint http://localhost:7010/sparql --out serialization.json

1. **Workload.** A ``MiddleTier`` is built with its QLever client's ``query`` replaced by a recorder,
   and the services are called as the routes call them: searches (each issues its 60-row window
   and its ``COUNT(*)`` total), search estimates, facets, related lists, and the HAL count queries
   and related-list probes an uncached record issues. So these are the exact query texts, with
   the middle tier's ``send=60``. Records are picked from the endpoint (``--records`` overrides),
   so nothing here is tied to one data build. ``--bulk`` adds plain SELECTs of N rows, sent in
   full, for the per-row cost.
2. **Correctness.** Each format's body is parsed into what ``process_qlever_results`` returns
   (``<>`` stripped, ``xsd:int`` / ``xsd:decimal`` coerced, plain literals kept quoted, unbound as
   None), and compared row for row with the current format. Reported per format: rows identical
   or not (with an example), whether ``send`` is honoured (else every row of an unlimited query is
   sent), and whether the result's total size is reported (facets page on it).
3. **Latency.** Every query is run once first, so QLever's result cache holds it: what remains is
   exactly what the format changes - QLever writing the result, the transfer, decompression and
   parsing in Python. Each (query, format, encoding) is then timed ``--repeats`` times,
   interleaved, and the median kept. QLever's own compute time from the warm-up is reported
   beside it, for scale.
4. **Throughput.** ``--concurrency`` clients share one event loop and one httpx client, as in a
   middle tier worker, cycling through the workload for ``--seconds``: requests a second and
   p50/p95 latency including the parse.

Formats QLever answers but the middle tier cannot use are still timed, and flagged: CSV cannot
tell an IRI from a literal or a number from a string, and the binary IDs are QLever's internal
vocabulary indices. ``text/turtle`` and the other RDF types apply to CONSTRUCT only; for a SELECT
QLever answers SPARQL JSON.

Needs the middle tier's own dependencies (it builds the catalogue) and a reachable QLever.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import io
import json
import os
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from qleverlux.clients.qlever import process_qlever_results  # noqa: E402

PREFIX = "PREFIX lux: <https://lux.collections.yale.edu/ns/>\n"
SPARQL_NS = "{http://www.w3.org/2005/sparql-results#}"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

# ---------------------------------------------------------------------------------------------
# Parsers: each turns a body into what process_qlever_results returns for the same result.


def _coerce(value: str, datatype: str):
    """process_qlever_results' rule for a typed literal: int and decimal become numbers, any
    other type is its bare lexical form."""
    if datatype.endswith("int"):
        return int(value)
    if datatype.endswith("decimal"):
        return float(value)
    return value


def _literal(value: str, lang: str | None) -> str:
    return f'"{value}"' + (f"@{lang}" if lang else "")


def parse_qlever_json(body: bytes) -> dict:
    return process_qlever_results(json.loads(body))


def parse_sparql_json(body: bytes) -> dict:
    data = json.loads(body)
    names = data["head"]["vars"]
    rows = []
    for binding in data["results"]["bindings"]:
        row = []
        for name in names:
            term = binding.get(name)
            if term is None:
                row.append(None)
            elif term["type"] == "uri":
                row.append(term["value"])
            elif term["type"] == "bnode":
                row.append("_:" + term["value"])
            elif "datatype" in term:
                row.append(_coerce(term["value"], term["datatype"]))
            else:
                row.append(_literal(term["value"], term.get("xml:lang")))
        rows.append(row)
    meta = data.get("meta", {})  # QLever's extension: query-time-ms, result-size-total
    return {"results": rows, "total": meta.get("result-size-total"),
            "time": f"{meta['query-time-ms']}ms" if "query-time-ms" in meta else "unknown",
            "variables": ["?" + n for n in names]}


def parse_sparql_xml(body: bytes) -> dict:
    root = ET.fromstring(body)
    names = [v.get("name") for v in root.iter(SPARQL_NS + "variable")]
    rows = []
    for result in root.iter(SPARQL_NS + "result"):
        values = {}
        for binding in result:
            term = binding[0]
            tag = term.tag.removeprefix(SPARQL_NS)
            text = term.text or ""
            if tag == "uri":
                values[binding.get("name")] = text
            elif tag == "bnode":
                values[binding.get("name")] = "_:" + text
            elif term.get("datatype"):
                values[binding.get("name")] = _coerce(text, term.get("datatype"))
            else:
                values[binding.get("name")] = _literal(text, term.get(XML_LANG))
        rows.append([values.get(n) for n in names])
    return {"results": rows, "total": None, "time": "unknown", "variables": ["?" + n for n in names]}


def _tsv_term(cell: str):
    """QLever's TSV does not escape: a tab or newline inside a literal is written as a space, so
    such a literal cannot come back as it was (the one way TSV rows can differ)."""
    if cell == "":
        return None
    if cell[0] == "<" and cell[-1] == ">":
        return cell[1:-1]
    if cell[0] == '"':
        if cell[-1] == ">" and '"^^<' in cell:
            lexical, datatype = cell[:-1].rsplit('"^^<', 1)
            return _coerce(lexical[1:], datatype)
        return cell  # quoted, with any @lang, as process_qlever_results keeps it
    if cell.startswith("_:"):
        return cell
    try:  # QLever writes numbers bare in TSV
        return int(cell)
    except ValueError:
        pass
    try:
        return float(cell)
    except ValueError:
        return cell


def parse_tsv(body: bytes) -> dict:
    lines = body.decode().split("\n")
    names = lines[0].split("\t") if lines and lines[0] else []
    rows = [[_tsv_term(cell) for cell in line.split("\t")] for line in lines[1:] if line]
    return {"results": rows, "total": None, "time": "unknown", "variables": names}


def parse_csv(body: bytes) -> dict:
    """Lossy by construction: an IRI, a literal and a number all arrive as bare text."""
    reader = csv.reader(io.StringIO(body.decode()))
    names = next(reader, [])
    rows = []
    for record in reader:
        row = []
        for cell in record:
            if cell == "":
                row.append(None)
                continue
            try:
                row.append(int(cell))
            except ValueError:
                row.append(cell)
        rows.append(row)
    return {"results": rows, "total": None, "time": "unknown", "variables": ["?" + n for n in names]}


def parse_binary_ids(body: bytes) -> dict:
    """QLever's internal 64-bit IDs, no header: nothing the middle tier can turn back into terms."""
    return {"results": [None] * (len(body) // 8), "total": None, "time": "unknown", "variables": []}


FORMATS = {
    "qlever-json": ("application/qlever-results+json", parse_qlever_json),
    "sparql-json": ("application/sparql-results+json", parse_sparql_json),
    "sparql-xml": ("application/sparql-results+xml", parse_sparql_xml),
    "tsv": ("text/tab-separated-values", parse_tsv),
    "csv": ("text/csv", parse_csv),
    "binary-ids": ("application/octet-stream", parse_binary_ids),
}
USABLE_NOTE = {"csv": "lossy: IRIs, literals and numbers indistinguishable",
               "binary-ids": "unusable: internal vocabulary IDs, no terms"}

# ---------------------------------------------------------------------------------------------
# Workload: the middle tier's own queries, recorded.

SEARCHES = [  # (scope, query, sort)
    ("item", {"text": "rembrandt"}, "relevance:DESC"),
    ("item", {"AND": [{"text": "landscape"}, {"hasDigitalImage": 1}]}, "anySortName:ASC"),
    ("work", {"text": "photography"}, "relevance:DESC"),
    ("work", {"aboutConcept": {"name": "china"}}, "workPublicationDate:DESC"),
    ("agent", {"name": "smith"}, "relevance:DESC"),
    ("concept", {"text": "architecture"}, "relevance:DESC"),
    ("place", {"partOf": {"name": "italy"}}, "anySortName:ASC"),
    ("set", {"text": "papers"}, "relevance:DESC"),
    ("event", {"text": "exhibition"}, "relevance:DESC"),
]
FACETS = [  # (scope, query, facet)
    ("item", {"text": "rembrandt"}, "itemTypeId"),
    ("item", {"text": "rembrandt"}, "itemMaterialId"),
    ("work", {"text": "photography"}, "workAboutConceptId"),
    ("work", {"text": "photography"}, "workCreationDate"),
    ("agent", {"name": "smith"}, "agentNationalityId"),
    ("place", {"partOf": {"name": "italy"}}, "placeTypeId"),
    ("set", {"text": "papers"}, "setAboutConceptId"),
]
#: scope -> the pattern ranking its records by how linked they are; the record at RECORD_RANK is
#: used, busy enough to have related lists and HAL links but not a giant.
RECORD_PATTERNS = {
    "agent": "?x lux:agentOfItemBeginning ?t",
    "concept": "?x lux:workAboutConcept ?t",
    "place": "?x lux:workAboutPlace ?t",
    "set": "?x lux:itemMemberOfSet ?t",
}
RECORD_RANK = 200


def pick_records(endpoint: str) -> dict[str, str]:
    out = {}
    for scope, pattern in RECORD_PATTERNS.items():
        query = (f"SELECT ?t (COUNT(?x) AS ?n) WHERE {{ {pattern} }} GROUP BY ?t "
                 f"ORDER BY DESC(?n) OFFSET {RECORD_RANK} LIMIT 1")
        body = urllib.parse.urlencode({"query": PREFIX + query}).encode()
        request = urllib.request.Request(endpoint, data=body, headers={"Accept": "application/qlever-results+json"})
        rows = json.load(urllib.request.urlopen(request, timeout=300))["res"]
        if rows:
            out[scope] = rows[0][0][1:-1]
    return out


async def capture_workload(records: dict[str, str], send: int) -> list[dict]:
    from qleverlux.app import MiddleTier
    from qleverlux.enums import scopeEnum, searchScopeEnum
    from qleverlux.settings import load_settings

    mt = MiddleTier(load_settings([]))
    catalogue = mt.catalogue
    captured: list[dict] = []
    current = {"kind": "", "label": ""}

    async def record(query, sheddable=True):
        kind = current["kind"]
        if kind == "search" and "COUNT(*)" in query:
            kind = "search-total"
        captured.append({"kind": kind, "label": current["label"], "query": query, "send": send})
        return {"results": [], "total": 0, "time": "0ms", "variables": []}

    mt.qlever.query = record

    async def run(kind, label, call):
        current.update(kind=kind, label=label)
        try:
            await call
        except Exception as error:  # a request the vocabulary rejects is left out, and said so
            print(f"workload: skipped {kind} {label}: {type(error).__name__}: {error}", file=sys.stderr)

    for scope, query, sort in SEARCHES:
        q = json.dumps(query)
        await run("search", f"{scope} {q} {sort}", mt.search.search(searchScopeEnum(scope), q, 1, 20, sort))
        await run("estimate", f"{scope} {q}", mt.search.estimate(searchScopeEnum(scope), q, 1))
    for scope, query, name in FACETS:
        q = json.dumps(query)
        await run("facet", f"{scope} {name} {q}", mt.facets.facet(scopeEnum(scope), q, name, 1, ""))
    for scope, uri in records.items():
        for name in catalogue.related_list_sparql.get(scope, {}):
            await run("related", f"{scope} {name}", mt.related.related_list(scopeEnum(scope), name, uri, 1, None))
    # HAL on a cache miss: every count query of the record's scope, and the related-list probes
    for scope, uri in records.items():
        for hal, query in catalogue.sparql_hal_queries.get(scope, {}).items():
            if not isinstance(query, str):  # a related list: answered by the probes below
                continue
            captured.append({"kind": "hal-count", "label": f"{scope} {hal}", "query": query.replace("URI-HERE", uri),
                             "send": send})
        for name, probes in catalogue.hal_related_list_tests.get(scope, {}).items():
            for i, query in enumerate(probes):
                captured.append({"kind": "hal-probe", "label": f"{scope} {name} #{i}",
                                 "query": query.replace("V_TARGET_URI", uri), "send": send})
    seen, unique = set(), []
    for item in captured:
        if item["query"] not in seen:
            seen.add(item["query"])
            unique.append(item)
    return unique


def bulk_queries(sizes: list[int]) -> list[dict]:
    return [{"kind": "bulk", "label": f"{n:,} rows", "send": n,
             "query": PREFIX + f"SELECT ?uri ?name WHERE {{ ?uri a lux:Item ; lux:itemPrimaryName ?name }} LIMIT {n}"}
            for n in sizes]

# ---------------------------------------------------------------------------------------------
# Measurement.


async def fetch(client, endpoint, item, fmt, encoding, timeout):
    accept, parse = FORMATS[fmt]
    started = time.perf_counter()
    response = await client.post(endpoint, data={"query": item["query"], "send": item["send"], "timeout": f"{timeout}s"},
                                 headers={"Accept": accept, "Accept-Encoding": encoding})
    body = response.content  # read and, for gzip, decompressed
    fetched = time.perf_counter()
    if response.status_code != 200:
        raise RuntimeError(f"{fmt}: HTTP {response.status_code}: {body[:300]!r}")
    parsed = parse(body)
    done = time.perf_counter()
    return {"http_ms": (fetched - started) * 1e3, "parse_ms": (done - fetched) * 1e3, "wire": response.num_bytes_downloaded,
            "size": len(body), "rows": len(parsed["results"]), "parsed": parsed}


def compare(reference: dict, parsed: dict, send: int) -> str | None:
    """None when ``parsed`` could replace ``reference``; else what differs first."""
    want = reference["results"][:send]
    got = parsed["results"][:send]
    if len(got) != len(want):
        return f"{len(got)} rows, expected {len(want)}"
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            return f"row {i}: {b!r} != {a!r}"
    if parsed["variables"] and parsed["variables"] != reference["variables"]:
        return f"variables {parsed['variables']} != {reference['variables']}"
    return None


def median(values):
    return statistics.median(values) if values else float("nan")


async def latency(endpoint, workload, formats, encodings, repeats, timeout):
    limits = httpx.Limits(max_connections=4)
    async with httpx.AsyncClient(http2=True, verify=False, timeout=timeout + 5, limits=limits) as client:
        # Warm QLever's result cache, keep the reference result and QLever's compute time
        for item in workload:
            started = time.perf_counter()
            response = await client.post(endpoint, data={"query": item["query"], "send": item["send"],
                                                         "timeout": f"{timeout}s"},
                                         headers={"Accept": "application/qlever-results+json"})
            raw = response.json()
            item["cold_ms"] = (time.perf_counter() - started) * 1e3
            item["qlever_compute"] = raw.get("time", {}).get("computeResult")
            item["error"] = raw.get("exception") if raw.get("status") == "ERROR" else None
            item["reference"] = process_qlever_results(raw)
        workload = [item for item in workload if not item["error"]]
        checks = {fmt: {"identical": 0, "differ": 0, "example": None, "send_ignored": 0, "total_reported": 0}
                  for fmt in formats}
        timings = {(i, fmt, enc): [] for i in range(len(workload)) for fmt in formats for enc in encodings}
        sizes = {}
        for rep in range(repeats):
            for i, item in enumerate(workload):
                combos = [(fmt, enc) for fmt in formats for enc in encodings]
                random.Random(rep * 7919 + i).shuffle(combos)  # interleaved, in a different order each time
                for fmt, enc in combos:
                    got = await fetch(client, endpoint, item, fmt, enc, timeout)
                    timings[(i, fmt, enc)].append((got["http_ms"], got["parse_ms"]))
                    sizes[(i, fmt, enc)] = (got["wire"], got["size"], got["rows"])
                    if rep == 0 and enc == encodings[0]:
                        check = checks[fmt]
                        if fmt not in USABLE_NOTE or fmt == "csv":
                            problem = compare(item["reference"], got["parsed"], item["send"])
                            if problem is None:
                                check["identical"] += 1
                            else:
                                check["differ"] += 1
                                check["example"] = check["example"] or f"{item['kind']} {item['label']}: {problem}"
                        if got["rows"] > item["send"]:
                            check["send_ignored"] += 1
                        if got["parsed"]["total"] == item["reference"]["total"]:
                            check["total_reported"] += 1
        return workload, checks, timings, sizes


async def throughput(endpoint, workload, fmt, encoding, concurrency, seconds, timeout):
    limits = httpx.Limits(max_connections=concurrency)
    latencies, errors = [], 0
    async with httpx.AsyncClient(http2=True, verify=False, timeout=timeout + 5, limits=limits) as client:
        deadline = time.perf_counter() + seconds

        async def worker(seed):
            nonlocal errors
            order = list(range(len(workload)))
            random.Random(seed).shuffle(order)
            k = 0
            while time.perf_counter() < deadline:
                item = workload[order[k % len(order)]]
                k += 1
                started = time.perf_counter()
                try:
                    await fetch(client, endpoint, item, fmt, encoding, timeout)
                    latencies.append((time.perf_counter() - started) * 1e3)
                except Exception:
                    errors += 1

        started = time.perf_counter()
        await asyncio.gather(*(worker(s) for s in range(concurrency)))
        elapsed = time.perf_counter() - started
    latencies.sort()
    pick = (lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]) if latencies else (lambda q: float("nan"))
    return {"rps": len(latencies) / elapsed, "p50": pick(0.5), "p95": pick(0.95), "errors": errors, "requests": len(latencies)}

# ---------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", default="http://localhost:7010/sparql")
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--encodings", default="identity,gzip", help="Accept-Encoding values to compare")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--send", type=int, default=60, help="the middle tier's send (rows exported)")
    parser.add_argument("--bulk", default="1000,10000,100000", help="row counts of the plain bulk SELECTs; empty for none")
    parser.add_argument("--concurrency", default="1,8,32", help="client counts for the throughput runs; empty to skip")
    parser.add_argument("--seconds", type=float, default=10, help="length of each throughput run")
    parser.add_argument("--throughput-formats", default="qlever-json,sparql-json,tsv",
                        help="formats given a throughput run (the usable ones by default)")
    parser.add_argument("--timeout", type=int, default=60, help="seconds per query (QLever's own limit applies)")
    parser.add_argument("--records", default="", help='JSON {scope: uri} instead of picking from the endpoint')
    parser.add_argument("--out", default="", help="write every measurement here as JSON")
    args = parser.parse_args()
    formats = args.formats.split(",")
    encodings = args.encodings.split(",")

    records = json.loads(args.records) if args.records else pick_records(args.endpoint)
    print(f"records: {records}")
    with contextlib.redirect_stdout(io.StringIO()):  # the services print their SPARQL and startup notes
        workload = asyncio.run(capture_workload(records, args.send))
    workload += bulk_queries([int(n) for n in args.bulk.split(",") if n])
    kinds = sorted({w["kind"] for w in workload}, key=[w["kind"] for w in workload].index)
    print(f"workload: {len(workload)} queries - " + ", ".join(f"{k} {sum(w['kind'] == k for w in workload)}" for k in kinds))

    workload, checks, timings, sizes = asyncio.run(
        latency(args.endpoint, workload, formats, encodings, args.repeats, args.timeout))
    report = {"endpoint": args.endpoint, "records": records, "send": args.send, "repeats": args.repeats,
              "checks": checks, "queries": [], "throughput": []}

    print("\n== Correctness (as a drop-in for process_qlever_results; send and total as the middle tier relies on them)")
    for fmt in formats:
        c = checks[fmt]
        verdict = USABLE_NOTE.get(fmt) or ("identical rows" if not c["differ"] else f"rows differ in {c['differ']} queries")
        print(f"  {fmt:12} {verdict}; identical {c['identical']}/{len(workload)}; send ignored in {c['send_ignored']};"
              f" total reported {c['total_reported']}/{len(workload)}" + (f"\n{'':15}e.g. {c['example']}" if c["example"] else ""))

    def medians(i, fmt, enc):
        rows = timings[(i, fmt, enc)]
        return median([h for h, _ in rows]), median([p for _, p in rows])

    print("\n== Latency against QLever's warm cache: median ms per query (HTTP incl. decompression + parse), summed per kind")
    header = f"  {'kind':12} {'n':>4} {'compute':>8}  " + "  ".join(f"{f + '/' + e[:4]:>18}" for f in formats for e in encodings)
    print(header)
    for kind in kinds + ["ALL (no bulk)"]:
        idx = [i for i, w in enumerate(workload) if (w["kind"] == kind or (kind.startswith("ALL") and w["kind"] != "bulk"))]
        if not idx:
            continue
        compute = sum(float(str(workload[i]["qlever_compute"] or "0").rstrip("ms") or 0) for i in idx)
        cells = []
        for fmt in formats:
            for enc in encodings:
                http = sum(medians(i, fmt, enc)[0] for i in idx)
                parse = sum(medians(i, fmt, enc)[1] for i in idx)
                cells.append(f"{http + parse:9.1f} ({parse:5.1f})")
        print(f"  {kind:12} {len(idx):>4} {compute:8.0f}  " + "  ".join(f"{c:>18}" for c in cells))
    print("  (in brackets: the parse share; compute: QLever's computeResult on the first, uncached run)")

    print("\n== Bytes on the wire, summed per kind (KB)")
    print(f"  {'kind':12} " + "  ".join(f"{f + '/' + e[:4]:>18}" for f in formats for e in encodings))
    for kind in kinds:
        idx = [i for i, w in enumerate(workload) if w["kind"] == kind]
        print(f"  {kind:12} " + "  ".join(f"{sum(sizes[(i, f, e)][0] for i in idx) / 1024:18,.1f}"
                                          for f in formats for e in encodings))

    for i, item in enumerate(workload):
        report["queries"].append({
            "kind": item["kind"], "label": item["label"], "send": item["send"], "cold_ms": round(item["cold_ms"], 1),
            "qlever_compute": item["qlever_compute"], "rows": len(item["reference"]["results"]),
            "formats": {f"{fmt}/{enc}": {"http_ms": round(medians(i, fmt, enc)[0], 2),
                                         "parse_ms": round(medians(i, fmt, enc)[1], 3),
                                         "wire": sizes[(i, fmt, enc)][0], "size": sizes[(i, fmt, enc)][1],
                                         "rows": sizes[(i, fmt, enc)][2]}
                        for fmt in formats for enc in encodings}})

    levels = [int(c) for c in args.concurrency.split(",") if c]
    if levels:
        mixed = [w for w in workload if w["kind"] != "bulk"]
        print(f"\n== Throughput over the middle tier's queries ({len(mixed)}, warm cache), {args.seconds:g} s each")
        print(f"  {'format':12} {'encoding':8} " + "  ".join(f"{'c=' + str(c):>26}" for c in levels))
        for fmt in [f for f in args.throughput_formats.split(",") if f in formats]:
            for enc in encodings:
                cells = []
                for c in levels:
                    r = asyncio.run(throughput(args.endpoint, mixed, fmt, enc, c, args.seconds, args.timeout))
                    report["throughput"].append({"format": fmt, "encoding": enc, "concurrency": c, **r})
                    cells.append(f"{r['rps']:7.0f}/s p50 {r['p50']:5.1f} p95 {r['p95']:6.1f}"
                                 + (f" err {r['errors']}" if r["errors"] else ""))
                print(f"  {fmt:12} {enc:8} " + "  ".join(f"{c:>26}" for c in cells), flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
