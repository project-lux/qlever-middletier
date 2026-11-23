"""Shows where similarity falls off, to choose vec:topK, vec:minScore and vec:maxGap.

For each text it asks the vector sidecar for the top hits in a scope and prints the score curve,
how many hits each candidate cutoff would keep, and (with --qlever) the names, so the point where
the hits stop being similar can be read off rather than guessed:

    python files/similarity_cutoffs.py --scope concept "quantum physics" "roman coins"
    python files/similarity_cutoffs.py --scope concept --qlever http://localhost:7011/ --names 30 "baroque"

When the sidecar is only reachable from QLever's host (it normally listens on loopback there),
``--through`` sends the same SERVICE call through QLever instead; ``--service`` is then the URL as
QLever sees it, and the names come back in the same query:

    python files/similarity_cutoffs.py --through http://qlever-host:7010/sparql "quantum physics"

``--recall`` (with ``--through``) checks the HNSW walk rather than the cutoffs. It scores exactly,
with no graph, the records that share a word with the text, then runs the unconstrained search at
each ``--efs`` value and reports how many of those exact top hits the walk found. A walk trapped in
a dense cluster shows up as low recall and a lower best score:

    python files/similarity_cutoffs.py --through http://qlever-host:7010/sparql --recall --efs default,1000,3000 "quantum entanglement"

``--try`` compares other ways of searching, each a comma-separated set of options:

    --try default --try ef=3000 --try ef=3000,rewalk=5 --try exact

When the records sharing a word are too many for QLever to push down (a common word such as
"new"), that text is reported and skipped.

A pooled hit counts only if it scores above the walk's last result, because more than top-K records
outside the pool may outscore it. The pool is a lower bound on what is relevant (it only holds
records sharing a word), so full recall means the walk found every pooled hit that belonged in its
top-K, not that nothing better exists outside the pool; a best score below the pool's own best is
a miss either way.

Only the standard library; the sidecar and QLever are reached over HTTP.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

VEC = "http://lux.collections.yale.edu/ns/vec#"
LUX = "https://lux.collections.yale.edu/ns/"
MARKS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500)


def sparql_literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def similar(service: str, text: str, scope: str, top_k: int) -> list[tuple[str, float]]:
    query = (f"PREFIX vec: <{VEC}> SELECT ?uri ?score {{ ?uri vec:similarTo {sparql_literal(text)} ; "
             f'vec:score ?score ; vec:topK {top_k} ; vec:index "{scope}" }}')
    request = urllib.request.Request(service, query.encode("utf-8"), {"Content-Type": "application/sparql-query"})
    with urllib.request.urlopen(request, timeout=120) as response:
        rows = json.load(response)["results"]["bindings"]
    return [(row["uri"]["value"], float(row["score"]["value"])) for row in rows]


def similar_through(qlever: str, service: str, text: str, scope: str, top_k: int):
    """The same SERVICE call, made by QLever; returns the hits and their names."""
    query = (f"PREFIX vec: <{VEC}> PREFIX lux: <{LUX}> SELECT ?uri ?score (SAMPLE(?n) AS ?name) {{ "
             f"SERVICE <{service}> {{ ?uri vec:similarTo {sparql_literal(text)} ; vec:score ?score ; "
             f'vec:topK {top_k} ; vec:index "{scope}" }} OPTIONAL {{ ?uri lux:{scope}PrimaryName ?n }} }} '
             f"GROUP BY ?uri ?score ORDER BY DESC(?score)")
    data = urllib.parse.urlencode({"query": query}).encode()
    request = urllib.request.Request(qlever, data, {"Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            rows = json.load(response)["results"]["bindings"]
    except urllib.error.HTTPError as error:
        raise SystemExit(f"QLever: {error.read().decode('utf-8', 'replace')[:500]}") from None
    hits = [(row["uri"]["value"], float(row["score"]["value"])) for row in rows]
    return hits, {row["uri"]["value"]: row["name"]["value"] for row in rows if "name" in row}


class QLeverError(Exception):
    pass


def qlever_rows(qlever: str, query: str) -> list[dict]:
    data = urllib.parse.urlencode({"query": query}).encode()
    request = urllib.request.Request(qlever, data, {"Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)["results"]["bindings"]
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        try:
            message = json.loads(body).get("exception", body)
        except ValueError:
            message = body
        raise QLeverError(message[:500]) from None


def variant_options(variant: str) -> str:
    """``default``, ``exact`` or ``ef=3000,rewalk=5`` -> extra vec: triples."""
    if variant == "default":
        return ""
    out = []
    for part in variant.split(","):
        name, _, value = part.partition("=")
        if name == "exact":
            out.append(f"vec:exact {value or 'true'}")
        elif name in ("ef", "rewalk") and value.isdigit():
            out.append(f"vec:{name} {value}")
        else:
            raise SystemExit(f"cannot read {part!r} in --try {variant!r}: use ef=N, rewalk=N or exact")
    return "".join(f" ; {option}" for option in out)


def exact_pool(qlever: str, service: str, text: str, scope: str, n: int) -> list[tuple[str, float]]:
    """The top ``n`` by exact cosine among records sharing a word with ``text``: QLever pushes them
    to the sidecar as VALUES, which it scores without the graph. ``constrained`` makes a pool too
    big to push an error rather than a silent graph walk."""
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]
    if not words:
        return []
    union = " UNION ".join(f'{{ SERVICE view:{scope}Words {{ [ view:column-word "{w}" ; view:column-uri ?uri ] }} }}'
                           for w in words)
    query = (f"PREFIX vec: <{VEC}> PREFIX view: <https://qlever.cs.uni-freiburg.de/materializedView/> "
             f"SELECT ?uri ?score {{ {{ SELECT DISTINCT ?uri {{ {union} }} }} SERVICE <{service}> {{ "
             f"?uri vec:similarTo {sparql_literal(text)} ; vec:score ?score ; vec:topK {n} ; "
             f'vec:index "{scope}" ; vec:mode "constrained" }} }} ORDER BY DESC(?score)')
    return [(row["uri"]["value"], float(row["score"]["value"])) for row in qlever_rows(qlever, query)]


def walk(qlever: str, service: str, text: str, scope: str, top_k: int, variant: str):
    query = (f"PREFIX vec: <{VEC}> SELECT ?uri ?score {{ SERVICE <{service}> {{ "
             f"?uri vec:similarTo {sparql_literal(text)} ; vec:score ?score ; vec:topK {top_k} ; "
             f'vec:index "{scope}"' + variant_options(variant) + " } } ORDER BY DESC(?score)")
    started = time.perf_counter()
    rows = qlever_rows(qlever, query)
    return [(row["uri"]["value"], float(row["score"]["value"])) for row in rows], time.perf_counter() - started


def recall_report(args, text: str) -> None:
    try:
        exact = exact_pool(args.through, args.service, text, args.scope, args.recall_n)
    except QLeverError as error:
        reason = ("too many records share a word to push down; skipped"
                  if "no VALUES arrived" in str(error) else str(error)[:200])
        print(f"\n== {text!r}: {reason}")
        return
    print(f"\n== {text!r}: exact top {len(exact)} of the records sharing a word"
          + (f", best {exact[0][1]:.3f}" if exact else " (none share a word; nothing to compare)"))
    if not exact:
        return
    variants = args.variants or [e if e == "default" else f"ef={e}" for e in args.efs.split(",")]
    for variant in variants:
        hits, seconds = walk(args.through, args.service, text, args.scope, args.top_k, variant)
        # A pooled hit belongs in the walk's top-K only if it beats the walk's last result: more
        # than K records outside the pool can outscore it, and then leaving it out is right.
        cutoff = hits[-1][1] if len(hits) >= args.top_k else -2.0
        due = {uri for uri, score in exact if score > cutoff}
        found = len(due & {uri for uri, _ in hits})
        best = f"{hits[0][1]:.3f}" if hits else "-"
        print(f"   {variant:>16}: found {found:>3}/{len(due)} pooled hits that belong in the top {args.top_k}"
              f" (above {max(cutoff, 0):.3f}), best {best}, {seconds * 1e3:6.0f} ms")


def names(endpoint: str, scope: str, uris: list[str]) -> dict[str, str]:
    if not uris:
        return {}
    values = " ".join(f"<{uri}>" for uri in uris)
    query = (f"PREFIX lux: <{LUX}> SELECT ?uri (SAMPLE(?n) AS ?name) {{ VALUES ?uri {{ {values} }} "
             f"?uri lux:{scope}PrimaryName ?n }} GROUP BY ?uri")
    data = urllib.parse.urlencode({"query": query}).encode()
    request = urllib.request.Request(endpoint, data, {"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        rows = json.load(response)["results"]["bindings"]
    return {row["uri"]["value"]: row["name"]["value"] for row in rows}


def report(text: str, hits: list[tuple[str, float]], floors: list[float], gaps: list[float],
           labels: dict[str, str], show: int) -> None:
    print(f"\n== {text!r}: {len(hits)} hits")
    if not hits:
        return
    best = hits[0][1]
    marks = [m for m in MARKS if m <= len(hits)]
    print("   score at rank  " + "  ".join(f"{m:>5}" for m in marks))
    print("                  " + "  ".join(f"{hits[m - 1][1]:5.3f}" for m in marks))
    print("   kept by minScore " + "  ".join(f"{f}: {sum(s >= f for _, s in hits)}" for f in floors))
    print("   kept by maxGap   " + "  ".join(f"{g}: {sum(s >= best - g for _, s in hits)}" for g in gaps))
    if show:
        for rank, (uri, score) in enumerate(hits[:show], 1):
            print(f"   {rank:4}  {score:.3f}  {best - score:.3f} below best  {labels.get(uri, uri)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("texts", nargs="+", help="queries, as a user would type them after 'similar to'")
    parser.add_argument("--scope", default="concept")
    parser.add_argument("--service", default="http://127.0.0.1:8081/sparql",
                        help="the vector sidecar (as QLever sees it, with --through)")
    parser.add_argument("--through", default="", help="QLever endpoint to make the SERVICE call through")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-scores", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8",
                        help="comma-separated vec:minScore values to count")
    parser.add_argument("--max-gaps", default="0.02,0.05,0.1,0.15,0.2",
                        help="comma-separated vec:maxGap values to count")
    parser.add_argument("--qlever", default="", help="SPARQL endpoint for names; without it, IRIs are shown")
    parser.add_argument("--names", type=int, default=20, help="hits to list; 0 lists none")
    parser.add_argument("--recall", action="store_true", help="measure HNSW recall instead (needs --through)")
    parser.add_argument("--efs", default="default,300,1000,3000,10000", help="vec:ef values to try with --recall")
    parser.add_argument("--try", dest="variants", action="append", default=[],
                        help="with --recall, a way to search: default, exact, or e.g. ef=3000,rewalk=5; repeatable")
    parser.add_argument("--recall-n", type=int, default=10, help="exact top hits to look for with --recall")
    args = parser.parse_args()
    if args.recall:
        if not args.through:
            parser.error("--recall goes through QLever: give --through")
        for text in args.texts:
            recall_report(args, text)
        return
    floors = [float(v) for v in args.min_scores.split(",") if v]
    gaps = [float(v) for v in args.max_gaps.split(",") if v]
    for text in args.texts:
        if args.through:
            hits, labels = similar_through(args.through, args.service, text, args.scope, args.top_k)
        else:
            hits = similar(args.service, text, args.scope, args.top_k)
            shown = [uri for uri, _ in hits[: args.names]]
            labels = names(args.qlever, args.scope, shown) if args.qlever and shown else {}
        report(text, hits, floors, gaps, labels, args.names)


if __name__ == "__main__":
    main()
