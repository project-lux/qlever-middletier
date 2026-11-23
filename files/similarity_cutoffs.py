"""Shows where similarity falls off, to choose vec:topK, vec:minScore and vec:maxGap.

For each text it asks the vector sidecar for the top hits in a scope and prints the score curve,
how many hits each candidate cutoff would keep, and (with --qlever) the names, so the point where
the hits stop being similar can be read off rather than guessed:

    python files/similarity_cutoffs.py --scope concept "quantum physics" "roman coins"
    python files/similarity_cutoffs.py --scope concept --qlever http://localhost:7011/ --names 30 "baroque"

Only the standard library; the sidecar and QLever are reached over HTTP.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--service", default="http://127.0.0.1:8081/sparql", help="the vector sidecar")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-scores", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8",
                        help="comma-separated vec:minScore values to count")
    parser.add_argument("--max-gaps", default="0.02,0.05,0.1,0.15,0.2",
                        help="comma-separated vec:maxGap values to count")
    parser.add_argument("--qlever", default="", help="SPARQL endpoint for names; without it, IRIs are shown")
    parser.add_argument("--names", type=int, default=20, help="hits to list; 0 lists none")
    args = parser.parse_args()
    floors = [float(v) for v in args.min_scores.split(",") if v]
    gaps = [float(v) for v in args.max_gaps.split(",") if v]
    for text in args.texts:
        hits = similar(args.service, text, args.scope, args.top_k)
        shown = [uri for uri, _ in hits[: args.names]]
        labels = names(args.qlever, args.scope, shown) if args.qlever and shown else {}
        report(text, hits, floors, gaps, labels, args.names)


if __name__ == "__main__":
    main()
