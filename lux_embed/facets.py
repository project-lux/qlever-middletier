"""Stage 1: incoming facets from QLever - what *other* records say about a work, agent or set.

The design doc imagined bulk TSV exports; batched ``VALUES`` queries do the same job without
multi-GB intermediate files, resume with the embedding loop, and stay inside QLever's query
timeout. Each batch answers ``(target, kind, value, count)`` rows, grouped per target and kind.

Ranking:
* works   - by frequency (a work's carrying objects mostly agree; this just caps edition noise);
* agents, sets - by tf-idf, ``count * log((N + 1) / (df + 1))``, where ``df`` is how many objects
  carry that value overall, so "photograph" or "England" stops crowding out what is distinctive.

The graph patterns use the ``lux:`` predicates the LUX QLever export emits. A pattern whose
predicate the endpoint lacks (MDS has no ``lux:agentOfWorkBeginning``, full LUX does) is dropped
once at start-up instead of failing every batch.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

PREFIX = "PREFIX lux: <https://lux.collections.yale.edu/ns/>\n"

# kind -> (graph pattern binding ?v from the joining variable ?x, predicate(s) it needs)
ITEM_FACETS = {
    "creator": ("?x lux:agentOfItemBeginning ?v", ["agentOfItemBeginning"]),
    "material": ("?x lux:material ?v", ["material"]),
    "object type": ("?x lux:itemClassification ?v", ["itemClassification"]),
    "place": ("?x lux:placeOfItemBeginning ?v", ["placeOfItemBeginning"]),
    "subject": ("?x lux:carries ?w . ?w lux:workAboutConcept ?v", ["carries", "workAboutConcept"]),
}
WORK_FACETS = {
    "subject": ("?x lux:workAboutConcept ?v", ["workAboutConcept"]),
    "work type": ("?x lux:workClassification ?v", ["workClassification"]),
}

# scope -> list of (how a member ?x relates to ?target, member facets, kinds kept, predicates needed).
# A kind kept may be (member facet, kind it is stored as): an agent's subjects are two different things
# - what the objects they made depict, and what the works they wrote or made are about - and a card
# that merged them could not tell an author on Edo prints from an Edo print designer.
SCOPE_FACETS = {
    "work": [("?x lux:carries ?target", ITEM_FACETS, ["creator", "material", "object type", "place"], ["carries"])],
    "agent": [
        ("?x lux:agentOfItemBeginning ?target", ITEM_FACETS,
         ["place", "object type", "material", ("subject", "depicted subject")], ["agentOfItemBeginning"]),
        ("?x lux:agentOfWorkBeginning ?target", WORK_FACETS,
         [("subject", "work subject"), "work type"], ["agentOfWorkBeginning"]),
    ],
    "set": [("?x lux:itemMemberOfSet ?target", ITEM_FACETS, ["creator", "object type", "subject"], ["itemMemberOfSet"])],
}
# scope -> {span: [(pattern binding ?s and ?e for a ?target, predicates it needs), ...]}. A span's
# patterns are UNIONed and its min start / max end become one date range on the card.
SCOPE_DATES = {
    "work": {"dates": [("?x lux:carries ?target . ?x lux:startOfItemBeginning ?s . ?x lux:endOfItemBeginning ?e",
                        ["carries", "startOfItemBeginning", "endOfItemBeginning"])]},
    "agent": {
        # when they made things: objects produced and works created
        "dates": [
            ("?x lux:agentOfItemBeginning ?target . ?x lux:startOfItemBeginning ?s . ?x lux:endOfItemBeginning ?e",
             ["agentOfItemBeginning", "startOfItemBeginning", "endOfItemBeginning"]),
            ("?x lux:agentOfWorkBeginning ?target . ?x lux:startOfWorkBeginning ?s . ?x lux:endOfWorkBeginning ?e",
             ["agentOfWorkBeginning", "startOfWorkBeginning", "endOfWorkBeginning"]),
        ],
        # when works they wrote or published came out: on full LUX far more works have this than a
        # creation date (24M against 0.6M), but for an author of old works it is mostly modern
        # editions, so it is its own range rather than stretching "Active" to the present
        "published": [
            ("?x lux:agentOfWorkBeginning ?target . ?x lux:startOfWorkPublication ?s . ?x lux:endOfWorkPublication ?e",
             ["agentOfWorkBeginning", "startOfWorkPublication", "endOfWorkPublication"]),
            ("?x lux:agentOfWorkPublication ?target . ?x lux:startOfWorkPublication ?s . ?x lux:endOfWorkPublication ?e",
             ["agentOfWorkPublication", "startOfWorkPublication", "endOfWorkPublication"]),
        ],
    },
}
#: Kinds whose values are a (start year, end year) range rather than records.
DATE_KINDS = frozenset(span for spans in SCOPE_DATES.values() for span in spans)
TOP = {"work": 5, "agent": 8, "set": 10}
TFIDF_SCOPES = {"agent", "set"}
UBIQUITOUS = 0.5  # a value on more than half of all objects ("works") says nothing about anyone
# Kinds kept however common. A work's carrier type is what tells a book from a journal, and on full
# LUX Books is on 77% of objects; the blanket "Collection Item" (99.8%) is dropped by the cards.
KEEP_UBIQUITOUS = {("work", "object type")}
# Targets per batched query. On full LUX a query costs QLever about 0.2 s fixed plus ~29 us per
# target (it scans each facet predicate and joins), so big batches pay: work extraction ran at
# 1.8k targets/s at 400 and 18k/s at 8000, and agent at 8000 ran at 10k/s with no timeouts.
# A batch that does hit the server timeout is retried in halves.
FACET_BATCH = 8000


class QLever:
    def __init__(self, endpoint: str, timeout: float = 120):
        self.endpoint, self.timeout = endpoint, timeout

    def rows(self, query: str) -> list[list[str]]:
        body = urllib.parse.urlencode({"query": PREFIX + query}).encode()
        request = urllib.request.Request(self.endpoint, data=body, headers={"Accept": "text/tab-separated-values"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            text = response.read().decode()
        reader = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader, None)  # header
        return [[_term(cell) for cell in row] for row in reader if row]

    def predicates(self) -> set[str]:
        return {
            row[0].rsplit("/", 1)[-1]
            for row in self.rows("SELECT DISTINCT ?p WHERE { ?s ?p ?o } ")
            if row[0].startswith("https://lux.collections.yale.edu/ns/")
        }


def _term(cell: str) -> str:
    if cell.startswith("<") and cell.endswith(">"):
        return cell[1:-1]
    if cell.startswith('"'):
        return cell[1 : cell.rindex('"')]
    return cell


class Facets:
    """Batched incoming-facet lookups for one scope."""

    def __init__(self, qlever: QLever, scope: str, cache_dir: str):
        self.qlever, self.scope = qlever, scope
        available = qlever.predicates()
        self.plans = []
        for join, member_facets, kinds, needs in SCOPE_FACETS.get(scope, []):
            if not set(needs) <= available:
                continue
            usable = {}
            for spec in kinds:
                facet, kind = (spec, spec) if isinstance(spec, str) else spec
                if facet in member_facets and set(member_facets[facet][1]) <= available:
                    usable[kind] = member_facets[facet]
            if usable:
                self.plans.append((join, usable))
        # span -> the patterns of it this endpoint can answer
        self.date_spans = {}
        for span, sources in SCOPE_DATES.get(scope, {}).items():
            patterns = [pattern for pattern, needs in sources if set(needs) <= available]
            if patterns:
                self.date_spans[span] = patterns
        self.df: dict[str, dict[str, int]] = {}
        self.total = 1
        if self.plans:
            self._load_document_frequencies(os.path.join(cache_dir, "facet_df.json"))

    @property
    def enabled(self) -> bool:
        return bool(self.plans)

    def _load_document_frequencies(self, path: str) -> None:
        """How many objects (or works) carry each value, per kind - one GROUP BY per kind, cached on
        disk and shared by every scope, so each kind is counted once per dataset."""
        cached = json.load(open(path)) if os.path.exists(path) else {"df": {}, "total": 0}
        kinds = {kind: pattern for _, usable in self.plans for kind, (pattern, _) in usable.items()}
        missing = {kind: pattern for kind, pattern in kinds.items() if kind not in cached["df"]}
        for kind, pattern in missing.items():
            print(f"facets: counting how many objects carry each {kind} (once per dataset)...", flush=True)
            started = time.time()
            rows = self.qlever.rows(f"SELECT ?v (COUNT(DISTINCT ?x) AS ?n) WHERE {{ {pattern} }} GROUP BY ?v")
            cached["df"][kind] = {v: int(n) for v, n in rows}
            print(f"facets: {len(rows):,} {kind} values counted in {time.time() - started:,.0f}s", flush=True)
        if not cached["total"]:
            cached["total"] = int(self.qlever.rows("SELECT (COUNT(?x) AS ?n) WHERE { ?x a lux:Item }")[0][0])
        if missing or not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            json.dump(cached, open(path, "w"))
        self.df, self.total = cached["df"], cached["total"]

    def fetch(self, targets: list[str]) -> dict[str, dict]:
        """``{target IRI: {kind: [value IRI, ...] ranked, span: (min, max)}}`` for one batch, a span
        being one of DATE_KINDS (``dates``, ``published``)."""
        out: dict[str, dict] = defaultdict(dict)
        if not targets:
            return out
        for join, usable in self.plans:
            counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
            for target, kind, value, n in self._grouped(targets, join, usable):
                counts[target][kind][value] = int(n)
            for target, kinds in counts.items():
                for kind, values in kinds.items():
                    out[target].setdefault(kind, [])
                    out[target][kind] += self._rank(kind, values)
        for span, patterns in self.date_spans.items():
            for target, start, end in self._dates(targets, patterns):
                out[target][span] = (start[:4].lstrip("0") or "0", end[:4].lstrip("0") or "0")
        return out

    def _rank(self, kind: str, values: dict[str, int]) -> list[str]:
        df = self.df.get(kind, {})
        if (self.scope, kind) not in KEEP_UBIQUITOUS:
            values = {v: n for v, n in values.items() if df.get(v, 0) <= UBIQUITOUS * self.total}
        if self.scope in TFIDF_SCOPES:
            score = {v: n * math.log((self.total + 1) / (df.get(v, 0) + 1)) for v, n in values.items()}
        else:
            score = values
        return sorted(score, key=score.get, reverse=True)[: TOP[self.scope]]

    def _grouped(self, targets, join, usable):
        branches = " UNION ".join(f'{{ {pattern} BIND("{kind}" AS ?k) }}' for kind, (pattern, _) in usable.items())
        query = (
            f"SELECT ?target ?k ?v (COUNT(DISTINCT ?x) AS ?n) WHERE {{ VALUES ?target {{ {_values(targets)} }} "
            f"{join} . {branches} }} GROUP BY ?target ?k ?v"
        )
        return self._run(query, targets, lambda part: self._grouped(part, join, usable))

    def _dates(self, targets, patterns):
        union = " UNION ".join(f"{{ {pattern} }}" for pattern in patterns)
        query = (
            f"SELECT ?target (MIN(?s) AS ?start) (MAX(?e) AS ?end) WHERE {{ VALUES ?target {{ {_values(targets)} }} "
            f"{union} }} GROUP BY ?target"
        )
        return self._run(query, targets, lambda part: self._dates(part, patterns))

    def _run(self, query, targets, retry):
        """Runs one batch; on a timeout or error, halves the batch, and drops a lone target that
        still fails (one prolific agent must not stall the whole build)."""
        try:
            return self.qlever.rows(query)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if len(targets) == 1:
                print(f"facets: giving up on {targets[0]} ({error})", flush=True)
                return []
            print(f"facets: a batch of {len(targets)} failed ({error}); retrying it in halves", flush=True)
            half = len(targets) // 2
            return [*retry(targets[:half]), *retry(targets[half:])]


def _values(iris: list[str]) -> str:
    return " ".join(f"<{iri}>" for iri in iris)
