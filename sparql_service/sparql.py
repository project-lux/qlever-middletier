"""Parses the SPARQL a QLever ``SERVICE`` clause sends into a vector-search plan.

QLever sends the outer query's prologue, then ``SELECT ?vars { <service body> }``, with any
pushed-down bindings appended just before the closing brace as ``VALUES (?v) { (<iri>) ... }``
(measured against QLever 0.6.0; up to ``service-max-value-rows`` rows, ~87 bytes each). This
parser understands exactly what the service can evaluate - ``vec:`` magic triples about one
subject variable plus one ``VALUES`` block - and refuses anything else with a message, rather than
silently ignoring patterns it cannot honour.

    PREFIX vec: <http://lux.collections.yale.edu/ns/vec#>
    SELECT ?concept ?score WHERE {
      ?concept vec:similarTo <https://lux.collections.yale.edu/data/concept/...> ;   # or "free text"
               vec:score ?score ;          # cosine similarity, xsd:double
               vec:rank ?rank ;            # optional, 1-based
               vec:topK 20 ;               # optional, default 20
               vec:minScore 0.6 ;          # optional: drop hits with a lower cosine
               vec:maxGap 0.1 ;            # optional: drop hits more than this below the best
               vec:ef 1000 ;               # optional: HNSW expansion factor for this query
               vec:rewalk 5 ;              # optional: walk again from the best 5 hits' vectors
               vec:rewalkEf 300 ;          # optional: expansion factor of those second walks
               vec:exact true ;            # optional: score the whole scope, no graph (or false)
               vec:index "concept" ;       # optional; inferred from IRIs otherwise
               vec:mode "constrained" .    # optional: refuse to run without pushed-down VALUES
    }

Several seeds - ``vec:similarTo <a>, <b>, "text"`` (or repeated triples) - combine by ``vec:combine``:

* ``"max"`` (default): one ranking, each record scored by its best match among the seeds. The
  seed with the tightest neighbourhood can fill every slot.
* ``"each"``: every seed's own top hits, interleaved round-robin, so each seed is represented;
  order by ``?rank``, since scores come from different seeds.
* ``"mean"``: one search with the seeds' normalized centroid, "similar to them as a group".

Seed IRIs are left out of the results unless ``vec:includeSeeds true``.

``vec:minScore`` and ``vec:maxGap`` cut the top-K list further: the first is an absolute cosine
floor, the second keeps only hits within that distance of the best one (of each seed's ranking,
for ``"each"``). Raw cosines sit in a band whose position depends on the query and the model, so
a floor that suits one query can empty another; the gap follows the query's own best match.

``vec:ef`` sets the HNSW expansion factor - how many candidates a graph walk keeps in play - for
one query, overriding the service's ``--expansion-factor`` (10,000). A walk can be trapped by a
dense cluster of near-duplicates (hundreds of "quintets (…)" concepts) and miss the real
neighbours; a wider walk escapes it, at a roughly linear cost. Exact scoring of pushed-down
candidates uses no graph and ignores it.

``vec:rewalk N`` walks again from the stored vectors of the best N hits (the service's
``--rewalk``, 10) and reranks the pool; ``vec:rewalkEf`` is the width of those walks
(``--rewalk-ef``, 300). A text query sits away from every record, where a walk misses records
the graph does link. Narrow second walks are cheap and recover a few of them (a top hit missed
for "japanese prints" on dev); wide ones recover more but cost as much as the first walk again.
``vec:exact true`` scores every vector of the scope instead (perfect recall, linear cost: 1-3 s
a query on dev), ``false`` forces the graph; by default a scope at or below the service's
``--exact-below`` size is scored exactly.

The ``VALUES`` block is split off and parsed with one regex pass before the (small) remainder is
tokenized, so a 100k-row pushdown costs tens of milliseconds, not a general SPARQL parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VEC = "http://lux.collections.yale.edu/ns/vec#"
XSD = "http://www.w3.org/2001/XMLSchema#"
DEFAULT_TOP_K = 20
MAX_TOP_K = 10_000
MODES = ("auto", "constrained", "global")
COMBINES = ("max", "each", "mean")
MAX_SEEDS = 100
MAX_EF = 50_000
MAX_REWALK = 50


class SparqlError(ValueError):
    """A query this service cannot evaluate; the message is returned to the caller."""


@dataclass(frozen=True)
class Term:
    kind: str  # "iri" | "var" | "literal"
    value: str
    datatype: str | None = None
    lang: str | None = None

    def __str__(self) -> str:
        return {"iri": f"<{self.value}>", "var": f"?{self.value}"}.get(self.kind, f'"{self.value}"')


@dataclass
class Plan:
    select: list[str]  # the variables the response's head must list, in order
    subject: str  # the variable that receives ranked IRIs
    seeds: list[Term]  # IRIs (use their stored vectors) and/or literals (encode the text)
    score: str | None = None
    rank: str | None = None
    top_k: int = DEFAULT_TOP_K
    scopes: list[str] | None = None
    mode: str = "auto"
    combine: str = "max"
    include_seeds: bool = False
    min_score: float | None = None  # absolute cosine floor
    max_gap: float | None = None  # keep hits within this of the best
    ef: int | None = None  # HNSW expansion factor; the service's default when None
    rewalk: int | None = None  # second walks from the best hits; the service's default when None
    rewalk_ef: int | None = None  # their expansion factor; the service's default when None
    exact: bool | None = None  # score the whole scope; by the service's size threshold when None
    candidates: list[str] | None = None  # pushed-down IRIs for `subject`, if any
    limit: int | None = None
    notes: list[str] = field(default_factory=list)


_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>\#[^\n]*)
  | (?P<iri><[^<>"{}|^`\\\s]*>)
  | (?P<var>[?$][A-Za-z_]\w*)
  | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')(?P<suffix>@[A-Za-z]+(?:-[A-Za-z0-9]+)*|\^\^(?:<[^>]*>|[A-Za-z_][\w-]*:[\w-]*))?
  | (?P<number>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<pname>(?:[A-Za-z_][\w-]*)?:(?:[\w-](?:[\w.-]*[\w-])?)?)
  | (?P<word>[A-Za-z]+)
  | (?P<punct>[{}().;,*])
    """,
    re.X,
)
_IRI = re.compile(r"<([^<>\s]*)>")
# The rest of a VALUES block up to its closing brace, string literals skipped - one C-level scan.
_VALUES_BODY = re.compile(r'[^"}]*(?:"(?:[^"\\]|\\.)*"[^"}]*)*')  # unrolled: long runs in one step
_VALUE_TERM = re.compile(r'<([^<>\s]*)>|(UNDEF)|("(?:[^"\\]|\\.)*"(?:\^\^<[^>]*>|@[\w-]+)?)|([^\s()]+)')


def _tokens(text: str) -> list[tuple[str, str, str | None]]:
    out, pos = [], 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match:
            raise SparqlError(f"cannot read the query at: {text[pos:pos + 40]!r}")
        kind = match.lastgroup if match.lastgroup != "suffix" else "string"
        if kind not in ("ws", "comment"):
            out.append((kind, match.group(kind), match.group("suffix") if kind == "string" else None))
        pos = match.end()
    return out


def _values_keywords(body: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each VALUES keyword: a case-insensitive word followed by '(' or a
    variable. ``str.find`` over an upper-cased copy is several times faster than a regex scan of a
    multi-megabyte pushdown."""
    upper, found, at = body.upper(), [], 0
    while (at := upper.find("VALUES", at)) != -1:
        end = at + 6
        before = body[at - 1] if at else " "
        after = body[end:end + 64].lstrip()
        if not (before.isalnum() or before in "_:/<#") and after[:1] in ("(", "?", "$"):
            found.append((at, len(body) - len(body[end:].lstrip()) if after else end))
        at = end
    return found


def _split_values(body: str) -> tuple[str, list[str], list[list[str | None]]]:
    """Removes the (single) VALUES block from a group body; returns the rest, its variables, rows."""
    found = _values_keywords(body)
    if not found:
        return body, [], []
    if len(found) > 1:
        raise SparqlError("only one VALUES block is supported")
    start, keyword_end = found[0]
    rest = body[keyword_end:]
    if rest.startswith("("):
        close = rest.index(")")
        variables = re.findall(r"[?$](\w+)", rest[:close])
        rest = rest[close + 1:]
    else:
        var = re.match(r"[?$](\w+)", rest)
        variables, rest = [var.group(1)], rest[var.end():]
    open_brace = rest.index("{")
    end = _VALUES_BODY.match(rest, open_brace + 1).end()
    if end >= len(rest) or rest[end] != "}":
        raise SparqlError("unterminated VALUES block")
    block = rest[open_brace + 1 : end]
    remainder = body[:start] + rest[end + 1 :]
    if len(variables) == 1 and '"' not in block and "UNDEF" not in block.upper():
        # QLever's pushdown: IRIs only, `(<a>) (<b>) ...` - straight to strings in one pass.
        rows = [[iri] for iri in _IRI.findall(block)]
    elif len(variables) == 1:  # `?v { a b }` or `(?v) { (a) (b) }`: terms never include parentheses
        rows = [[_value(m)] for m in _VALUE_TERM.finditer(block)]
    else:
        rows = []
        for row in re.findall(r"\(([^()]*)\)", block):
            terms = [_value(m) for m in _VALUE_TERM.finditer(row)]
            if len(terms) != len(variables):
                raise SparqlError("a VALUES row has the wrong number of terms")
            rows.append(terms)
    return remainder, variables, rows


def _value(match: re.Match) -> str | None:
    iri, undef, literal, other = match.groups()
    if iri is not None:
        return iri
    if undef:
        return None
    return literal or other


def _matching_brace(tokens, open_index: int) -> int:
    depth = 0
    for i in range(open_index, len(tokens)):
        if tokens[i][1] == "{":
            depth += 1
        elif tokens[i][1] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise SparqlError("unbalanced braces")


def parse(query: str) -> Plan:
    # The VALUES block is the only large part; cut it out before tokenizing the rest.
    remainder, value_vars, rows = _split_values(query)
    tokens = _tokens(remainder)
    prefixes: dict[str, str] = {}
    i = 0
    while i < len(tokens) and tokens[i][1].upper() in ("PREFIX", "BASE"):
        if tokens[i][1].upper() == "PREFIX":
            prefixes[tokens[i + 1][1][:-1]] = tokens[i + 2][1][1:-1]
            i += 3
        else:
            i += 2
    if i >= len(tokens) or tokens[i][1].upper() != "SELECT":
        raise SparqlError("expected a SELECT query")
    select, star = [], False
    i += 1
    while i < len(tokens) and tokens[i][1] != "{" and tokens[i][1].upper() != "WHERE":
        kind, text, _ = tokens[i]
        if kind == "var":
            select.append(text[1:])
        elif text == "*":
            star = True
        elif text.upper() not in ("DISTINCT", "REDUCED"):
            raise SparqlError(f"unsupported projection {text!r} (only plain variables)")
        i += 1
    if i < len(tokens) and tokens[i][1].upper() == "WHERE":
        i += 1
    if i >= len(tokens) or tokens[i][1] != "{":
        raise SparqlError("expected '{' after SELECT")
    close = _matching_brace(tokens, i)
    group = tokens[i + 1 : close]
    limit = _limit(tokens[close + 1 :])
    # QLever wraps a body that is a lone subquery in an extra group: `{ { SELECT ... } }`.
    while group and group[0][1] == "{" and _matching_brace(group, 0) == len(group) - 1:
        group = group[1:-1]
    if group and group[0][1].upper() == "SELECT":
        inner = parse(_detokenize(prefixes, group))
        inner.select = select or inner.select
        inner.limit = limit or inner.limit
        return _finish(inner, value_vars, rows)

    triples = _triples(group, prefixes)
    plan = _plan(triples, select, star)
    plan.limit = limit
    return _finish(plan, value_vars, rows)


def _detokenize(prefixes, tokens) -> str:
    prologue = "".join(f"PREFIX {p}: <{iri}>\n" for p, iri in prefixes.items())
    return prologue + " ".join(text + (suffix or "") for _, text, suffix in tokens)


def _limit(tail) -> int | None:
    for j, (_, text, _) in enumerate(tail):
        if text.upper() == "LIMIT" and j + 1 < len(tail):
            return int(tail[j + 1][1])
    return None


def _term(token, prefixes) -> Term:
    kind, text, suffix = token
    if kind == "iri":
        return Term("iri", text[1:-1])
    if kind == "var":
        return Term("var", text[1:])
    if kind == "pname":
        prefix, _, local = text.partition(":")
        if prefix not in prefixes:
            raise SparqlError(f"undeclared prefix {prefix!r}")
        return Term("iri", prefixes[prefix] + local)
    if kind == "string":
        value = bytes(text[1:-1], "utf-8").decode("unicode_escape") if "\\" in text else text[1:-1]
        if suffix and suffix.startswith("@"):
            return Term("literal", value, lang=suffix[1:])
        if suffix:
            dt = suffix[2:]
            return Term("literal", value, datatype=dt[1:-1] if dt.startswith("<") else _term(("pname", dt, None), prefixes).value)
        return Term("literal", value)
    if kind == "number":
        return Term("literal", text, datatype=XSD + ("integer" if re.fullmatch(r"[+-]?\d+", text) else "decimal"))
    if kind == "word" and text == "a":
        return Term("iri", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    if kind == "word" and text in ("true", "false"):
        return Term("literal", text, datatype=XSD + "boolean")
    raise SparqlError(f"unexpected {text!r} in the graph pattern")


def _triples(tokens, prefixes) -> list[tuple[Term, Term, Term]]:
    """Subject/predicate/object lists with ';' and ',' - no nested groups, paths or blank nodes."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i][1] in ("{", "}", "(", ")") or tokens[i][1].upper() in ("FILTER", "OPTIONAL", "UNION", "BIND", "MINUS", "GRAPH", "SERVICE"):
            raise SparqlError(f"the vector service evaluates only vec: triples, not {tokens[i][1]!r}")
        subject = _term(tokens[i], prefixes)
        i += 1
        while True:
            predicate = _term(tokens[i], prefixes)
            i += 1
            while True:
                out.append((subject, predicate, _term(tokens[i], prefixes)))
                i += 1
                if i < len(tokens) and tokens[i][1] == ",":
                    i += 1
                    continue
                break
            if i < len(tokens) and tokens[i][1] == ";":
                i += 1
                if i < len(tokens) and tokens[i][1] in (".", "}"):  # trailing ';'
                    break
                continue
            break
        if i < len(tokens) and tokens[i][1] == ".":
            i += 1
    return out


def _plan(triples, select: list[str], star: bool) -> Plan:
    subjects = {s for s, _, _ in triples}
    if len(subjects) != 1 or next(iter(subjects)).kind != "var":
        raise SparqlError("the vec: pattern needs exactly one subject, a variable")
    subject = next(iter(subjects)).value
    found: dict[str, Term] = {}
    seeds: list[Term] = []
    for _, predicate, obj in triples:
        if predicate.kind != "iri" or not predicate.value.startswith(VEC):
            raise SparqlError(f"the vector service evaluates only vec: predicates, not <{predicate.value}>")
        name = predicate.value[len(VEC):]
        if name == "similarTo":  # the one repeatable predicate: several seeds
            if obj.kind == "var":
                raise SparqlError("vec:similarTo needs IRIs or text literals, not a variable")
            if obj not in seeds:
                seeds.append(obj)
            continue
        if name in found:
            raise SparqlError(f"vec:{name} given twice")
        found[name] = obj
    if not seeds:
        raise SparqlError("vec:similarTo is required")
    if len(seeds) > MAX_SEEDS:
        raise SparqlError(f"at most {MAX_SEEDS} vec:similarTo seeds")
    plan = Plan(select=[], subject=subject, seeds=seeds)
    for name, obj in found.items():
        if name in ("score", "rank"):
            if obj.kind != "var":
                raise SparqlError(f"vec:{name} binds a variable")
            setattr(plan, name, obj.value)
        elif name == "topK":
            plan.top_k = max(1, min(MAX_TOP_K, int(float(obj.value))))
        elif name == "index":
            plan.scopes = [s.strip() for s in obj.value.split(",") if s.strip()]
        elif name == "mode":
            if obj.value not in MODES:
                raise SparqlError(f"vec:mode is one of {', '.join(MODES)}")
            plan.mode = obj.value
        elif name == "combine":
            if obj.value not in COMBINES:
                raise SparqlError(f"vec:combine is one of {', '.join(COMBINES)}")
            plan.combine = obj.value
        elif name in ("minScore", "maxGap"):
            try:
                number = float(obj.value)
            except ValueError:
                raise SparqlError(f"vec:{name} is a number") from None
            if name == "minScore" and not -1.0 <= number <= 1.0:
                raise SparqlError("vec:minScore is a cosine, between -1 and 1")
            if name == "maxGap" and number < 0:
                raise SparqlError("vec:maxGap is not negative")
            setattr(plan, "min_score" if name == "minScore" else "max_gap", number)
        elif name == "ef":
            try:
                number = int(float(obj.value))
            except ValueError:
                raise SparqlError("vec:ef is a number") from None
            if not 1 <= number <= MAX_EF:
                raise SparqlError(f"vec:ef is between 1 and {MAX_EF}")
            plan.ef = number
        elif name == "rewalk":
            try:
                number = int(float(obj.value))
            except ValueError:
                raise SparqlError("vec:rewalk is a number") from None
            if not 0 <= number <= MAX_REWALK:
                raise SparqlError(f"vec:rewalk is between 0 and {MAX_REWALK}")
            plan.rewalk = number
        elif name == "rewalkEf":
            try:
                number = int(float(obj.value))
            except ValueError:
                raise SparqlError("vec:rewalkEf is a number") from None
            if not 1 <= number <= MAX_EF:
                raise SparqlError(f"vec:rewalkEf is between 1 and {MAX_EF}")
            plan.rewalk_ef = number
        elif name == "exact":
            if obj.value.lower() not in ("true", "false"):
                raise SparqlError("vec:exact is true or false")
            plan.exact = obj.value.lower() == "true"
        elif name == "includeSeeds":
            if obj.value.lower() not in ("true", "false"):
                raise SparqlError("vec:includeSeeds is true or false")
            plan.include_seeds = obj.value.lower() == "true"
        else:
            raise SparqlError(f"unknown predicate vec:{name}")
    bound = [v for v in (plan.subject, plan.score, plan.rank) if v]
    plan.select = bound if star else select
    unknown = [v for v in plan.select if v not in bound]
    if unknown:
        plan.notes.append(f"unbound: {', '.join('?' + v for v in unknown)}")
    return plan


def _finish(plan: Plan, value_vars: list[str], rows: list[list[str | None]]) -> Plan:
    if value_vars:
        if value_vars != [plan.subject]:
            other = [v for v in value_vars if v != plan.subject]
            if plan.subject not in value_vars or any(row[value_vars.index(v)] is not None for v in other for row in rows):
                raise SparqlError(f"VALUES may bind only ?{plan.subject}, not {', '.join('?' + v for v in other)}")
        column = value_vars.index(plan.subject)
        plan.candidates = [row[column] for row in rows if row[column] is not None]
    if plan.limit:
        plan.top_k = min(plan.top_k, plan.limit) if plan.top_k != DEFAULT_TOP_K else plan.limit
    return plan
