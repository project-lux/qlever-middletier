"""Related-list query construction.

For each scope/related-list, ``config/related_list_scopes.json`` gives a set of
``from-to`` field paths and the scope each one lands in. Every path becomes two
things: an ``OPTIONAL { SELECT ... COUNT }`` fragment in one big count query,
and a cheap ``LIMIT 1`` existence probe used when deciding whether to emit the
HAL link. Both are ordered by a hand-tuned heuristic so the HAL check can
short-circuit on the first hit.

This used to live on ``MTConfig`` and reach into ``SparqlTranslator`` for the
predicate vocabulary; it now reads ``query.predicates`` directly, so building a
related-list query no longer needs a translator or a middle tier.
"""

from __future__ import annotations

import json

from qleverlux.query.predicates import SCOPE_FIELDS

PREFIXES = """
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX qlss: <https://qlever.cs.uni-freiburg.de/spatialSearch/>
PREFIX textSearch: <https://qlever.cs.uni-freiburg.de/textSearch/>
PREFIX lux: <https://lux.collections.yale.edu/ns/>
"""

SUB_TEMPLATE = """
    OPTIONAL {
      SELECT ?uri (COUNT(?what) AS ?V_NAME_REL) WHERE {
        ?uri V_URI_REL ?what .
        ?what V_TARGET_REL <V_TARGET_URI> .
	  } GROUP BY ?uri
	}
"""

HAL_TEMPLATE = """
SELECT ?uri WHERE {
    ?uri V_URI_REL/V_TARGET_REL <V_TARGET_URI> .
} LIMIT 1
"""


class RelatedListBuilder:
    """Builds the SPARQL and JSON forms of every related-list relation."""

    def __init__(self, lux_config, inverses, related_list_scopes):
        self.lux_config = lux_config
        self.terms = lux_config.lux_config["terms"]
        self.inverses = inverses
        self.related_list_scopes = related_list_scopes
        #: (scope, list, relation, what was missing) for each relation dropped.
        self.skipped = []

    def build_all(self):
        """Compile every scope/related-list. Returns (sparql, json, hal_probes)."""
        sparql = {}
        as_json = {}
        probes = {}
        for scope, entry in self.related_list_scopes.items():
            sparql[scope] = {}
            as_json[scope] = {}
            probes[scope] = {}
            for qtype, queries in entry.items():
                spql, hal_probes, skipped_keys = self.query_stub(scope, qtype)
                sparql[scope][qtype] = spql
                probes[scope][qtype] = hal_probes
                as_json[scope][qtype] = {}
                for qname, qscope in queries.items():
                    if qname in skipped_keys:
                        continue
                    try:
                        stub = self.json_stub(qname, scope, qscope)
                    except Exception as e:
                        # the JSON side needs config/terms_inverse.json to agree
                        # with the predicate vocabulary; if it does not, drop the
                        # relation rather than fail to start
                        self.skipped.append((scope, qtype, qname, f"json: {e}"))
                        continue
                    as_json[scope][qtype][qname] = json.dumps(
                        stub, separators=(",", ":")
                    )
        if self.skipped:
            print(
                f"Related lists: skipped {len(self.skipped)} relation(s) with no "
                "QLever predicate. Regenerate config/related_list_scopes.json with "
                "files/derive_from_upstream.py to drop them at source:"
            )
            for scope, qtype, key, why in self.skipped[:10]:
                print(f"  {scope}/{qtype} {key} (no predicate for {why})")
            if len(self.skipped) > 10:
                print(f"  ... and {len(self.skipped) - 10} more")
        return sparql, as_json, probes

    def json_stub(self, qname, scope, qscope):
        """The user-facing JSON query for one related-list relation.

        Given e.g. ``created-createdBy``, produce
        ``AND: [createdBy: {id: FROM}, createdBy: {id: TO}]``.
        """
        # given created, createdBy
        # produce AND: [createdBy: id: X, createdBy: id: Y]

        from_uri = "V_FROM_URI"
        to_uri = "V_TO_URI"
        fields = qname.split("-")

        q = {"AND": []}
        if len(fields) == 2:
            inv = self.inverses[scope][fields[0]]
            q["AND"].append({inv: {"id": from_uri}})
            q["AND"].append({fields[1]: {"id": to_uri}})
        else:
            # Find the first point at which query scope is the same as target scope
            aq = {}
            topa = aq
            target_scope = scope
            while target_scope != qscope:
                f = fields.pop(0)
                inv = self.inverses[target_scope][f]
                try:
                    target_scope = self.terms[target_scope][f][
                        "relation"
                    ]
                except KeyError:
                    print(f"Failed to traverse for {qname}")
                    print(f"Invalid field '{f}' for scope '{target_scope}'")
                    break
                aq[inv] = {}
                aq = aq[inv]
            aq["id"] = from_uri
            q["AND"].append(topa)
            bq = {}
            topb = bq
            for f in fields:
                bq[f] = {}
                bq = bq[f]
            bq["id"] = to_uri
            q["AND"].append(topb)
        return q

    def query_stub(self, scope, qtype):
        """Return (count query, ordered HAL existence probes) for one related list.

        The probes are cheap ``LIMIT 1`` versions of each relation, ordered by
        the same heuristic as the fragments so a HAL check can stop at the
        first hit instead of running the whole count query. Relations that
        cannot be expressed are reported in the third element so the JSON side
        drops exactly the same ones.
        """
        names = []
        fragments = []
        hal_tests = []
        skipped_keys = set()
        flds = SCOPE_FIELDS




        for key, rscope in self.related_list_scopes[scope][qtype].items():
            fields = key.split("-")
            # print(f"    {key} -> {rscope}")
            missing = None
            if len(fields) == 2:
                # .get, not [] - a config generated without predicate
                # validation can name a term QLever has no predicate for, and
                # that must skip the relation, not stop the server starting.
                aq = [flds.get(scope, {}).get(fields[0])]
                bq = [flds.get(rscope, {}).get(fields[1])]
                if not aq[0]:
                    missing = f"{scope}.{fields[0]}"
                elif not bq[0]:
                    missing = f"{rscope}.{fields[1]}"
            else:
                # Construct p1 and p2 as property paths using the same logic as the JSON search builder
                aq = []
                target_scope = scope
                while target_scope not in ["item", "work", "set", "collection"]:
                    f = fields.pop(0)
                    try:
                        p = flds[target_scope][f]
                    except KeyError:
                        missing = f"{target_scope}.{f}"
                        break
                    aq.append(p)
                    try:
                        target_scope = self.terms[
                            target_scope
                        ][f]["relation"]
                    except KeyError:
                        missing = f"{target_scope}.{f} (no target scope)"
                        break

                bq = []
                for f in fields:
                    try:
                        p2 = flds[target_scope][f]
                    except KeyError:
                        missing = f"{target_scope}.{f}"
                        break
                    bq.append(p2)
                    target_scope = self.terms[target_scope][f][
                        "relation"
                    ]

            # A hop with no QLever predicate - absent, or mapped to "" - means
            # the property path cannot be built. Drop the relation rather than
            # emit a broken one or, worse, a half-built path from a loop that
            # gave up part way through.
            if missing or not aq or not bq or not all(aq) or not all(bq):
                self.skipped.append((scope, qtype, key, missing or "empty predicate"))
                skipped_keys.add(key)
                continue

            p = "/".join([f"^lux:{x[1:]}" if x[0] == "^" else f"lux:{x}" for x in aq])
            p2 = "/".join([f"^lux:{x[1:]}" if x[0] == "^" else f"lux:{x}" for x in bq])

            score = 1
            if "Classification" in p:
                score += 3
            if "Classification" in p2:
                score += 3
            if "workAbout" in p:
                score += 1
            if "workAbout" in p2:
                score += 1
            if "agentOf" in p:
                score += 1
            if "agentOf" in p2:
                score += 1
            if "Beginning" in p:
                score += 0.5
            if "Beginning" in p2:
                score += 0.5
            if "/" in p:
                score -= 1
            if "/" in p2:
                score -= 1
            if "Set" in p or "Event" in p:
                score -= 1
            if "Set" in p2 or "Event" in p2:
                score -= 1

            kn = key.replace("-", "_")
            names.append([kn, score])
            tmpl = (
                SUB_TEMPLATE.replace("V_NAME_REL", kn)
                .replace("V_URI_REL", p)
                .replace("V_TARGET_REL", p2)
            )
            tmp2 = HAL_TEMPLATE.replace("V_URI_REL", p).replace("V_TARGET_REL", p2)
            fragments.append([tmpl, score])

            hal_tests.append([tmp2, score])

        names.sort(key=lambda x: x[1], reverse=True)
        names = [x[0] for x in names]
        coalesces = " + ".join([f"COALESCE(?{x}, 0)" for x in names])
        vars = " ".join([f"?{x}" for x in names])

        # construct ordered series of tests for this related list for HAL testing
        # this allows the MT to step through each in turn and can bail when any
        # of them match. It also doesn't tie up the CPU as much on a single query
        hal_tests.sort(key=lambda x: x[1], reverse=True)
        hal_probes = [f"{PREFIXES}\n{x[0]}" for x in hal_tests]

        fragments.sort(key=lambda x: x[1], reverse=True)
        fragments = [x[0] for x in fragments]

        newline = "\n"  # work around no backslash in f string
        q = f"""
{PREFIXES}
SELECT ?uri ?total {vars} WHERE {{
    {newline.join(fragments)}
    FILTER(!(?uri = <V_TARGET_URI>))
    BIND({coalesces} AS ?total)
}} ORDER BY DESC(?total) LIMIT 20"""
        return q, hal_probes, skipped_keys
