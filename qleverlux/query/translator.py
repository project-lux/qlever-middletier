import re

from luxql import LuxBoolean, LuxLeaf, LuxRelationship

from qleverlux.query import predicates
from qleverlux.query.predicates import SCOPE_FIELDS, SCOPE_LEAF_FIELDS
from qleverlux.query.text import (
    ANYWHERE_FIELD,
    ID_FIELD,
    NAME_FIELD,
    TextSearchBuilder,
)
from qleverlux.sparql import (
    Binding,
    Filter,
    GroupBy,
    OrderBy,
    Prefix,
    SelectQuery,
    Triple,
    Values,
    iri,
    string_literal,
)
from qleverlux.sparql import (
    GraphPattern as Pattern,
)

#: A whole date value: luxql's ``valid_date_re`` (with the fractional seconds and ``Z`` its own error
#: message describes), matched to the end. luxql only anchors the start, so ``1900" . <any SPARQL>``
#: passed it and reached the query text.
DATE_VALUE = re.compile(
    r"(-[0-9]{2})?[0-9]{4}(-[0-1][0-9]-[0-3][0-9](T[0-2][0-9]:[0-5][0-9]:[0-5][0-9](\.[0-9]+)?Z?)?)?"
)
#: A record type is a Linked Art class name, written into the query after ``la:``.
RECORD_TYPE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

#: The search terms answered by vector similarity rather than text search:
#: `similar` takes free text, `similarRecords` the records to start from.
SIMILAR_FIELD = "similar"
SIMILAR_RECORDS_FIELD = "similarRecords"
#: Seed records the vector sidecar accepts in one call.
MAX_SEED_RECORDS = 100


def seed_ids(node):
    """The URIs of ``{"id": ...}`` or an ``OR`` of only those, else None (a query to run)."""
    if isinstance(node, LuxLeaf) and node.field == "id":
        return [node.value]
    if (
        isinstance(node, LuxBoolean)
        and node.field == "OR"
        and node.children
        and all(isinstance(c, LuxLeaf) and c.field == "id" for c in node.children)
    ):
        return [c.value for c in node.children]
    return None

#: Sort keys computed from the query itself rather than looked up in
#: sorts.json: "relevance" (text and vector combined) and "similarity"
#: (cosine alone, text only breaking ties).
RANKED_SORTS = ("relevance", "similarity")


class SparqlTranslator:
    def __init__(self, config, stopwords=None):
        """``config`` is a luxql ``LuxConfig``; ``stopwords`` a list or dict, or
        None to leave text queries unfiltered."""
        self.config = config
        self.counter = 0
        self.scored = []
        self.vector_scored = []
        self.portal = None
        self.prefixes = {
            #            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            #            "geo": "http://www.opengis.net/ont/geosparql#",
            #            "geof": "http://www.opengis.net/def/function/geosparql/",
            #            "qlss": "https://qlever.cs.uni-freiburg.de/spatialSearch/",
            "la": "https://linked.art/ns/terms/",
            "lux": "https://lux.collections.yale.edu/ns/",
            "view": "https://qlever.cs.uni-freiburg.de/materializedView/",
            "vec": "http://lux.collections.yale.edu/ns/vec#",
        }

        self.text = TextSearchBuilder(stopwords=stopwords)
        self.anywhere_field = ANYWHERE_FIELD
        self.id_field = ID_FIELD
        self.name_field = NAME_FIELD

        # The predicate vocabulary itself lives in query/predicates.py.
        # Kept as attributes because the facet resolver and the related-list
        # query builder both read them off a translator instance.
        self.scope_leaf_fields = SCOPE_LEAF_FIELDS
        self.scope_fields = SCOPE_FIELDS

        self.text_scale = 1.0
        self.vector_scale = 1.0
        self.vector_service = ""
        self.vector_top_k = 100
        self.vector_nested_top_k = 5
        # (min_score, max_gap) at the top and nested; 0 leaves a cutoff off.
        self.vector_cutoffs = (0.0, 0.0)
        self.vector_nested_cutoffs = (0.0, 0.0)
        self.vector_constrain_in_and = True
        self.max_hops = 8

    def configure_scoring(self, text_scale=1.0, vector_scale=1.0):
        """Weights of the two relevance sources, each already 0..1. See _relevance_query."""
        self.text_scale = text_scale
        self.vector_scale = vector_scale

    def configure_paths(self, max_hops=8):
        """Longest chain a transitive term (``broader+`` and friends) follows; 0 is unbounded."""
        self.max_hops = max_hops

    def configure_vectors(self, service="", top_k=100, constrain_in_and=True, nested_top_k=5,
                          min_score=0.0, max_gap=0.0, nested_min_score=0.0, nested_max_gap=0.0):
        """Where the vector SERVICE is; an empty ``service`` disables ``similar``.

        ``top_k`` is for a similarity clause at the top of the query, where the
        results are the similar records and rank by cosine. ``nested_top_k`` is
        for one under a relationship, where each similar record pulls in its
        related records unranked, so the weak matches near the bottom of a long
        list would swamp the strong ones.

        ``min_score`` / ``max_gap`` (and their nested forms) become the sidecar's
        ``vec:minScore`` / ``vec:maxGap``, cutting that top-K list by cosine: an
        absolute floor, and a distance below the best hit. 0 leaves one off.

        ``service`` is a bare URL. It is bracketed here, and the builder passes
        a bracketed name through untouched because it contains a ``:``.
        """
        self.vector_service = iri(service) if service else ""
        self.vector_top_k = int(top_k)
        self.vector_nested_top_k = int(nested_top_k)
        self.vector_cutoffs = (float(min_score), float(max_gap))
        self.vector_nested_cutoffs = (float(nested_min_score), float(nested_max_gap))
        self.vector_constrain_in_and = constrain_in_and

    @property
    def stopwords(self):
        return self.text.stopwords

    def set_stopwords(self, stopwords):
        self.text.set_stopwords(stopwords)

    def translate_search(
        self,
        query,
        scope=None,
        limit=None,
        offset=0,
        sort="",
        order="",
        sortDefault="ZZZZZZZZZZ",
    ):
        # Implement translation logic here
        self.counter = 0
        self.scored = []
        self.vector_scored = []
        self.calculate_scores = False
        self.calculate_scores = (
            True  # always calculate scores for now until cache key is fixed
        )
        if limit is None:
            sparql = SelectQuery(offset=offset)
        else:
            sparql = SelectQuery(limit=limit, offset=offset)

        for pfx, uri in self.prefixes.items():
            sparql.add_prefix(Prefix(pfx, uri))
        if sort and sort not in RANKED_SORTS:
            sparql.add_variables(["?uri", "?type", "(MIN(?sortWithDefault) AS ?sort)"])
        else:
            sparql.add_variables(["?uri", "?type", "(SUM(?score) AS ?sscore)"])
            self.calculate_scores = True

        where = Pattern()

        if self.portal is not None:
            t = Triple("?uri", "lux:source", f"lux:{self.portal}")
            where.add_triples([t])

        self.add_scope_type(where, scope)

        query.var = "?uri"
        self.translate_query(query, where)

        gby = GroupBy(["?uri", "?type"])
        sparql.add_group_by(gby)

        if sort in RANKED_SORTS:
            ranked = self._relevance_query(where, limit, offset, by=sort, order=order)
            if ranked is not None:
                return ranked
        elif sort:
            spatt = Pattern(optional=True)
            spatt.add_triples([Triple("?uri", sort, "?sortValue")])
            if "SortName" in sort:
                spatt.add_filter(Filter("!isNumeric(?sortValue)"))
            where.add_nested_graph_pattern(spatt)
            where.add_binding(
                Binding(f'COALESCE(?sortValue, "{sortDefault}")', "?sortWithDefault")
            )
            ob = OrderBy(["?sort"], order == "DESC")
            sparql.add_order_by(ob)

        sparql.set_where_pattern(where)
        return sparql

    @staticmethod
    def _scaled(expr, scale):
        return expr if scale == 1 else f"{expr} * {scale!r}"

    def _relevance_query(self, where, limit, offset, by="relevance", order="DESC"):
        """The query ranked by relevance or similarity, or None when no clause is scored.

        ``by="similarity"`` ranks on the cosine alone, honouring ``order``;
        text clauses still filter, and their raw score breaks ties - which
        matters for ``similar OR text``, where text-only matches have no cosine.
        It needs no per-query max, so it calls the vector SERVICE once. With no
        ``similar`` clause in the query it falls back to relevance. Relevance
        is always descending, as it always has been.

        Two sources, each brought to 0..1 per result before they are added:

        * text: the result's summed ``?score_N`` divided by the largest such
          sum in this query, so the best text hit is 1. Raw view scores vary
          by two orders of magnitude between queries (a name search tops out
          around 5-30, an anywhere search around 500), so no fixed constant
          could do this. Dividing by one number cannot reorder anything, so a
          text-only ranking is exactly what it was before.
        * vector: the cosine from the SERVICE (MAX rather than SUM, so a
          record that joins several rows is not counted twice), min-max
          scaled over this query's vector matches so the best is 1 and the
          weakest 0. Raw cosines sit in a narrow, model-dependent band (about
          0.7-0.9 for EmbeddingGemma 2, 0.5-0.75 for Harrier), so adding them
          raw would let the model decide how much vector matches weigh against
          text. Text-only matches have no cosine (0) and are kept out of the
          minimum; a lone vector match counts as 1. Like the text max, this is
          only needed when both kinds are present: on its own, scaling cannot
          reorder anything, so a vector-only query keeps the raw cosine and its
          single SERVICE call.

        The max needs the grouped scores of the whole result set, so the
        grouped SELECT appears twice: once to rank and once under MAX. It is
        the same object, so the text is identical and QLever evaluates it once.
        """
        if not (self.scored or self.vector_scored):
            return None
        similarity = by == "similarity" and bool(self.vector_scored)
        aggregates = []
        if self.scored:
            total = " + ".join(f"COALESCE(?score_{x}, 0)" for x in self.scored)
            where.add_binding(Binding(total, "?tscore"))
            aggregates.append("(SUM(?tscore) AS ?ts)")
        if self.vector_scored:
            total = " + ".join(f"COALESCE(?score_{x}, 0)" for x in self.vector_scored)
            where.add_binding(Binding(total, "?vscore"))
            aggregates.append("(MAX(?vscore) AS ?vs)")

        grouped = SelectQuery()
        grouped.add_variables(["?uri", "?type", *aggregates])
        grouped.set_where_pattern(where)
        grouped.add_group_by(GroupBy(["?uri", "?type"]))

        outer_where = Pattern()
        outer_where.add_nested_select_query(grouped)
        if similarity:
            outer_where.add_binding(Binding("COALESCE(?vs, 0)", "?sscore"))
            ranked = SelectQuery(limit=limit or 0, offset=offset)
            for pfx, uri in self.prefixes.items():
                ranked.add_prefix(Prefix(pfx, uri))
            ranked.add_variables(["?uri", "?type", "?sscore"])
            ranked.set_where_pattern(outer_where)
            ranked.add_order_by(OrderBy(["?sscore"], str(order).upper() != "ASC"))
            if self.scored:
                ranked.add_order_by(OrderBy(["?ts"], True))
            return ranked

        terms = []
        if self.scored:
            best = SelectQuery()
            best.add_variables(["(MAX(?ts) AS ?tmax)"])
            if self.vector_scored:
                # cosines are <= 1, so 2 keeps text-only matches (?vs = 0) out of the minimum
                best.add_variables(["(MAX(?vs) AS ?vmax)", "(MIN(IF(?vs > 0, ?vs, 2)) AS ?vmin)"])
            best_where = Pattern()
            best_where.add_nested_select_query(grouped)
            best.set_where_pattern(best_where)
            outer_where.add_nested_select_query(best)
            # a zero or missing max divides to an error; COALESCE makes it 0
            terms.append(self._scaled("COALESCE(?ts / ?tmax, 0)", self.text_scale))
            if self.vector_scored:
                normalised = "COALESCE(IF(?vs > 0, IF(?vmax > ?vmin, (?vs - ?vmin) / (?vmax - ?vmin), 1), 0), 0)"
                terms.append(self._scaled(normalised, self.vector_scale))
        elif self.vector_scored:
            terms.append(self._scaled("COALESCE(?vs, 0)", self.vector_scale))
        outer_where.add_binding(Binding(" + ".join(terms), "?sscore"))

        ranked = SelectQuery(limit=limit or 0, offset=offset)
        for pfx, uri in self.prefixes.items():
            ranked.add_prefix(Prefix(pfx, uri))
        ranked.add_variables(["?uri", "?type", "?sscore"])
        ranked.set_where_pattern(outer_where)
        ranked.add_order_by(OrderBy(["?sscore"], True))
        return ranked

    def add_scope_type(self, pattern, scope, with_type=True):
        """Restrict ``?uri`` to ``scope`` and bind its Linked Art class to ``?type``.

        The class is what a search result reports as its ``type``. With no
        scope, or ``any``, ``?type`` ranges over every class of every scope.
        A count has no use for the class, so ``with_type=False`` leaves it out.
        """
        if scope is not None and scope != "any":
            pattern.add_triples([Triple("?uri", "a", f"lux:{scope.title()}")])
            classes = predicates.SCOPE_TYPES[scope]
        else:
            classes = [c for cs in predicates.SCOPE_TYPES.values() for c in cs]
        if not with_type:
            return
        pattern.add_triples([Triple("?uri", "a", "?type")])
        pattern.add_filter(Filter(" || ".join(f"?type = la:{c}" for c in classes)))

    def _multi_where(self, branches, with_type=True):
        """The UNION of one pattern per branch, each rooted in its own scope.

        ``branches`` is [(scope, parsed_query), ...]. Each alternative carries
        its own ``?uri a lux:<Scope>`` (and portal filter) rather than sharing a
        single one, which is what makes a query across unlike scopes possible:
        concepts merged with events as readily as items with sets.

        The clause counter deliberately keeps running across branches, so the
        per-clause variables of one alternative cannot collide with another's.
        """
        where = Pattern()
        for index, (scope, query) in enumerate(branches):
            clause = Pattern() if index == 0 else Pattern(union=True)
            if self.portal is not None:
                clause.add_triples([Triple("?uri", "lux:source", f"lux:{self.portal}")])
            self.add_scope_type(clause, scope, with_type)
            query.var = "?uri"
            self.translate_query(query, clause)
            where.add_nested_graph_pattern(clause)
        return where

    def translate_multi_search(
        self,
        branches,
        limit=None,
        offset=0,
        sort="",
        order="",
        sortDefault="ZZZZZZZZZZ",
    ):
        """A search across several scopes at once, returning one merged list.

        Sorting is optional and often impossible here: a sort predicate has to
        exist in every scope involved, so a multi search with no usable sort
        falls back to relevance, exactly as a single-scope one does.
        """
        self.counter = 0
        self.scored = []
        self.vector_scored = []
        self.calculate_scores = True

        if limit is None:
            sparql = SelectQuery(offset=offset)
        else:
            sparql = SelectQuery(limit=limit, offset=offset)
        for pfx, uri in self.prefixes.items():
            sparql.add_prefix(Prefix(pfx, uri))
        if sort and sort not in RANKED_SORTS:
            sparql.add_variables(["?uri", "?type", "(MIN(?sortWithDefault) AS ?sort)"])
        else:
            sparql.add_variables(["?uri", "?type", "(SUM(?score) AS ?sscore)"])

        where = self._multi_where(branches)
        sparql.add_group_by(GroupBy(["?uri", "?type"]))

        if not sort or sort in RANKED_SORTS:
            ranked = self._relevance_query(where, limit, offset, by=sort or "relevance", order=order)
            if ranked is not None:
                return ranked
        else:
            # ``sort`` may be one predicate every branch shares, or one per
            # branch when the same sort key is spelled differently per scope
            # (anySortName -> lux:<scope>SortName). The latter becomes a UNION
            # so each branch contributes its own sort value to one ?sortValue.
            preds = sort if isinstance(sort, list) else [sort]
            spatt = Pattern(optional=True)
            if len(preds) == 1:
                spatt.add_triples([Triple("?uri", preds[0], "?sortValue")])
                if "SortName" in preds[0]:
                    spatt.add_filter(Filter("!isNumeric(?sortValue)"))
            else:
                for index, pred in enumerate(preds):
                    alt = Pattern() if index == 0 else Pattern(union=True)
                    alt.add_triples([Triple("?uri", pred, "?sortValue")])
                    if "SortName" in pred:
                        alt.add_filter(Filter("!isNumeric(?sortValue)"))
                    spatt.add_nested_graph_pattern(alt)
            where.add_nested_graph_pattern(spatt)
            where.add_binding(
                Binding(f'COALESCE(?sortValue, "{sortDefault}")', "?sortWithDefault")
            )
            sparql.add_order_by(OrderBy(["?sort"], order == "DESC"))

        sparql.set_where_pattern(where)
        return sparql

    def _count_of(self, inner):
        """``SELECT (COUNT(*) AS ?count) { inner }``, with the prefixes."""
        top = SelectQuery()
        for pfx, uri in self.prefixes.items():
            top.add_prefix(Prefix(pfx, uri))
        top.add_variables(["(COUNT(*) AS ?count)"])
        topwhere = Pattern()
        topwhere.add_nested_select_query(inner)
        top.set_where_pattern(topwhere)
        return top

    def translate_search_total(self, query, scope=None):
        """How many rows ``translate_search`` lists: the same pattern and the same
        (``?uri``, ``?type``) grouping, counted, with no sort or scoring to compute.

        A search page asks for one window of rows with ``LIMIT``, which leaves
        QLever counting only that window, so its total comes from here. Unlike
        ``translate_search_count`` (HAL), it keeps the class filter and the
        per-class grouping, so it agrees with the list even for a record with
        two Linked Art classes in scope, or none.
        """
        self.counter = 0
        self.scored = []
        self.vector_scored = []
        self.calculate_scores = True

        inner = SelectQuery()
        inner.add_variables(["?uri", "?type"])
        where = Pattern()
        if self.portal is not None:
            where.add_triples([Triple("?uri", "lux:source", f"lux:{self.portal}")])
        self.add_scope_type(where, scope)
        query.var = "?uri"
        self.translate_query(query, where)
        inner.add_group_by(GroupBy(["?uri", "?type"]))
        inner.set_where_pattern(where)
        return self._count_of(inner)

    def translate_multi_search_total(self, branches):
        """``translate_search_total`` for a multi-scope search."""
        self.counter = 0
        self.scored = []
        self.vector_scored = []
        self.calculate_scores = True

        inner = SelectQuery()
        inner.add_variables(["?uri", "?type"])
        inner.set_where_pattern(self._multi_where(branches))
        inner.add_group_by(GroupBy(["?uri", "?type"]))
        return self._count_of(inner)

    def translate_multi_search_count(self, branches):
        """``COUNT(*)`` over a multi-scope search, for HAL estimates."""
        self.counter = 0
        self.scored = []
        self.vector_scored = []
        self.calculate_scores = True

        inner = SelectQuery()
        inner.add_variables(["?uri"])
        inner.set_where_pattern(self._multi_where(branches, with_type=False))
        inner.add_group_by(GroupBy(["?uri"]))

        top = SelectQuery()
        for pfx, uri in self.prefixes.items():
            top.add_prefix(Prefix(pfx, uri))
        top.add_variables(["(COUNT(*) AS ?count)"])
        topwhere = Pattern()
        topwhere.add_nested_select_query(inner)
        top.set_where_pattern(topwhere)
        return top

    def translate_search_count(self, query, scope=None):
        # Implement translation logic here
        self.counter = 0
        self.calculate_scores = True

        sparql = SelectQuery()
        sparql.add_variables(["?uri"])
        where = Pattern()
        if scope is not None and scope != "any":
            t = Triple("?uri", "a", f"lux:{scope.title()}")
            where.add_triples([t])
        if self.portal is not None:
            t = Triple("?uri", "lux:source", f"lux:{self.portal}")
            where.add_triples([t])

        query.var = "?uri"
        self.translate_query(query, where)
        sparql.add_group_by(GroupBy(["?uri"]))
        sparql.set_where_pattern(where)

        # Now wrap in a COUNT(*)

        top = SelectQuery()
        for pfx, uri in self.prefixes.items():
            top.add_prefix(Prefix(pfx, uri))
        top.add_variables(["(COUNT(*) AS ?count)"])
        topwhere = Pattern()
        topwhere.add_nested_select_query(sparql)
        top.set_where_pattern(topwhere)

        return top

    def translate_search_related(self, query, scope=None):
        self.counter = 0
        self.scored = []
        self.vector_scored = []
        self.calculate_scores = True
        sparql = SelectQuery(limit=100)
        for pfx, uri in self.prefixes.items():
            sparql.add_prefix(Prefix(pfx, uri))
        sparql.add_variables(["?uri", "(COUNT(?uri) AS ?count)"])

        where = Pattern()
        query.var = "?uri"

        if self.portal is not None:
            t = Triple("?uri", "lux:source", f"lux:{self.portal}")
            where.add_triples([t])

        self.translate_query(query, where)
        where.add_filter(Filter("?uri != <URI-HERE>"))

        sparql.set_where_pattern(where)
        gby = GroupBy(["?uri"])
        sparql.add_group_by(gby)
        ob = OrderBy(["?count"], True)
        sparql.add_order_by(ob)
        return sparql

    def translate_facet(
        self, query, facet, scope=None, limit=None, offset=0, sort="", order=""
    ):
        self.calculate_scores = True
        self.counter = 0
        gb = GroupBy(["?facet"])
        if not order:
            ob = OrderBy(["?facetCount"], True)
        else:
            ob = OrderBy(["?facet"], order == "DESC")

        if limit is None:
            sparql = SelectQuery(offset=offset)
        else:
            sparql = SelectQuery(limit=limit, offset=offset)

        for pfx, uri in self.prefixes.items():
            sparql.add_prefix(prefix=Prefix(pfx, uri))
        sparql.add_variables(["?facet", "(COUNT(?facet) AS ?facetCount)"])

        inner = SelectQuery(distinct=True)
        inner.add_variables(["?uri"])
        where = Pattern()
        if scope is not None and scope != "any":
            t = Triple("?uri", "a", f"lux:{scope.title()}")
            where.add_triples([t])
        if self.portal is not None:
            t = Triple("?uri", "lux:source", f"lux:{self.portal}")
            where.add_triples([t])

        query.var = "?uri"
        self.translate_query(query, where)
        inner.set_where_pattern(where)

        outer = Pattern()
        outer.add_nested_select_query(inner)
        outer.add_triples([Triple("?uri", facet, "?facet")])

        sparql.add_group_by(gb)
        sparql.add_order_by(ob)
        sparql.set_where_pattern(outer)
        return sparql

    def translate_facet_count(self, query, facet):
        """
        PREFIX lux: <https://lux.collections.yale.edu/ns/>
        SELECT (COUNT(?facet) AS ?count) WHERE {
          {
            SELECT ?facet WHERE {
              {
                SELECT DISTINCT ?uri WHERE {
                  ?uri lux:placeOfItemBeginning <https://lux.collections.yale.edu/data/place/02cff2e2-4285-4f82-bc5a-8d3b33596c9c> .
                }
              }
              ?uri lux:itemClassification ?facet .
            }
            GROUP BY ?facet
          }
        }
        """
        self.counter = 0
        self.calculate_scores = True
        sparql = SelectQuery()
        for pfx, uri in self.prefixes.items():
            sparql.add_prefix(prefix=Prefix(pfx, uri))
        sparql.add_variables(["(COUNT(?facet) AS ?count)"])

        inner = SelectQuery()
        gb = GroupBy(["?facet"])
        inner.add_variables(["?facet"])

        inner2 = SelectQuery(distinct=True)
        inner2.add_variables(["?uri"])
        where = Pattern()

        if self.portal is not None:
            t = Triple("?uri", "lux:source", f"lux:{self.portal}")
            where.add_triples([t])

        query.var = "?uri"
        self.translate_query(query, where)
        inner2.set_where_pattern(where)

        outer = Pattern()
        outer.add_nested_select_query(inner2)
        outer.add_triples([Triple("?uri", facet, "?facet")])
        inner.add_group_by(gb)
        inner.set_where_pattern(outer)

        swhere = Pattern()
        swhere.add_nested_select_query(inner)
        sparql.set_where_pattern(swhere)
        return sparql

    def translate_query(self, query, where):
        # print(f"translate query: {query.to_json()}")
        if isinstance(query, LuxBoolean):
            if query.field == "AND":
                self.translate_and(query, where)
            elif query.field == "OR":
                self.translate_or(query, where)
            elif query.field == "NOT":
                self.translate_not(query, where)
        elif isinstance(query, LuxRelationship):
            self.translate_relationship(query, where)
        elif isinstance(query, LuxLeaf):
            self.translate_leaf(query, where)
        else:
            print(f"Got {type(query)}")

    def translate_or(self, query, parent):
        # UNION a,b,c...
        x = 0
        for child in query.children:
            child.var = query.var
            if x == 0:
                clause = Pattern()
            else:
                clause = Pattern(union=True)
            x += 1
            self.translate_query(child, clause)
            parent.add_nested_graph_pattern(clause)

    def translate_and(self, query, parent):
        # just add the patterns in
        for child in query.children:
            child.var = query.var
            self.translate_query(child, parent)

    def translate_not(self, query, parent):
        # FILTER NOT EXISTS { ...}
        clause = Pattern(not_exists=True)
        query.children[0].var = query.var
        self.translate_query(query.children[0], clause)
        parent.add_nested_graph_pattern(clause)

    def get_predicate(self, rel, scope):
        """Relationship term -> prefixed predicate, transitive ones bounded to
        ``max_hops``. See query/predicates.py."""
        return predicates.bound_path(predicates.get_predicate(rel, scope), self.max_hops)

    def translate_relationship(self, query, parent):
        if query.field == SIMILAR_RECORDS_FIELD:
            self.translate_similar_records(query, parent, query.parent.provides_scope)
            return
        query.children[0].var = f"?var{self.counter}"
        self.counter += 1
        pred = self.get_predicate(query.field, query.parent.provides_scope)
        # test if only leaf is id:<uri>
        lf = query.children[0]
        if type(lf) is LuxLeaf and lf.field == "id":
            if lf.value[0] == "?":
                # basic test for sparql injection by requiring only a-zA-Z0-9_
                if not lf.value[1:].replace("_", "").isalnum():
                    raise ValueError("Invalid variable name")
                parent.add_triples([Triple(query.var, pred, lf.value)])
            else:
                parent.add_triples([Triple(query.var, pred, iri(lf.value))])
        else:
            parent.add_triples([Triple(query.var, pred, query.children[0].var)])
            self.translate_query(query.children[0], parent)

            if self.portal is not None:
                t = Triple(query.children[0].var, "lux:source", f"lux:{self.portal}")
                parent.add_triples([t])

    def translate_similar(self, query, parent, scope):
        """A ``similar`` leaf: free text, encoded by the vector sidecar.

        Always text, even when it looks like a URI: records to start from are
        ``similarRecords``. The cosine becomes this clause's ``?score_N``, which
        _relevance_query adds to the ranking.
        """
        value = str(query.value).strip()
        if not value:
            raise ValueError("Empty similarity query")
        self._similarity_service(query, parent, scope, [string_literal(value)])

    def translate_similar_records(self, query, parent, scope):
        """A ``similarRecords`` relationship: the stored vectors of the records
        it names are the seeds, and a result is scored by its nearest seed.

        The records are ``{"id": <uri>}``, an ``OR`` of them, or any query in
        the scope. A query has to be run first to know its records, which the
        translator cannot do: the service runs it and leaves the URIs on the
        node as ``seed_records`` (services/seeds.py). No records means no
        results. The seeds themselves are left out (the sidecar's default).
        """
        seeds = self._seed_records(query)
        if seeds:
            self._similarity_service(query, parent, scope, [iri(seed) for seed in seeds])
        else:
            nothing = Pattern()
            nothing.add_filter(Filter("false"))
            parent.add_nested_graph_pattern(nothing)
        self.counter += 1  # translate_leaf advances it for leaves; this is not one

    @staticmethod
    def _seed_records(query):
        seeds = getattr(query, "seed_records", None)
        if seeds is None:
            seeds = seed_ids(query.children[0] if query.children else None)
        if seeds is None:
            raise ValueError(
                "similarRecords with a query rather than ids has to be resolved against QLever first"
            )
        seeds = list(dict.fromkeys(str(seed).strip() for seed in seeds))
        if not all(seed.startswith(("http://", "https://")) for seed in seeds):
            raise ValueError("similarRecords ids must be record URIs")
        if len(seeds) > MAX_SEED_RECORDS:
            raise ValueError(
                f"similarRecords takes at most {MAX_SEED_RECORDS} records, not {len(seeds)}"
            )
        return seeds

    def _similarity_service(self, query, parent, scope, seeds):
        """One SERVICE call to the vector sidecar for this clause.

        ``seeds`` are already-rendered terms: one text literal, or record IRIs.
        """
        if not self.vector_service:
            raise ValueError("Similarity search is not configured")
        n = self.counter
        if self._under_relationship(query):
            top_k, (min_score, max_gap) = self.vector_nested_top_k, self.vector_nested_cutoffs
        else:
            top_k, (min_score, max_gap) = self.vector_top_k, self.vector_cutoffs
        triples = [Triple(query.var, "vec:similarTo", seed) for seed in seeds]
        triples += [
            Triple(query.var, "vec:score", f"?score_{n}"),
            Triple(query.var, "vec:topK", str(top_k)),
            Triple(query.var, "vec:index", string_literal(scope)),
        ]
        if min_score > 0:
            triples.append(Triple(query.var, "vec:minScore", repr(min_score)))
        if max_gap > 0:
            triples.append(Triple(query.var, "vec:maxGap", repr(max_gap)))
        if self.vector_constrain_in_and and self._has_and_siblings(query):
            triples.append(Triple(query.var, "vec:mode", '"constrained"'))

        svc = Pattern(service=self.vector_service)
        svc.add_triples(triples)
        parent.add_nested_graph_pattern(svc)
        self.vector_scored.append(n)

    @staticmethod
    def _under_relationship(query):
        """True when a relationship stands between ``query`` and the root, as in
        works ``aboutConcept`` {similar}. Booleans do not count: an AND or OR of
        clauses at the top still returns the similar records themselves. For
        ``similarRecords`` ``query`` is the relationship, so the walk starts above it."""
        node = getattr(query, "parent", None)
        while isinstance(node, (LuxBoolean, LuxRelationship, LuxLeaf)):
            if isinstance(node, LuxRelationship):
                return True
            node = getattr(node, "parent", None)
        return False

    @staticmethod
    def _has_and_siblings(query):
        """True when ``query`` is one of several children of an AND."""
        parent = getattr(query, "parent", None)
        return (
            isinstance(parent, LuxBoolean)
            and parent.field == "AND"
            and len(parent.children) > 1
        )

    def get_leaf_predicate(self, field, scope):
        """Leaf term -> predicate, or a [start, end] pair for dates."""
        return predicates.get_leaf_predicate(field, scope)

    def translate_leaf(self, query, parent):
        typ = query.provides_scope  # text / date / number etc.
        scope = query.parent.provides_scope  # item/work/etc

        if typ == "text":
            if query.field == self.id_field:
                if query.value[0] == "?":
                    # a variable ... assume the user knows what they're doing...
                    pass
                else:
                    v = Values([iri(query.value)], query.var)
                    parent.add_value(v)
            elif query.field == "identifier":
                # do exact match on the string
                pred = f"lux:{scope}Identifier"
                literal = string_literal(query.value.lower())
                # UNION with equivalent if it is an http(s) IRI; a value that only looks like
                # one (a space, a quote) is matched as a string alone
                equivalent = None
                if query.value.startswith(("http://", "https://")):
                    try:
                        equivalent = iri(query.value)
                    except ValueError:
                        equivalent = None
                if equivalent:
                    p1 = Pattern()
                    p1.add_triples([Triple(query.var, pred, literal)])
                    p2 = Pattern(union=True)
                    p2.add_triples([Triple(query.var, "la:equivalent", equivalent)])
                    parent.add_nested_graph_pattern(p1)
                    parent.add_nested_graph_pattern(p2)
                else:
                    parent.add_triples([Triple(query.var, pred, literal)])
            elif query.field == SIMILAR_FIELD:
                self.translate_similar(query, parent, scope)
            elif query.field == "recordType":
                if not RECORD_TYPE.fullmatch(str(query.value)):
                    raise ValueError(f"Not a record type: {query.value!r}")
                parent.add_triples([Triple(query.var, "a", f"la:{query.value}")])
            elif query.field == self.name_field and query.complete:
                # an exact match on the whole name: escaped like any literal. Stripping quotes alone
                # left a trailing backslash or a newline to break the query.
                pred = f"lux:{scope}Name"
                parent.add_triples([Triple(query.var, pred, string_literal(query.value.lower()))])
            else:
                self.text.build(query, parent, scope, self.counter)
                # this clause emitted ?score_<counter>; record it so the
                # relevance BIND below can sum it
                self.scored.append(self.counter)

        elif typ == "date":
            # do date query per qlever
            dt = query.value
            if not DATE_VALUE.fullmatch(str(dt)):
                raise ValueError(
                    "Dates require a specific format: 'YYYY-MM-DDThh:mm:ss.000Z' or '-YYYYYY-MM-DDThh:mm:ss.000Z'"
                )

            # make sure date is in a valid format
            if ":" not in dt:
                dt += "T00:00:00Z"

            comp = query.comparitor
            if comp == "==":
                comp = "="
            field = query.field
            # botb, eote
            preds = self.get_leaf_predicate(field, scope)
            qvar = query.var
            bvar = f"?date1{self.counter}"
            evar = f"?date2{self.counter}"

            # This is insufficient -- it needs to turn the query into a range, and then compare
            #
            p = Pattern()
            trips = [Triple(qvar, preds[0], bvar), Triple(qvar, preds[1], evar)]
            p.add_triples(trips)
            p.add_filter(Filter(f'{bvar} {comp} "{dt}"^^xsd:dateTime'))
            parent.add_nested_graph_pattern(p)

        elif typ == "float":
            # do number query per qlever
            dt = query.value
            comp = query.comparitor
            field = query.field
            pred = self.get_leaf_predicate(field, scope)
            qvar = query.var
            fvar = f"?float{self.counter}"

            p = Pattern()
            trips = [Triple(qvar, pred, fvar)]
            p.add_triples(trips)
            p.add_filter(Filter(f'{fvar} {comp} "{dt}"^^xsd:float'))
            parent.add_nested_graph_pattern(p)

        elif typ == "boolean":
            dt = query.value
            field = query.field
            pred = self.get_leaf_predicate(field, scope)
            qvar = query.var

            p = Pattern()
            trips = [Triple(qvar, pred, f'"{dt}"^^xsd:decimal')]
            p.add_triples(trips)
            parent.add_nested_graph_pattern(p)

        else:
            # Unknown
            raise ValueError(f"Unknown provides_scope: {typ}")
        self.counter += 1
