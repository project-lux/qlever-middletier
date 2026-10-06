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
)
from qleverlux.sparql import (
    GraphPattern as Pattern,
)


class SparqlTranslator:
    def __init__(self, config, stopwords=None):
        """``config`` is a luxql ``LuxConfig``; ``stopwords`` a list or dict, or
        None to leave text queries unfiltered."""
        self.config = config
        self.counter = 0
        self.scored = []
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
        if sort and sort != "relevance":
            sparql.add_variables(["?uri", "(MIN(?sortWithDefault) AS ?sort)"])
        else:
            sparql.add_variables(["?uri", "(SUM(?score) AS ?sscore)"])
            self.calculate_scores = True

        where = Pattern()

        if self.portal is not None:
            t = Triple("?uri", "lux:source", f"lux:{self.portal}")
            where.add_triples([t])

        if scope is not None and scope != "any":
            t = Triple("?uri", "a", f"lux:{scope.title()}")
            where.add_triples([t])

        query.var = "?uri"
        self.translate_query(query, where)

        gby = GroupBy(["?uri"])
        sparql.add_group_by(gby)

        if sort == "relevance":
            bs = []
            for x in self.scored:
                bs.append(f"COALESCE(?score_{x}, 0)")
            if bs:
                where.add_binding(Binding(" + ".join(bs), "?score"))
                ob = OrderBy(["?sscore"], True)
                sparql.add_order_by(ob)
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

    def _multi_where(self, branches):
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
            if scope is not None and scope != "any":
                clause.add_triples([Triple("?uri", "a", f"lux:{scope.title()}")])
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
        self.calculate_scores = True

        if limit is None:
            sparql = SelectQuery(offset=offset)
        else:
            sparql = SelectQuery(limit=limit, offset=offset)
        for pfx, uri in self.prefixes.items():
            sparql.add_prefix(Prefix(pfx, uri))
        if sort and sort != "relevance":
            sparql.add_variables(["?uri", "(MIN(?sortWithDefault) AS ?sort)"])
        else:
            sparql.add_variables(["?uri", "(SUM(?score) AS ?sscore)"])

        where = self._multi_where(branches)
        sparql.add_group_by(GroupBy(["?uri"]))

        if sort == "relevance" or not sort:
            bs = [f"COALESCE(?score_{x}, 0)" for x in self.scored]
            if bs:
                where.add_binding(Binding(" + ".join(bs), "?score"))
                sparql.add_order_by(OrderBy(["?sscore"], True))
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

    def translate_multi_search_count(self, branches):
        """``COUNT(*)`` over a multi-scope search, for HAL estimates."""
        self.counter = 0
        self.scored = []
        self.calculate_scores = True

        inner = SelectQuery()
        inner.add_variables(["?uri"])
        inner.set_where_pattern(self._multi_where(branches))
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
        """Relationship term -> prefixed predicate. See query/predicates.py."""
        return predicates.get_predicate(rel, scope)

    def translate_relationship(self, query, parent):
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
                parent.add_triples([Triple(query.var, pred, f"<{lf.value}>")])
        else:
            parent.add_triples([Triple(query.var, pred, query.children[0].var)])
            self.translate_query(query.children[0], parent)

            if self.portal is not None:
                t = Triple(query.children[0].var, "lux:source", f"lux:{self.portal}")
                parent.add_triples([t])

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
                    v = Values([f"<{query.value}>"], query.var)
                    parent.add_value(v)
            elif query.field == "identifier":
                # do exact match on the string
                pred = f"lux:{scope}Identifier"
                # UNION with equivalent if starts with http
                if query.value.startswith("http://") or query.value.startswith(
                    "https://"
                ):
                    p1 = Pattern()
                    p1.add_triples(
                        [Triple(query.var, pred, f'"{query.value.lower()}"')]
                    )
                    p2 = Pattern(union=True)
                    p2.add_triples(
                        [Triple(query.var, "la:equivalent", f"<{query.value}>")]
                    )
                    parent.add_nested_graph_pattern(p1)
                    parent.add_nested_graph_pattern(p2)
                else:
                    parent.add_triples(
                        [Triple(query.var, pred, f'"{query.value.lower()}"')]
                    )
            elif query.field == "recordType":
                parent.add_triples([Triple(query.var, "a", f"la:{query.value}")])
            elif query.field == self.name_field and query.complete:
                pred = f"lux:{scope}Name"
                val = query.value.lower()
                val = val.replace('"', "")
                parent.add_triples([Triple(query.var, pred, f'"{val}"')])
            else:
                self.text.build(query, parent, scope, self.counter)
                # this clause emitted ?score_<counter>; record it so the
                # relevance BIND below can sum it
                self.scored.append(self.counter)

        elif typ == "date":
            # do date query per qlever
            dt = query.value

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
