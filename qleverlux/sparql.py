from . import LuxLeaf, LuxBoolean, LuxRelationship
from .SPARQLQueryBuilder import *
import shlex
import unicodedata
from string import whitespace, punctuation

Pattern = GraphPattern  # noqa


class SparqlTranslator:
    def __init__(self, config):
        self.config = config
        self.counter = 0
        self.scored = []
        self.prefixes = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "la": "https://linked.art/ns/terms/",
            "lux": "https://lux.collections.yale.edu/ns/",
            "textSearch": "https://qlever.cs.uni-freiburg.de/textSearch/",
        }

        self.remove_diacritics = False
        self.min_word_chars = 4
        # self.padding_char2 = "Þ"
        self.padding_char = b"\xc3\xbe".decode("utf-8")

        # assert self.padding_char == self.padding_char2

        self.anywhere_field = "text"
        self.id_field = "id"
        self.name_field = "name"
        self.name_weight = 14
        self.text_weight = 5
        self.refs_weight = 1

        self.scope_leaf_fields = {
            "agent": {},
            "concept": {},
            "event": {},
            "place": {},
            "set": {},
            "work": {},
            "item": {},
        }

        self.scope_fields = {
            "agent": {
                "startAt": "placeOfAgentBeginning",
                "endAt": "placeOfAgentEnding",
                "foundedBy": "agentOfAgentBeginning",
                "gender": "gender",
                "occupation": "occupation",
                "nationality": "nationality",
                "professionalActivity": "typeOfAgentActivity",
                "activeAt": "placeOfAgentActivity",
                "createdSet": "^agentOfSetBeginning",
                "produced": "^agentOfItemBeginning",
                "created": "^agentOfWorkBeginning",
                "carriedOut": "^eventCarriedOutBy",
                "curated": "^setCuratedBy",
                "encountered": "^agentOfItemEncounter",
                "founded": "^agentOfAgentBeginning",
                "memberOfInverse": "^agentMemberOfGroup",
                "influencedProduction": "^agentInfluenceOfItemBeginning",
                "influencedCreation": "^agentInfluenceOfWorkBeginning",
                "publishedSet": "^agentOfSetPublication",
                "published": "^agentOfWorkPublication",
                "subjectOfSet": "^setAboutAgent",
                "subjectOfWork": "^workAboutAgent",
            },
            "item": {
                "producedAt": "placeOfItemBeginning",
                "producedBy": "agentOfItemBeginning",
                "producedUsing": "typeOfItemBeginning",
                "productionInfluencedBy": "agentInfluenceOfItemBeginning",
                "encounteredAt": "placeOfItemEncounter",
                "encounteredBy": "agentOfItemEncounter",
                "carries": "carries",
                "material": "material",
                "subjectOfSet": "^setAboutItem",
                "subjectOfWork": "^workAboutItem",
            },
            "concept": {
                "broader": "broader",
                "broaderPlus": "broader+",
                "classificationOfSet": "^setClassification",
                "classificationOfConcept": "^conceptClassification",
                "classificationOfEvent": "^eventClassification",
                "classificationOfItem": "^itemClassification",
                "classificationOfAgent": "^agentClassification",
                "classificationOfPlace": "^placeClassification",
                "classificationOfWork": "^workClassification",
                "genderOf": "^gender",
                "languageOf": "^workLanguage",
                "languageOfSet": "^setLanguage",
                "materialOfItem": "^material",
                "narrower": "^broader",
                "nationalityOf": "^nationality",
                "occupationOf": "^occupation",
                "professionalActivityOf": "^typeOfAgentActivity",
                "subjectOfSet": "^setAboutConcept",
                "subjectOfWork": "^workAboutConcept",
                "usedToProduce": "^typeOfItemBeginning",
            },
            "event": {
                "carriedOutBy": "agentOfEvent",
                "tookPlaceAt": "placeOfEvent",
                "used": "eventUsedSet",
                "causeOfEvent": "causeOfEvent",
                "causedCreationOf": "^causeOfWorkBeginning",
                "subjectOfSet": "^setAboutEvent",
                "subjectOfWork": "^workAboutEvent",
            },
            "place": {
                "partOf": "placePartOf",
                "partOfPlus": "placePartOf+",
                "activePlaceOfAgent": "^placeOfAgentActivity",
                "startPlaceOfAgent": "^placeOfAgentBeginning",
                "producedHere": "^placeOfItemBeginning",
                "createdHere": "^placeOfWorkBeginning",
                "endPlaceOfAgent": "^placeOfAgentEnding",
                "encounteredHere": "^placeOfItemEncounter",
                "placeOfEvent": "^placeOfEvent",
                "setPublishedHere": "^placeOfSetPublication",
                "publishedHere": "^placeOfWorkPublication",
                "subjectOfSet": "^setAboutPlace",
                "subjectOfWork": "^workAboutPlace",
            },
            "set": {
                "aboutConcept": "setAboutConcept",
                "aboutEvent": "setAboutEvent",
                "aboutItem": "setAboutItem",
                "aboutAgent": "setAboutAgent",
                "aboutPlace": "setAboutPlace",
                "aboutWork": "setAboutWork",
                "createdAt": "placeOfSetBeginning",
                "createdBy": "agentOfSetBeginning",
                "creationCausedBy": "causeOfSetBeginning",
                "curatedBy": "setCuratedBy",
                "publishedAt": "placeOfSetPublication",
                "publishedBy": "agentOfSetPublication",
                "containingSet": "^setMemberOfSet",
                "containingItem": "^itemMemberOfSet",
                "usedForEvent": "^eventUsedSet",
            },
            "work": {
                "aboutConcept": "workAboutConcept",
                "aboutEvent": "workAboutEvent",
                "aboutItem": "workAboutItem",
                "aboutAgent": "workAboutAgent",
                "aboutPlace": "workAboutPlace",
                "aboutWork": "workAboutWork",
                "createdAt": "placeOfWorkBeginning",
                "createdBy": "agentOfWorkBeginning",
                "creationCausedBy": "causeOfWorkBeginning",
                "creationInfluencedBy": "agentInfluenceOfWorkBeginning",
                "publishedAt": "placeOfWorkPublication",
                "publishedBy": "agentOfWorkPublication",
                "language": "workLanguage",
                "partOfWork": "workPartOf",
                "subjectOfSet": "^setAboutWork",
                "subjectOfWork": "^workAboutWork",
                "carriedBy": "^carries",
                "containsWork": "^partOfWork",
            },
        }
        # Property Path Notation:
        # ^elt is inverse path (e.g. from object to subject)
        # elt* is zero or more
        # elt+ is one or more
        # elt? is zero or one

    def translate_search(self, query, scope=None, limit=25, offset=0, sort="", order="", sortDefault="ZZZZZZZZZZ"):
        # Implement translation logic here
        self.counter = 0
        self.scored = []
        self.calculate_scores = False
        self.calculate_scores = True  # always calculate scores for now until cache key is fixed
        sparql = SelectQuery(limit=limit, offset=offset)
        for pfx, uri in self.prefixes.items():
            sparql.add_prefix(Prefix(pfx, uri))
        if sort and sort != "relevance":
            sparql.add_variables(["?uri", "(MIN(?sortWithDefault) AS ?sort)"])
        else:
            sparql.add_variables(["?uri", "(SUM(?score) AS ?sscore)"])
            self.calculate_scores = True

        where = Pattern()
        if scope is not None and scope != "any":
            t = Triple("?uri", "a", f"lux:{scope.title()}")
            where.add_triples([t])

        query.var = f"?uri"
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
            where.add_binding(Binding(f'COALESCE(?sortValue, "{sortDefault}")', "?sortWithDefault"))
            ob = OrderBy(["?sort"], order == "DESC")
            sparql.add_order_by(ob)

        sparql.set_where_pattern(where)
        return sparql

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
        query.var = f"?uri"
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
        query.var = f"?uri"
        self.translate_query(query, where)
        where.add_filter(Filter("?uri != <URI-HERE>"))

        sparql.set_where_pattern(where)
        gby = GroupBy(["?uri"])
        sparql.add_group_by(gby)
        ob = OrderBy(["?count"], True)
        sparql.add_order_by(ob)

        return sparql

    def translate_facet(self, query, facet, scope=None, limit=25, offset=0):
        self.calculate_scores = True
        self.counter = 0
        gb = GroupBy(["?facet"])
        ob = OrderBy(["?facetCount"], True)

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
        # only relationships
        if rel == "classification":
            return f"lux:{scope}{rel[0].upper()}{rel[1:]}"
        elif rel == "memberOf":
            typ = "Group" if scope == "agent" else "Set"
            return f"lux:{scope}{rel[0].upper()}{rel[1:]}{typ}"
        else:
            p = self.scope_fields[scope].get(rel, "missed")
            if p[0] == "^":
                return f"^lux:{p[1:]}"
            else:
                return f"lux:{p}"

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

    def get_leaf_predicate(self, field, scope):
        if field in ["height", "width", "depth", "weight", "dimension"]:
            return f"lux:{field}"

        if field in ["startDate", "producedDate", "createdDate"]:
            return [f"lux:startOf{scope.title()}Beginning", f"lux:endOf{scope.title()}Beginning"]
        elif field == "endDate":
            return [f"lux:startOf{scope.title()}Ending", f"lux:endOf{scope.title()}Ending"]
        elif field == "activeDate":
            return [f"lux:startOf{scope.title()}Activity", f"lux:endOf{scope.title()}Activity"]
        elif field == "publishedDate":
            return [f"lux:startOf{scope.title()}Publication", f"lux:endOf{scope.title()}Publication"]
        elif field == "encounteredDate":
            return [f"lux:startOf{scope.title()}Encounter", f"lux:endOf{scope.title()}Encounter"]

        if field == "hasDigitalImage":
            return f"lux:{scope}{field[0].upper()}{field[1:]}"
        elif field == f"{scope}HasDigitalImage":
            return f"lux:{field}"

        pred = self.scope_leaf_fields[scope].get(field, "missed")
        return pred

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
                parent.add_triples([Triple(query.var, pred, f'"{query.value}"')])
            elif query.field == "recordType":
                parent.add_triples([Triple(query.var, "a", f"lux:{query.value}")])
            else:
                self.do_text_search(query, parent, scope)

        elif typ == "date":
            # do date query per qlever
            dt = query.value

            # make sure date is in a valid format
            if not ":" in dt:
                dt += "T00:00:00Z"

            comp = query.comparitor
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

    def do_text_search(self, query, parent, scope):
        # extract words
        val = query.value.lower()
        if self.remove_diacritics:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode("ascii")
        try:
            shwords = shlex.split(val)
        except:
            raise
        phrases = [w for w in shwords if " " in w]
        words = val.replace('"', "").split()

        if self.min_word_chars > 1:
            words = [
                word.strip(whitespace + punctuation).ljust(self.min_word_chars, self.padding_char) for word in words
            ]

        top = Pattern()
        wx = 0

        if query.field == self.name_field:
            field = f"lux:{scope}Name"
            nameVar = f"?name_{self.counter}"
            top.add_triples(Triple(query.var, field, nameVar))
            svc = Pattern(service="textSearch")
            strips = []
            tsvar = f"?ts{self.counter}"

            # Only one field, so can have one service with one entry per word
            for w in words:
                cfvar = f"?cf{self.counter}{wx}"
                strips.append(Triple(tsvar, "textSearch:contains", cfvar))
                strips.append(Triple(cfvar, "textSearch:word", f'"{w}"'))
                if self.calculate_scores:
                    strips.append(Triple(cfvar, "textSearch:score", f"?score_{self.counter}{wx}"))
                wx += 1

            cfvar = f"?cf{self.counter}{wx}"
            strips.append(Triple(tsvar, "textSearch:contains", cfvar))
            strips.append(Triple(cfvar, "textSearch:entity", nameVar))
            svc.add_triples(strips)
            top.add_nested_graph_pattern(svc)
            # if self.calculate_scores:
            #    binds = []
            #    for x in range(wx):
            #        binds.append(f"COALESCE(?score_{self.counter}{x}, 0)")
            #    top.add_binding(Binding(" + ".join(binds), f"?score_{self.counter}"))

        elif query.field == self.anywhere_field:
            # This can't be text:A and text:B and text:C OR ref:A and ref:B and ref:C
            # as it could match only as text:A and ref:B and text:C
            # so must do (text:A OR ref:A) AND (text:B OR ref:B) AND (text:C OR ref:C)
            for w in words:
                wpatt = Pattern()
                p1 = Pattern()
                opt1 = self.make_sparql_ref(query, scope, w, wx, False)
                p1.add_nested_graph_pattern(opt1)
                opt2 = self.make_sparql_anywhere(query, scope, w, wx, True)
                p1.add_nested_graph_pattern(opt2)
                if self.calculate_scores:
                    p1.add_binding(
                        Binding(
                            f"COALESCE(?score_refs_{self.counter}{wx}, 0) * {self.refs_weight} + \
COALESCE(?score_name_{self.counter}{wx}, 0) * {self.name_weight} + \
COALESCE(?score_text_{self.counter}{wx}, 0) * {self.text_weight}",
                            f"?score_{self.counter}{wx}",
                        )
                    )
                wpatt.add_nested_graph_pattern(p1)

                p2 = Pattern(union=True)
                opt2 = self.make_sparql_anywhere(query, scope, w, wx, False)
                p2.add_nested_graph_pattern(opt2)
                opt1 = self.make_sparql_ref(query, scope, w, wx, True)
                p2.add_nested_graph_pattern(opt1)
                if self.calculate_scores:
                    p2.add_binding(
                        Binding(
                            f"COALESCE(?score_refs_{self.counter}{wx}, 0) * {self.refs_weight} + \
COALESCE(?score_name_{self.counter}{wx}, 0) * {self.name_weight} + \
COALESCE(?score_text_{self.counter}{wx}, 0) * {self.text_weight}",
                            f"?score_{self.counter}{wx}",
                        )
                    )
                wpatt.add_nested_graph_pattern(p2)
                top.add_nested_graph_pattern(wpatt)
                wx += 1

        parent.add_nested_graph_pattern(top)
        # Still need to coalesce the scores across different words
        if self.calculate_scores:
            binds = []
            for x in range(wx):
                binds.append(f"COALESCE(?score_{self.counter}{x}, 0)")
            parent.add_binding(Binding(" + ".join(binds), f"?score_{self.counter}"))
            self.scored.append(self.counter)
        if phrases:
            fvar = f"?fld2{self.counter}0"
            ### FIXME: How to also test OR in name text?
            for p in phrases:
                top.add_filter(Filter(f'CONTAINS(LCASE({fvar}), "{p}")'))

    def make_sparql_ref(self, query, scope, w, wx, optional):
        opt1 = Pattern(optional=optional)
        n = 1
        strips = []
        tsvar = f"?ts{n}{self.counter}{wx}"
        cfvar = f"?cf{n}{self.counter}{wx}"
        fldvar = f"?fld{n}{self.counter}{wx}"

        opt1.add_triples([Triple(query.var, f"lux:{scope}Any/lux:primaryName", fldvar)])

        svc = Pattern(service="textSearch")
        strips.append(Triple(tsvar, "textSearch:contains", cfvar))
        strips.append(Triple(cfvar, "textSearch:word", f'"{w}"'))
        if self.calculate_scores:
            strips.append(Triple(cfvar, "textSearch:score", f"?score_refs_{self.counter}{wx}"))
        cfvar = f"?cf{self.counter}{wx}2"
        strips.append(Triple(tsvar, "textSearch:contains", cfvar))
        strips.append(Triple(cfvar, "textSearch:entity", fldvar))

        svc.add_triples(strips)
        opt1.add_nested_graph_pattern(svc)

        return opt1

    def make_sparql_anywhere(self, query, scope, w, wx, optional):
        opt1 = Pattern(optional=optional)
        n = 2
        strips = []
        tsvar = f"?ts{n}{self.counter}{wx}"
        cfvar = f"?cf{n}{self.counter}{wx}"
        fldvar = f"?fld{n}{self.counter}{wx}"

        opt1.add_triples([Triple(query.var, f"lux:recordText", fldvar)])
        svc = Pattern(service="textSearch")
        strips.append(Triple(tsvar, "textSearch:contains", cfvar))
        strips.append(Triple(cfvar, "textSearch:word", f'"{w}"'))
        cfvar = f"?cf{n}{self.counter}{wx}2"
        strips.append(Triple(tsvar, "textSearch:contains", cfvar))
        strips.append(Triple(cfvar, "textSearch:entity", fldvar))

        if self.calculate_scores:
            strips.append(Triple(cfvar, "textSearch:score", f"?score_text_{self.counter}{wx}"))
        svc.add_triples(strips)
        opt1.add_nested_graph_pattern(svc)

        opt1n = Pattern(optional=True)
        n = 3
        strips = []
        tsvar = f"?ts{n}{self.counter}{wx}"
        cfvar = f"?cf{n}{self.counter}{wx}"
        fldvar = f"?fld{n}{self.counter}{wx}"

        opt1n.add_triples([Triple(query.var, f"lux:{scope}PrimaryName", fldvar)])

        svc = Pattern(service="textSearch")
        strips.append(Triple(tsvar, "textSearch:contains", cfvar))
        strips.append(Triple(cfvar, "textSearch:word", f'"{w}"'))
        cfvar = f"?cf{n}{self.counter}{wx}3"
        strips.append(Triple(tsvar, "textSearch:contains", cfvar))
        strips.append(Triple(cfvar, "textSearch:entity", fldvar))

        if self.calculate_scores:
            strips.append(Triple(cfvar, "textSearch:score", f"?score_name_{self.counter}{wx}"))
        svc.add_triples(strips)
        opt1n.add_nested_graph_pattern(svc)
        opt1.add_nested_graph_pattern(opt1n)

        return opt1
