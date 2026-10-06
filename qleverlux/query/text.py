"""Free-text search: turning a text query into QLever patterns.

Two very different shapes come out of here. A search against a *name* field
uses ``ql:has-word`` inside a ``GRAPH ?tf`` pattern so QLever reports a
per-word score; a search against the *any-text* field goes through the
materialized views declared in ``files/Qleverfile`` (``<scope>Words``), one
``SERVICE`` block per word. Quoted phrases become ``CONTAINS()`` filters over a
UNION of the three text sources.

Per-word ``?tf_*`` scores are summed into ``?score_N``, which
``SparqlTranslator.translate_search`` in turn sums into ``?sscore`` to order by.
"""

from __future__ import annotations

import shlex
import unicodedata
from string import punctuation, whitespace

from qleverlux.sparql import Binding, BNode, Filter, Triple
from qleverlux.sparql import GraphPattern as Pattern

#: The LUX search-term names this module and the translator special-case.
ANYWHERE_FIELD = "text"
ID_FIELD = "id"
NAME_FIELD = "name"

#: Relative weights the <scope>Words materialized views are built with, kept
#: here because the relevance ordering depends on them. Changing these means
#: rebuilding the views in files/Qleverfile.
NAME_WEIGHT = 14
TEXT_WEIGHT = 5
REFS_WEIGHT = 1


class TextSearchBuilder:
    """Builds the graph patterns for one free-text clause.

    Holds the tunables that used to sit on the translator; ``build()`` is the
    old ``SparqlTranslator.do_text_search`` with the translator's clause
    counter passed in rather than read off ``self``.
    """

    def __init__(
        self,
        stopwords=None,
        remove_diacritics: bool = False,
        min_word_chars: int = 0,
    ):
        self.stopwords = {}
        if stopwords:
            self.set_stopwords(stopwords)
        self.remove_diacritics = remove_diacritics
        self.min_word_chars = min_word_chars
        # self.padding_char2 = "\u00de"
        self.padding_char = b"\xc3\xbe".decode("utf-8")
        self.anywhere_field = ANYWHERE_FIELD
        self.name_field = NAME_FIELD

    def set_stopwords(self, stopwords):
        if type(stopwords) is list:
            self.stopwords = dict(zip(stopwords, [1] * len(stopwords)))
        elif type(stopwords) is dict:
            self.stopwords = stopwords

    def build(self, query, parent, scope, counter):
        """Add the patterns for a text clause to ``parent``.

        Raises ``ValueError`` if every word in the query is a stopword.
        """
        # extract words
        val = query.value.lower()
        if self.remove_diacritics:
            val = (
                unicodedata.normalize("NFKD", val)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
        try:
            shwords = shlex.split(val)
        except:
            raise
        phrases = [w for w in shwords if " " in w]
        words1 = val.replace('"', "").split()
        words = []
        for w in words1:
            if w not in self.stopwords:
                words.append(w)

        if not words:
            # all words were stopwords, search as phrase?
            raise ValueError("No valid words found")

        if self.min_word_chars > 1:
            words = [
                word.strip(whitespace + punctuation).ljust(
                    self.min_word_chars, self.padding_char
                )
                for word in words
            ]

        top = Pattern()
        wx = 0

        if query.field == self.name_field:
            field = f"lux:{scope}Name"

            top.add_triples(Triple(query.var, field, f"?text_{counter}"))
            for w in words:
                gpat = Pattern(graph_name=f"?tf_{counter}_{wx}")
                gpat.add_triples(
                    Triple(f"?text_{counter}", "ql:has-word", f'"{w}"')
                )
                top.add_nested_graph_pattern(gpat)
                wx += 1

            for p in phrases:
                top.add_filter(Filter(f'CONTAINS(?text_{counter}, "{p}")'))

        elif query.field == self.anywhere_field:
            view = f"view:{scope}Words"
            for w in words:
                # svar = f"?view_"
                svc = Pattern(service=view)
                bnode = BNode()
                bnode.add_triples(Triple("", "view:column-word", f'"{w}"'))
                bnode.add_triples(Triple("", "view:column-uri", query.var))
                # Must match the variables summed into ?score_N below, and must
                # be qualified by the clause counter: two text clauses in one
                # query would otherwise both bind ?tf_0 and join on it.
                bnode.add_triples(
                    Triple("", "view:column-score", f"?tf_{counter}_{wx}")
                )
                svc.add_bnode(bnode)
                top.add_nested_graph_pattern(svc)
                wx += 1

            if phrases:
                # Do this pattern:
                """
                SELECT ?uri WHERE {
                ?uri a lux:Item .
                {
                    SERVICE view:itemWords {
                    [
                        view:column-word "thomas" ;
                        view:column-uri ?uri ;
                    ]
                    }
                    SERVICE view:itemWords {
                    [
                        view:column-word "eugene" ;
                        view:column-uri ?uri ;
                    ]
                    }
                    ?uri lux:itemPrimaryName ?text .
                    FILTER (CONTAINS(?text,"thomas eugene"))
                }
                UNION {
                    SERVICE view:itemWords {
                    [
                        view:column-word "thomas" ;
                        view:column-uri ?uri ;
                    ]
                    }
                    SERVICE view:itemWords {
                    [
                        view:column-word "eugene" ;
                        view:column-uri ?uri ;
                    ]
                    }
                    ?uri lux:recordText ?text .
                    FILTER (CONTAINS(?text,"thomas eugene"))
                }
                UNION {
                    SERVICE view:itemWords {
                    [
                        view:column-word "thomas" ;
                        view:column-uri ?uri ;
                    ]
                    }
                    SERVICE view:itemWords {
                    [
                        view:column-word "eugene" ;
                        view:column-uri ?uri ;
                    ]
                    }
                    ?uri lux:itemAny/lux:primaryName ?text .
                    FILTER (CONTAINS(?text,"thomas eugene"))
                }
                }
                GROUP BY ?uri
                """

                # Calculate the strings and filter
                p1 = Pattern()
                p1.add_triples(Triple(query.var, f"lux:{scope}PrimaryName", "?text"))
                p2 = Pattern(union=True)
                p2.add_triples(Triple(query.var, "lux:recordText", "?text"))
                p3 = Pattern(union=True)
                p3.add_triples(
                    Triple(query.var, f"lux:{scope}Any/lux:primaryName", "?text")
                )
                top.add_nested_graph_pattern(p1)
                top.add_nested_graph_pattern(p2)
                top.add_nested_graph_pattern(p3)
                for p in phrases:
                    top.add_filter(Filter(f'CONTAINS(?text, "{p}")'))

        addn = " + ".join([f"?tf_{counter}_{i}" for i in range(wx)])
        bnd = Binding(addn, f"?score_{counter}")
        top.add_binding(bnd)
        parent.add_nested_graph_pattern(top)
