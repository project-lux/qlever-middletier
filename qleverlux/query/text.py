"""Free-text search: turning a text query into QLever patterns.

Two very different shapes come out of here. A search against a *name* field
uses ``ql:has-word`` inside a ``GRAPH ?tf`` pattern so QLever reports a
per-word score; a search against the *any-text* field goes through the
materialized views declared in ``files/Qleverfile`` (``<scope>Words``), one
``SERVICE`` block per word.

Words are split the way QLever's text index splits them: on every character
that is not a letter or a digit, keeping accents (``côte`` and ``cote`` are
different words). So "children's" is the two words ``children`` and ``s``, as it
is in the index; asked for as one word it matched nothing. Only double quotes
make a phrase - an apostrophe is not a quote, which ``shlex`` took it for, so
"children's books" failed with "No closing quotation".

The index finds records with all the words; two things are then checked
against the text the words were found in, by one REGEX each
(``match_pattern``): a double-quoted phrase of more than one word, and a word
written with an apostrophe ("o'brien"). The words must come in order and
whole, ignoring case, separated by anything that is not a letter or a digit -
except where the query has an apostrophe, where only an apostrophe-like mark
(any of ``APOSTROPHES``) and/or spaces may come between. So "john o'brien"
matches "John O'Brien", "john o’brien" and "john o brien", but not
"John O. Brien".

In a name search that text is the name the words were found in, and the check
costs milliseconds. In a text search it is the record's primary name, its
record text or a linked record's name; each is searched from the text index on
its own, as a UNION of three sub-selects, because any other way of joining them
read all 34M item record texts (out of memory, or past the timeout). Even so a
phrase of rare words takes seconds and one of common words tens of seconds, so
in a text search an unquoted apostrophe word is not checked - that would make
every search for "children's books" take 16 s - and is just its words.

Per-word ``?tf_*`` scores are summed into ``?score_N``, which
``SparqlTranslator.translate_search`` in turn sums into ``?sscore`` to order by.
"""

from __future__ import annotations

import re
import unicodedata
from string import punctuation, whitespace

from qleverlux.sparql import Binding, BNode, Filter, SelectQuery, Triple
from qleverlux.sparql import GraphPattern as Pattern
from qleverlux.sparql.terms import string_literal

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

#: Marks the data writes an apostrophe with: straight and curly quotes, prime,
#: modifier letter, grave and acute accents, and a combining comma above
#: ("o̕brien"). None is special inside a regex character class.
APOSTROPHES = "'’‘′ʼ`´̕"

#: A word as QLever's text index has it: a run of letters and digits.
_WORD = re.compile(r"[^\W_]+")
#: A phrase: anything between double quotes. An unclosed quote is ignored.
_PHRASE = re.compile(r'"([^"]*)"')
#: A word written with an apostrophe, such as "o'brien" or "d'ivoire".
_JOINED = re.compile(rf"[^\W_]+(?:[{APOSTROPHES}][^\W_]+)+")
#: Text between two words that is an apostrophe, perhaps with spaces round it.
_APOSTROPHE_GAP = re.compile(rf"\s*[{APOSTROPHES}]+\s*")

# RE2 (QLever's REGEX), not Python: \pL and \pN are Unicode letters and digits.
_EDGE_START = r"(?:^|[^\pL\pN])"
_EDGE_END = r"(?:$|[^\pL\pN])"
_ANY_GAP = r"[^\pL\pN]+"
_APOSTROPHE_RE = rf"\s*[{APOSTROPHES}]?\s*"


def _normal(value: str) -> str:
    return unicodedata.normalize("NFC", value).lower()


def tokenize(value: str) -> tuple[list[str], list[str]]:
    """(words, phrases) of a text query, both lower-cased.

    Words are split as QLever's index splits text. A phrase is a double-quoted
    span of more than one word; a quoted single word is just that word.
    """
    value = _normal(value)
    words = _WORD.findall(value)
    phrases = [p.strip() for p in _PHRASE.findall(value) if len(_WORD.findall(p)) > 1]
    return words, phrases


def joined_words(value: str) -> list[str]:
    """Words written with an apostrophe outside any phrase ("o'brien"), lower-cased,
    each once. A phrase already checks the ones inside it."""
    return list(dict.fromkeys(_JOINED.findall(_PHRASE.sub(" ", _normal(value)))))


def match_pattern(text: str) -> str:
    """The RE2 pattern a phrase or an apostrophe word has to match, case aside:
    its words in order and whole, separated by anything that is not a letter or
    a digit, or, where ``text`` has an apostrophe, by at most an apostrophe-like
    mark and spaces. The words are letters and digits, so nothing needs escaping."""
    text = _normal(text)
    words = _WORD.findall(text)
    gaps = _WORD.split(text)[1:-1]
    parts = [words[0]]
    for gap, word in zip(gaps, words[1:]):
        parts.append(_APOSTROPHE_RE if _APOSTROPHE_GAP.fullmatch(gap) else _ANY_GAP)
        parts.append(word)
    return _EDGE_START + "".join(parts) + _EDGE_END


def _regex_filter(variable: str, text: str) -> Filter:
    return Filter(f'REGEX({variable}, {string_literal(match_pattern(text))}, "i")')


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
        # self.padding_char2 = "Þ"
        self.padding_char = b"\xc3\xbe".decode("utf-8")
        self.anywhere_field = ANYWHERE_FIELD
        self.name_field = NAME_FIELD

    def set_stopwords(self, stopwords):
        if type(stopwords) is list:
            self.stopwords = dict(zip(stopwords, [1] * len(stopwords)))
        elif type(stopwords) is dict:
            self.stopwords = stopwords

    def _index_words(self, words):
        """Words as the index holds them (padded, when it pads short words)."""
        if self.min_word_chars > 1:
            return [
                word.strip(whitespace + punctuation).ljust(self.min_word_chars, self.padding_char)
                for word in words
            ]
        return words

    def build(self, query, parent, scope, counter):
        """Add the patterns for a text clause to ``parent``.

        Raises ``ValueError`` if every word in the query is a stopword.
        """
        val = str(query.value)
        if self.remove_diacritics:
            val = (
                unicodedata.normalize("NFKD", val)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
        words, phrases = tokenize(val)
        # Each word once: a repeated word only repeats the same join.
        words = list(dict.fromkeys(w for w in words if w not in self.stopwords))
        if not words:
            # all words were stopwords, search as phrase?
            raise ValueError("No valid words found")
        words = self._index_words(words)

        top = Pattern()
        wx = 0

        if query.field == self.name_field:
            field = f"lux:{scope}Name"

            top.add_triples(Triple(query.var, field, f"?text_{counter}"))
            for w in words:
                gpat = Pattern(graph_name=f"?tf_{counter}_{wx}")
                gpat.add_triples(
                    Triple(f"?text_{counter}", "ql:has-word", string_literal(w))
                )
                top.add_nested_graph_pattern(gpat)
                wx += 1

            # The name the words were found in: checking it is cheap.
            for text in phrases + joined_words(val):
                top.add_filter(_regex_filter(f"?text_{counter}", text))

        elif query.field == self.anywhere_field:
            view = f"view:{scope}Words"
            for w in words:
                svc = Pattern(service=view)
                bnode = BNode()
                bnode.add_triples(Triple("", "view:column-word", string_literal(w)))
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

            # Quoted phrases only: see the module docstring for why an unquoted
            # apostrophe word is not checked here.
            for x, phrase in enumerate(phrases):
                top.add_nested_graph_pattern(self._phrase_sources(query.var, scope, phrase, counter, x))

        addn = " + ".join([f"?tf_{counter}_{i}" for i in range(wx)])
        bnd = Binding(addn, f"?score_{counter}")
        top.add_binding(bnd)
        parent.add_nested_graph_pattern(top)

    def _phrase_sources(self, var, scope, phrase, counter, x):
        """Records whose primary name, record text or a linked record's name has
        ``phrase``: a UNION of one sub-select per source, each starting from the
        text index (the phrase's words, or its stopwords if it has nothing else)
        and checked with the phrase's REGEX.

        Each source has to be searched on its own: with the sources joined to
        each other or to the word views, QLever read every record text in the
        scope before filtering (34M for items).
        """
        words, _ = tokenize(phrase)
        words = list(dict.fromkeys(words))
        lookup = self._index_words([w for w in words if w not in self.stopwords] or words)
        text = f"?phrase_{counter}_{x}"
        sources = Pattern()
        paths = (f"lux:{scope}PrimaryName", f"lux:{scope}RecordText", f"lux:{scope}Any/lux:primaryName")
        for i, path in enumerate(paths):
            where = Pattern()
            where.add_triples(Triple(var, path, text))
            for j, w in enumerate(lookup):
                gpat = Pattern(graph_name=f"?pg_{counter}_{x}_{j}")
                gpat.add_triples(Triple(text, "ql:has-word", string_literal(w)))
                where.add_nested_graph_pattern(gpat)
            where.add_filter(_regex_filter(text, phrase))
            select = SelectQuery(distinct=True)
            select.add_variables([var])
            select.set_where_pattern(where)
            branch = Pattern(union=i > 0)
            branch.add_nested_select_query(select)
            sources.add_nested_graph_pattern(branch)
        return sources
