"""Text and name searches split words as QLever's index does, and quote nothing but phrases.

"children's books" failed with "No closing quotation": shlex took the
apostrophe for a quote. And asked for as one word, "children's" matched
nothing, because the index has it as "children" and "s". Phrases and words
written with an apostrophe are then checked with one REGEX against the text the
words were found in, so "o'brien" is not "o. brien".
"""

import re

import pytest

from qleverlux.query.text import joined_words, match_pattern, tokenize
from qleverlux.sparql.terms import string_literal


@pytest.mark.parametrize("value, words, phrases", [
    ("children's books", ["children", "s", "books"], []),
    ("children’s books", ["children", "s", "books"], []),          # typographic apostrophe
    ("self-portrait", ["self", "portrait"], []),
    ("O'Brien", ["o", "brien"], []),
    ("U.S. 19th c.", ["u", "s", "19th", "c"], []),
    ("Côte d'Ivoire", ["côte", "d", "ivoire"], []),                # accents kept, as in the index
    ("côte", ["côte"], []),                                  # decomposed accent composed first
    ('"Rembrandt van Rijn" etching', ["rembrandt", "van", "rijn", "etching"], ["rembrandt van rijn"]),
    ('"children\'s books"', ["children", "s", "books"], ["children's books"]),
    ('"single" word', ["single", "word"], []),                     # a quoted single word is a word
    ('unclosed "quote', ["unclosed", "quote"], []),
    ("snake_case", ["snake", "case"], []),
])
def test_tokenize(value, words, phrases):
    assert tokenize(value) == (words, phrases)


@pytest.mark.parametrize("value, joined", [
    ("john o'brien", ["o'brien"]),
    ("côte d’ivoire", ["d’ivoire"]),
    ('"john o\'brien" o\'brien d\'arcy', ["o'brien", "d'arcy"]),   # in a phrase it is the phrase's
    ("children s books", []),
    ("o' brien", []),                                               # spaced: just words
])
def test_joined_words(value, joined):
    assert joined_words(value) == joined


def python_re(pattern):
    """The RE2 pattern in Python's dialect, which has no \\pL / \\pN."""
    return re.compile(pattern.replace(r"[^\pL\pN]", r"[\W_]"), re.IGNORECASE)


@pytest.mark.parametrize("query, text, matches", [
    ("o'brien", "John O'Brien", True),
    ("o'brien", "john o’brien, 1900", True),
    ("o'brien", "JOHN O BRIEN", True),
    ("o'brien", "john o' brien", True),
    ("o'brien", "John O. Brien", False),
    ("o'brien", "o'callaghan, j. brien", False),
    ("o'brien", "john o'briens", False),
    ("night watch", "The Night Watch", True),
    ("night watch", "night-watch", True),
    ("night watch", "midnight watchman", False),
    ("night watch", "watch night", False),
    ("children's books", "Children’s Books", True),
    ("children's books", "childrens books", True),       # an apostrophe may be left out...
    ("children's books", "children s books", True),
    ("children's books", "children books", False),       # ...but not the s
    ("children's books", "children; s. books", False),
    ("côte d'ivoire", "abidjan (côte d’ivoire)", True),
    ("côte d'ivoire", "cote d'ivoire", False),
])
def test_match_pattern(query, text, matches):
    assert bool(python_re(match_pattern(query)).search(text)) is matches


@pytest.fixture(scope="module")
def compile_search():
    pytest.importorskip("luxql")
    from qleverlux.query.catalogue import QueryCatalogue
    from qleverlux.settings import load_settings

    catalogue = QueryCatalogue(load_settings([]))

    def compile_(query, scope):
        parsed = catalogue.json_reader.read(query, scope)
        return catalogue.translator.translate_search(parsed, scope=scope).get_text()

    return compile_


def test_an_apostrophe_in_a_name_search_is_checked_against_the_name(compile_search):
    text = compile_search({"classification": {"name": "children's books"}}, "work")
    assert re.findall(r'ql:has-word "([^"]*)"', text) == ["children", "s", "books"]
    assert len(re.findall(r"REGEX\(\?text_1, ", text)) == 1


def test_an_apostrophe_in_a_text_search_is_just_its_words(compile_search):
    text = compile_search({"text": "children’s books"}, "work")
    assert re.findall(r'view:column-word "([^"]*)"', text) == ["children", "s", "books"]
    assert "REGEX" not in text


def test_a_name_phrase_is_one_regex_on_the_name(compile_search):
    text = compile_search({"name": '"van rijn" rembrandt'}, "agent")
    assert re.findall(r"REGEX\((\?\w+), ", text) == ["?text_0"]
    assert "UNION" not in text


def test_a_text_phrase_searches_each_source_from_the_text_index(compile_search):
    text = compile_search({"text": '"the night watch"'}, "item")
    for path in ("lux:itemPrimaryName", "lux:itemRecordText", "lux:itemAny/lux:primaryName"):
        assert f"?uri {path} ?phrase_0_0" in text
    assert text.count("SELECT DISTINCT ?uri") == 3 and text.count("UNION") == 2
    assert text.count('ql:has-word "night"') == 3 and 'ql:has-word "the"' not in text  # stopwords: not looked up
    assert text.count("REGEX(?phrase_0_0, ") == 3
    assert "lux:recordText" not in text


def test_a_phrase_of_stopwords_is_looked_up_by_its_stopwords(compile_search):
    text = compile_search({"text": 'hamlet "it be not"'}, "work")   # all three are stopwords
    assert re.findall(r'view:column-word "([^"]*)"', text) == ["hamlet"]
    assert text.count('ql:has-word "be"') == 3


def test_each_word_is_looked_up_once(compile_search):
    text = compile_search({"text": '"to or to" to or'}, "work")
    assert re.findall(r'view:column-word "([^"]*)"', text) == ["to", "or"]
    assert text.count('ql:has-word "to"') == 3   # once per source


def test_a_backslash_cannot_break_out_of_the_literal(compile_search):
    text = compile_search({"name": 'say "it\'s \\ fine" now'}, "concept")
    # the backslash is not a letter or digit: it only separates words ("it" is a stopword)
    assert re.findall(r'ql:has-word "([^"]*)"', text) == ["say", "s", "fine", "now"]
    phrase = re.search(r'REGEX\(\?text_0, ("(?:[^"\\\\]|\\\\.)*"), "i"\)', text).group(1)
    assert phrase == string_literal(match_pattern("it's \\ fine"))
    assert "fine" in phrase and "\\\\ " not in phrase


def test_nothing_but_punctuation_is_a_bad_query(compile_search):
    with pytest.raises(ValueError, match="No valid words"):
        compile_search({"name": "' \" -"}, "concept")
