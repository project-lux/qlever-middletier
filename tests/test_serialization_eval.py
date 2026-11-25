"""The serialization evaluation's parsers against what process_qlever_results gives the middle tier.

The bodies are QLever's own answers (local LUX QLever, 2026-11-25) to one VALUES row holding an
IRI, an xsd:int, an xsd:decimal, an xsd:double (QLever answers it as a decimal), a plain and a
language-tagged literal, literals with a tab and with double quotes, and an unbound variable. A
parser that could replace the current format must give exactly the same rows.
"""

import json

import pytest

pytest.importorskip("httpx")

from files.qlever_serialization_eval import (  # noqa: E402
    parse_csv, parse_qlever_json, parse_sparql_json, parse_sparql_xml, parse_tsv)

QLEVER_JSON = json.dumps({
    "selected": ["?iri", "?int", "?dec", "?dbl", "?plain", "?lang", "?tab", "?quote", "?unbound"], "status": "OK",
    "res": [["<http://example.org/a>", "\"3\"^^<http://www.w3.org/2001/XMLSchema#int>",
             "\"1.5\"^^<http://www.w3.org/2001/XMLSchema#decimal>", "\"2.5\"^^<http://www.w3.org/2001/XMLSchema#decimal>",
             "\"plain\"", "\"chat\"@fr", "\"a\tb\"", "\"say \"hi\"\"", None]],
    "resultSizeTotal": 1, "time": {"total": "4ms"}}).encode()
SPARQL_JSON = (
    b'{"head":{"vars":["iri","int","dec","dbl","plain","lang","tab","quote","unbound"]},"results":{"bindings":[{'
    b'"iri":{"type":"uri","value":"http://example.org/a"},'
    b'"int":{"datatype":"http://www.w3.org/2001/XMLSchema#int","type":"literal","value":"3"},'
    b'"dec":{"datatype":"http://www.w3.org/2001/XMLSchema#decimal","type":"literal","value":"1.5"},'
    b'"dbl":{"datatype":"http://www.w3.org/2001/XMLSchema#decimal","type":"literal","value":"2.5"},'
    b'"plain":{"type":"literal","value":"plain"},"lang":{"type":"literal","value":"chat","xml:lang":"fr"},'
    b'"tab":{"type":"literal","value":"a\\tb"},"quote":{"type":"literal","value":"say \\"hi\\""}}]},'
    b'"meta":{"query-time-ms":4,"result-size-total":1}}')
SPARQL_XML = (
    b'<?xml version="1.0"?>\n<sparql xmlns="http://www.w3.org/2005/sparql-results#">\n<head>\n'
    + b"".join(b'  <variable name="%s"/>\n' % n for n in
               [b"iri", b"int", b"dec", b"dbl", b"plain", b"lang", b"tab", b"quote", b"unbound"])
    + b'</head>\n<results>\n  <result>\n'
    b'    <binding name="iri"><uri>http://example.org/a</uri></binding>\n'
    b'    <binding name="int"><literal datatype="http://www.w3.org/2001/XMLSchema#int">3</literal></binding>\n'
    b'    <binding name="dec"><literal datatype="http://www.w3.org/2001/XMLSchema#decimal">1.5</literal></binding>\n'
    b'    <binding name="dbl"><literal datatype="http://www.w3.org/2001/XMLSchema#decimal">2.5</literal></binding>\n'
    b'    <binding name="plain"><literal>plain</literal></binding>\n'
    b'    <binding name="lang"><literal xml:lang="fr">chat</literal></binding>\n'
    b'    <binding name="tab"><literal>a\tb</literal></binding>\n'
    b'    <binding name="quote"><literal>say &quot;hi&quot;</literal></binding>\n'
    b'  </result>\n</results>\n</sparql>')
TSV = (b'?iri\t?int\t?dec\t?dbl\t?plain\t?lang\t?tab\t?quote\t?unbound\n'
       b'<http://example.org/a>\t3\t1.5\t2.5\t"plain"\t"chat"@fr\t"a b"\t"say "hi""\t\n')
CSV = (b'iri,int,dec,dbl,plain,lang,tab,quote,unbound\n'
       b'http://example.org/a,3,1.5,2.5,plain,chat,a\tb,"say ""hi""",\n')

TAB = 6  # the column holding a literal with a tab in it


@pytest.fixture
def reference():
    return parse_qlever_json(QLEVER_JSON)


def test_the_reference_is_what_the_middle_tier_sees(reference):
    assert reference["results"] == [["http://example.org/a", 3, 1.5, 2.5, '"plain"', '"chat"@fr', '"a\tb"',
                                     '"say "hi""', None]]
    assert reference["total"] == 1 and reference["variables"][0] == "?iri"


@pytest.mark.parametrize("parse, body", [(parse_sparql_json, SPARQL_JSON), (parse_sparql_xml, SPARQL_XML)])
def test_sparql_json_and_xml_rows_are_identical(reference, parse, body):
    parsed = parse(body)
    assert parsed["results"] == reference["results"]
    assert parsed["variables"] == reference["variables"]


def test_only_sparql_json_also_carries_the_total(reference):
    assert parse_sparql_json(SPARQL_JSON)["total"] == reference["total"]
    assert parse_sparql_json(SPARQL_JSON)["time"] == "4ms"
    assert parse_sparql_xml(SPARQL_XML)["total"] is None and parse_tsv(TSV)["total"] is None


def test_tsv_is_identical_except_a_literal_holding_a_tab(reference):
    row, want = parse_tsv(TSV)["results"][0], reference["results"][0]
    assert row[:TAB] + row[TAB + 1:] == want[:TAB] + want[TAB + 1:]
    assert row[TAB] == '"a b"' != want[TAB]  # QLever writes the tab as a space: not recoverable


def test_csv_cannot_tell_a_literal_from_an_iri(reference):
    row = parse_csv(CSV)["results"][0]
    assert row[0] == "http://example.org/a" and row[4] == "plain"  # the literal arrives unquoted, like the IRI
    assert row != reference["results"][0]
