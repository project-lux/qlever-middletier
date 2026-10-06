#!/usr/bin/env python
"""Read a query from stdin, print the SPARQL it translates to.

The fastest feedback loop for query-translation changes: it exercises the
parser -> JsonReader -> SparqlTranslator chain without needing QLever, a
database, or the middle tier. Accepts either a simple string query or a LUX
JSON query.
"""

import json
import sys

from luxql import JsonReader
from luxql.string_parser import QueryParser

from qleverlux.query.catalogue import build_lux_config
from qleverlux.query.translator import SparqlTranslator


def main(scope="item"):
    cfg = build_lux_config()
    rdr = JsonReader(cfg)
    st = SparqlTranslator(cfg)
    query_parser = QueryParser()

    query_string = input("Enter your query string: ")

    if query_string.lstrip().startswith("{"):
        qjs = json.loads(query_string)
    else:
        qjs = query_parser.parse(query_string).to_json()

    print(json.dumps(qjs, indent=2))

    parsed = rdr.read(qjs, scope)
    print(st.translate_search(parsed, scope=scope).get_text())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "item")
