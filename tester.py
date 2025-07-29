# read query string from user, translate to json, translate to sparql, spit out the query
#

import json
from qleverlux.boolean_query_parser import BooleanQueryParser
from luxql import JsonReader, LuxConfig
from qleverlux.sparql import SparqlTranslator

query_parser = BooleanQueryParser()
cfg = LuxConfig()
rdr = JsonReader(cfg)
st = SparqlTranslator(cfg)

query_string = input("Enter your query string: ")

if query_string[0] == "{":
    qjs = json.loads(query_string)
else:
    q = query_parser.parse(query_string)
    # now translate AST into JSON query
    qjs = q.to_json()
print(qjs)
scope = "item"
parsed = rdr.read(qjs, scope)
spq = st.translate_search(parsed, scope=scope)
qt = spq.get_text()
print(qt)
