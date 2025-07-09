# configuration for middletier

import os
import json
from luxql import JsonReader, LuxConfig
from luxql.sparql import SparqlTranslator
from argparse import ArgumentParser
import getpass

cfg = LuxConfig()
rdr = JsonReader(cfg)
st = SparqlTranslator(cfg)


parser = ArgumentParser()

user = getpass.getuser()
table = "lux_data_cache"
sparql = "http://localhost:7010/sparql"
port = 5000
uri_host = "localhost"
protocol = "http"
path = "/"
pageLength = 20
data_uri = "https://lux.collections.yale.edu/"

parser.add_argument("--user", type=str, help="Postgres username", default=user)
parser.add_argument("--db", type=str, help="Postgres database", default=user)
parser.add_argument("--table", type=str, help="Postgres records table", default=table)

parser.add_argument("--loglevel", type=str, help="Log level for uvicorn", default="info")
parser.add_argument("--sparql", type=str, help="SPARQL endpoint URL", default=sparql)

parser.add_argument("--port", type=int, help="Port for uvicorn", default=port)
parser.add_argument("--host", type=str, help="Host for URI substitution", default=uri_host)
parser.add_argument("--protocol", type=str, help="Protocol for URI substitution", default=protocol)
parser.add_argument("--path", type=str, help="Path for URI substitution", default=path)
parser.add_argument("--data-uri", type=str, help="Data URI for URI substitution", default=data_uri)

parser.add_argument("--pageLength", type=int, help="Page length for pagination", default=20)
parser.add_argument("--portal", type=str, help="Which source unit, if any, to filter for", default="")

args, rest = parser.parse_known_args()

MY_URI = f"{args.protocol}://{args.host}:{args.port}{args.path}"
SPARQL_ENDPOINT = args.sparql
PAGE_LENGTH = args.pageLength
DATA_URI = args.data_uri
PG_TABLE = args.table
PORTAL_SOURCE = args.portal

ENGLISH = "http://vocab.getty.edu/aat/300388277"
PRIMARY = "http://vocab.getty.edu/aat/300404670"
RESULTS_FIELDS = [
    "produced_by",
    "created_by",
    "encountered_by",
    "classified_as",
    "member_of",
    "language",
    "referred_to_by",
    "representation",
    "part_of",
    "broader",
    "defined_by",
    "took_place_at",
    "timespan",
    "carried_out_by",
]

with open("config/config_facets.json") as fh:
    facets = json.load(fh)

qs = os.listdir("queries")
queries = {}
for q in qs:
    if q.endswith(".json"):
        with open(f"queries/{q}", "r") as f:
            queries[q[:-5]] = json.load(f)

with open("config/query_by_scope.json") as fh:
    hal_queries = json.load(fh)
for block in hal_queries.values():
    for k, v in block.items():
        if k.endswith("RelatedAgents") or k.endswith("RelatedConcepts") or k.endswith("RelatedPlaces"):
            block[k] = k
        else:
            try:
                block[k] = queries[v]
            except Exception:
                print(f"Failed to resolve {k}")

with open("config/hal_link_templates.json") as fh:
    hal_link_templates = json.load(fh)
for k, v in hal_link_templates.items():
    hal_link_templates[k] = v.replace("{searchUriHost}", MY_URI[:-1])

# strip trailing /

# TODO: Test if a single query for ?uri ?pred <uri> and then looking for which hal
# links match the returned predicates would be faster or not

sparql_hal_queries = {}
for scope in hal_queries:
    sparql_hal_queries[scope] = {}
    for hal, query in hal_queries[scope].items():
        if type(query) is str:
            sparql_hal_queries[scope][hal] = ""
            continue
        try:
            qscope = query["_scope"]
        except KeyError:
            # Already been processed
            continue
        try:
            parsed = rdr.read(query, qscope)
        except Exception as e:
            print(f"Error parsing query for {hal}: {e}\n{query}")
            continue
        spq = st.translate_search_count(parsed, qscope)
        sparql_hal_queries[scope][hal] = spq

with open("config/config_sorts.json", "r") as f:
    sorts = json.load(f)

with open("config/config_related_list_names.json", "r") as f:
    related_list_names = json.load(f)

related_list_queries = {}
related_list_sparql = {}
for name in related_list_names.keys():
    bits = name.split("-")
    bits.append("padding-for-id")
    q = {}
    top_q = q
    for b in bits[:-1]:
        q[b] = {}
        q = q[b]
    q["id"] = "URI-HERE"
    scope = None
    for s in cfg.lux_config["terms"]:
        if bits[0] in cfg.lux_config["terms"][s]:
            scope = s
            break
    if scope is None:
        print(f"Couldn't find scope for {name}")
        continue
    lq = rdr.read(top_q, scope)
    spq = st.translate_search_related(lq)
    try:
        related_list_queries[scope][name] = json.dumps(top_q, separators=(",", ":"))
        related_list_sparql[scope][name] = spq.get_text()
    except Exception:
        related_list_queries[scope] = {name: json.dumps(top_q, separators=(",", ":"))}
        related_list_sparql[scope] = {name: spq.get_text()}
