# configuration for middletier

import os
import sys
import json
from luxql import luxql as luxql_mod
from luxql import JsonReader, LuxConfig
from qleverlux.sparql import SparqlTranslator
from argparse import ArgumentParser
from dotenv import load_dotenv
from getpass import getuser

from pydantic import BaseModel
from enum import StrEnum

if "--pgpass" in sys.argv or "--qlpass" in sys.argv:
    print("Don't put passwords in the command line, as they're visible via ps")
    print("Instead use the .env file with QLMT_PGPASS / QLMT_QLPASS")
    sys.exit(1)

load_dotenv()


class MTConfig:
    def __init__(self):
        # Environment variables to connect to the PostgresQL record cache

        self.config_path = os.getenv("QLMT_CONFIG_PATH", "config")
        self.queries_path = os.getenv("QLMT_QUERIES_PATH", "queries")
        self.hal_cache_path = os.getenv("QLMT_HAL_CACHE_PATH", "hal_cache")

        self.pguser = os.getenv("QLMT_PGUSER", getuser())
        self.pgpass = os.getenv("QLMT_PGPASS", "")
        self.pghost = os.getenv("QLMT_PGHOST", "")
        self.pgport = int(os.getenv("QLMT_PGPORT", 5432))
        self.pgdb = os.getenv("QLMT_PGDB", getuser())
        self.pgtable = os.getenv("QLMT_PGTABLE", "lux_data_cache")
        self.pgsslmode = os.getenv("QLMT_PGSSLMODE", "require")

        # Environment variables to connect to the QLever SPARQL endpoint
        self.qlproto = os.getenv("QLMT_QLPROTO", "http")
        self.qlhost = os.getenv("QLMT_QLHOST", "localhost")
        self.qlport = int(os.getenv("QLMT_QLPORT", 7010))
        self.qlpath = os.getenv("QLMT_QLPATH", "sparql")
        # self.qluser = os.getenv("QLMT_QLUSER", "")
        # self.qlpass = os.getenv("QLMT_QLPASS", "")  # API key for update?

        # Environment variables for where the middle tier listens
        self.mthost = os.getenv("QLMT_MTHOST", "0.0.0.0")
        self.mtport = int(os.getenv("QLMT_MTPORT", 5000))
        self.mtproto = os.getenv("QLMT_MTPROTO", "https")
        self.mtpath = os.getenv("QLMT_MTPATH", "")
        self.cert_name = os.getenv("QLMT_CERTNAME", "qleverlux")

        # Environment variables for replacing the data URIs for this middle tier instance
        self.data_uri = os.getenv("QLMT_DATAURI", "https://lux.collections.yale.edu/")
        self.replace_proto = os.getenv("QLMT_REPLACE_PROTO", "https")
        self.replace_host = os.getenv("QLMT_EXTERNAL_HOST", "qleverlux.collections.yale.edu")
        self.replace_port = int(os.getenv("QLMT_EXTERNAL_PORT", -1))
        self.replace_path = os.getenv("QLMT_EXTERNAL_PATH", "")

        # Advanced Search config to load
        self.search_config = os.getenv("QLMT_SEARCH_CONFIG", "")

        # Page Length for one search results page
        self.page_length = int(os.getenv("QLMT_PAGELENGTH", 20))

        # Source value to run as a portal
        # Options: YPM, YCBA, YUAG, PMC, IPCH
        self.portal = os.getenv("QLMT_PORTAL", "")  # YPM

        # log level for output
        self.log_level = os.getenv("QLMT_LOGLEVEL", "info")

        # use stopwords or not, default to yes
        self.use_stopwords = os.getenv("QLMT_USESTOPWORDS", "true").lower() == "true"

        # use httpx or aiohttp, default to httpx for HTTP/2 support
        self.use_httpx = os.getenv("QLMT_USEHTTPX", "true").lower() == "true"

        # Now look for overrides from the command line

        parser = ArgumentParser()

        parser.add_argument("--config-path", type=str, help="Path to config file directory", default=self.config_path)
        parser.add_argument("--queries-path", type=str, help="Path to queries directory", default=self.queries_path)
        parser.add_argument(
            "--hal-cache-path", type=str, help="Path to HAL cache directory", default=self.hal_cache_path
        )

        parser.add_argument("--pghost", type=str, help="Postgres host", default=self.pghost)
        parser.add_argument("--pgport", type=int, help="Postgres port", default=self.pgport)
        parser.add_argument("--pguser", type=str, help="Postgres username", default=self.pguser)
        parser.add_argument("--pgdb", type=str, help="Postgres database", default=self.pgdb)
        parser.add_argument("--pgtable", type=str, help="Postgres records table", default=self.pgtable)

        parser.add_argument("--qlproto", type=str, help="Qlever protocol for SPARQL", default=self.qlproto)
        parser.add_argument("--qlhost", type=str, help="Qlever host for SPARQL", default=self.qlhost)
        parser.add_argument("--qlport", type=int, help="Qlever port for SPARQL", default=self.qlport)
        parser.add_argument("--qlpath", type=str, help="Qlever path for SPARQL", default=self.qlpath)

        parser.add_argument("--mtproto", type=str, help="HTTP/HTTPS for middletier", default=self.mtproto)
        parser.add_argument("--mthost", type=str, help="Middletier listen host", default=self.mthost)
        parser.add_argument("--mtport", type=int, help="Middletier listen port", default=self.mtport)
        parser.add_argument("--mtpath", type=str, help="Middletier listen path", default=self.mtpath)

        parser.add_argument("--data-uri", type=str, help="Data URI for URI substitution", default=self.data_uri)
        parser.add_argument(
            "--replace-proto", type=str, help="Protocol for URI substitution", default=self.replace_proto
        )
        parser.add_argument("--replace-host", type=str, help="Host for URI substitution", default=self.replace_host)
        parser.add_argument("--replace-port", type=int, help="Port for replacement", default=self.replace_port)
        parser.add_argument("--replace-path", type=str, help="Path for URI substitution", default=self.replace_path)

        parser.add_argument("--log-level", type=str, help="Log level for uvicorn", default=self.log_level)
        parser.add_argument("--cert-name", type=str, help="prefix for cert files", default=self.cert_name)

        parser.add_argument("--page-length", type=int, help="Page length for pagination", default=self.page_length)
        parser.add_argument(
            "--portal", type=str, help="Which source unit, if any, to filter for", default=self.portal
        )
        parser.add_argument("--facet-delay", type=int, help="Delay in milliseconds for facets", default=0)
        parser.add_argument("--use-stopwords", action="store_true", help="Use stopwords")
        parser.add_argument("--use-httpx", action="store_true", help="Use stopwords")
        parser.add_argument(
            "--search-config", type=str, help="Path to search configuration file", default=self.search_config
        )

        args, rest = parser.parse_known_args()
        self.remaining_args = rest

        # overwrite ENV defaults from CLI args
        for key, value in vars(args).items():
            setattr(self, key, value)

        # Allow an upstream load balancer to have a different port
        if self.replace_port > 0:
            self.replace_port = f":{args.replace_port}"
        else:
            self.replace_port = ""
        if self.qlport > 0:
            self.qlport = f":{args.qlport}"
        else:
            self.qlport = ""

        self.mt_uri = f"{args.replace_proto}://{args.replace_host}{self.replace_port}{args.replace_path}"
        self.sparql_endpoint = f"{args.qlproto}://{args.qlhost}{self.qlport}/{args.qlpath}"

        with open(os.path.join(self.config_path, "facets.json")) as fh:
            self.facets = json.load(fh)

        qs = os.listdir(self.queries_path)
        self.queries = {}
        for q in qs:
            if q.endswith(".json"):
                with open(os.path.join(self.queries_path, q), "r") as f:
                    self.queries[q[:-5]] = json.load(f)

        with open(os.path.join(self.config_path, "hal_links.json")) as fh:
            self.hal_queries = json.load(fh)
            for v in self.hal_queries.values():
                v["template"] = v["template"].replace("{searchUriHost}", self.mt_uri[:-1])

        with open(os.path.join(self.config_path, "sorts.json"), "r") as f:
            self.sorts = json.load(f)

        with open(os.path.join(self.config_path, "related_lists.json"), "r") as f:
            self.related_list_names = json.load(f)

        with open(os.path.join(self.config_path, "related_list_scopes.json"), "r") as f:
            self.related_list_scopes = json.load(f)

        with open(os.path.join(self.config_path, "terms_inverse.json"), "r") as f:
            self.inverses = json.load(f)

        with open(os.path.join(self.config_path, "stopwords.json"), "r") as f:
            self.stopwords = json.load(f)

        self.aat_english = "http://vocab.getty.edu/aat/300388277"
        self.aat_primary = "http://vocab.getty.edu/aat/300404670"
        self.results_fields = [
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

        # Init LuxQL FIXME: pass search_config through from env/cli
        self.lux_config = LuxConfig()
        # recache it into the module so auto configs use it
        luxql_mod._cached_lux_config = self.lux_config

        self.lux_config.lux_config["terms"]["work"]["workCreationOrPublicationDate"] = {
            "label": "Creation or Publication Date",
            "relation": "date",
        }
        self.lux_config.lux_config["terms"]["set"]["setCreationOrPublicationDate"] = {
            "label": "Creation or Publication Date",
            "relation": "date",
        }
        self.lux_config.inverted["workCreationOrPublicationDate"] = ["work"]
        self.lux_config.inverted["setCreationOrPublicationDate"] = ["set"]

        # Trim facets to those with search terms
        for k, v in list(self.facets.items())[:]:
            stn = v["searchTermName"]
            found = False
            for scope, terms in self.lux_config.lux_config["terms"].items():
                if stn in terms:
                    found = True
                    break
            if not found:
                print(f" *** Could not find search term {stn} for facet {k} in search terms, deleting ***")
                del self.facets[k]

        self.json_reader = JsonReader(self.lux_config)

        # And init translation layer
        self.sparql_translator = SparqlTranslator(self.lux_config, self)

        self.sparql_hal_queries = {}
        self.related_list_json = {}
        self.related_list_sparql = {}
        self.cache_sparql_queries()

    def make_related_json_stub(self, qname, scope, qscope):
        # given created, createdBy
        # produce AND: [createdBy: id: X, createdBy: id: Y]

        from_uri = "V_FROM_URI"
        to_uri = "V_TO_URI"
        fields = qname.split("-")

        q = {"AND": []}
        if len(fields) == 2:
            inv = self.inverses[scope][fields[0]]
            q["AND"].append({inv: {"id": from_uri}})
            q["AND"].append({fields[1]: {"id": to_uri}})
        else:
            # Find the first point at which query scope is the same as target scope
            aq = {}
            topa = aq
            target_scope = scope
            while target_scope != qscope:
                f = fields.pop(0)
                inv = self.inverses[target_scope][f]
                target_scope = self.lux_config.lux_config["terms"][target_scope][f]["relation"]
                aq[inv] = {}
                aq = aq[inv]
            aq["id"] = from_uri
            q["AND"].append(topa)
            bq = {}
            topb = bq
            for f in fields:
                bq[f] = {}
                bq = bq[f]
            bq["id"] = to_uri
            q["AND"].append(topb)
        return q

    def make_related_query_stub(self, scope, qtype):
        names = []
        fragments = []
        flds = self.sparql_translator.scope_fields

        PREFIXES = """
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX qlss: <https://qlever.cs.uni-freiburg.de/spatialSearch/>
PREFIX textSearch: <https://qlever.cs.uni-freiburg.de/textSearch/>
PREFIX lux: <https://lux.collections.yale.edu/ns/>
"""

        SUB_TEMPLATE = """
    OPTIONAL {
      SELECT ?uri (COUNT(?what) AS ?V_NAME_REL) WHERE {
        ?uri V_URI_REL ?what .
        ?what V_TARGET_REL <V_TARGET_URI> .
	  } GROUP BY ?uri
	}
"""

        for key, rscope in self.related_list_scopes[scope][qtype].items():
            fields = key.split("-")
            # print(f"    {key} -> {rscope}")
            if len(fields) == 2:
                aq = [flds[scope][fields[0]]]
                bq = [flds[rscope][fields[1]]]
            else:
                # Construct p1 and p2 as property paths using the same logic as the JSON search builder
                aq = []
                target_scope = scope
                while target_scope not in ["item", "work", "set", "collection"]:
                    f = fields.pop(0)
                    p = flds[target_scope][f]
                    aq.append(p)
                    try:
                        target_scope = self.lux_config.lux_config["terms"][target_scope][f]["relation"]
                    except KeyError:
                        print(f"KeyError: {f} not found in {target_scope}")
                        continue

                bq = []
                for f in fields:
                    try:
                        p2 = flds[target_scope][f]
                    except KeyError:
                        print(f"KeyError: {f} not found in {target_scope}")
                        continue
                    bq.append(p2)
                    target_scope = self.lux_config.lux_config["terms"][target_scope][f]["relation"]

            p = "/".join([f"^lux:{x[1:]}" if x[0] == "^" else f"lux:{x}" for x in aq])
            p2 = "/".join([f"^lux:{x[1:]}" if x[0] == "^" else f"lux:{x}" for x in bq])

            kn = key.replace("-", "_")
            names.append(kn)
            tmpl = SUB_TEMPLATE.replace("V_NAME_REL", kn).replace("V_URI_REL", p).replace("V_TARGET_REL", p2)
            fragments.append(tmpl)

        coalesces = " + ".join([f"COALESCE(?{x}, 0)" for x in names])
        vars = " ".join([f"?{x}" for x in names])

        q = f"""
{PREFIXES}
SELECT ?uri ?total {vars} WHERE {{
    {"\n".join(fragments)}
    FILTER(!(?uri = <V_TARGET_URI>))
    BIND({coalesces} AS ?total)
}} ORDER BY DESC(?total) LIMIT 50"""
        return q

    def cache_sparql_queries(self):
        for hal, entry in self.hal_queries.items():
            scope = entry["scope"]
            qname = entry["queryName"]
            if qname == "-" and "related-list" in entry["template"]:
                # related lists get generated differently
                self.sparql_hal_queries[scope][hal] = entry
                continue
            if scope not in self.sparql_hal_queries:
                self.sparql_hal_queries[scope] = {}

            query = self.queries.get(qname, {})
            if not query:
                print(f"Could not find query {qname} referenced from HAL {hal}")
                continue
            try:
                qscope = query["_scope"]
            except KeyError:
                # Already been processed
                print(f"Couldn't find scope for {hal} / {qname}?")
            try:
                parsed = self.json_reader.read(query, qscope)
            except Exception as e:
                print(f"Error parsing query for {hal}: {e}\n{query}")
                continue
            spq = self.sparql_translator.translate_search_count(parsed, qscope)
            self.sparql_hal_queries[scope][hal] = spq.get_text()

        for scope, entry in self.related_list_scopes.items():
            self.related_list_sparql[scope] = {}
            self.related_list_json[scope] = {}
            for qtype, queries in entry.items():
                # print(f"{scope}/{qtype}")
                spql = self.make_related_query_stub(scope, qtype)
                self.related_list_sparql[scope][qtype] = spql
                self.related_list_json[scope][qtype] = {}
                for qname, qscope in queries.items():
                    jq = self.make_related_json_stub(qname, scope, qscope)
                    self.related_list_json[scope][qtype][qname] = json.dumps(jq, separators=(",", ":"))


class scopeEnum(StrEnum):
    ITEM = "item"
    WORK = "work"
    AGENT = "agent"
    PLACE = "place"
    CONCEPT = "concept"
    SET = "set"
    EVENT = "event"


class classEnum(StrEnum):
    OBJECT = "object"
    DIGITAL = "digital"
    TEXT = "text"
    VISUAL = "visual"
    PLACE = "place"
    PERSON = "person"
    GROUP = "group"
    SET = "set"
    CONCEPT = "concept"
    EVENT = "event"
    PERIOD = "period"
    ACTIVITY = "activity"


class profileEnum(StrEnum):
    DEFAULT = ""
    NAME = "name"
    RESULTS = "results"


class SearchScopes(BaseModel):
    searchScopes: dict[str, int]


class StatisticsResponse(BaseModel):
    estimates: SearchScopes
