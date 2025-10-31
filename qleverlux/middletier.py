import os
import sys
import json
from uuid import UUID
import urllib

from psycopg import AsyncConnection
from psycopg.rows import dict_row

import aiohttp
import httpx

from async_lru import alru_cache
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from luxql.string_parser import QueryParser
from qleverlux.middletier_config import MTConfig
from qleverlux.middletier_config import scopeEnum, classEnum, profileEnum, StatisticsResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

local_module = sys.modules[__name__]


@app.get("/api/advanced-search-config", operation_id="get_config")
async def api_get_search_config():
    return await local_module.mt.do_get_config()


@app.get("/api/stats", response_model=StatisticsResponse, operation_id="get_statistics")
async def api_get_statistics():
    return await local_module.mt.do_stats()


@app.get("/data/{scope}/{identifier}", operation_id="get_record")
async def api_get_record(scope: classEnum, identifier: UUID, profile: profileEnum = None):
    return await local_module.mt.do_get_record(scope, identifier, profile)


@app.get("/api/translate/{scope}", operation_id="translate_string_query")
async def api_get_translate(scope: scopeEnum, q: str):
    return await local_module.mt.do_translate(scope, q)


@app.get("/api/related-list/{scope}", operation_id="get_related_list")
async def api_get_related_list(scope: scopeEnum, name: str, uri: str, page: int = 1):
    return await local_module.mt.do_related_list(scope, name, uri, page)


@app.get("/api/facets/{scope}", operation_id="get_facet")
async def api_get_facet(scope: scopeEnum, q: str, name: str, page: int = 1, sort: str = ""):
    return await local_module.mt.do_facet(scope, q, name, page, sort)


@app.get("/api/search-estimate/{scope}", operation_id="get_estimate")
async def api_get_search_estimate(scope: scopeEnum, q={}, page=1):
    return await local_module.mt.do_search_estimate(scope, q, page)


@app.get("/api/search/{scope}", operation_id="get_search")
async def api_get_search(scope: scopeEnum, q: str, page: int = 1, pageLength: int = 0, sort: str = "relevance:DESC"):
    return await local_module.mt.do_search(scope, q, page, pageLength, sort)


class QLeverLuxMiddleTier:
    def __init__(self):
        self.config = MTConfig()
        self.config.print_config()
        self.json_reader = self.config.json_reader
        self.sparql_translator = self.config.sparql_translator
        self.query_parser = QueryParser()
        self.sparql_client = None
        self.postgres_conn = None
        self.open_requests = 0
        self.warning_request_limit = 16

        if not os.path.exists("hal_cache"):
            os.makedirs("hal_cache")

    def start(self):
        self.connect_to_postgres()
        self.connect_to_qlever()

    def connect_to_qlever(self):
        # The async pool needs to exist before async clients can be created
        if self.config.use_httpx:
            self.sparql_client = httpx.AsyncClient(http2=True, verify=False, timeout=self.config.qlever_timeout)
        else:
            self.sparql_client = aiohttp.ClientSession()

    async def connect_to_postgres(self):
        try:
            if self.config.pghost:
                conninfo = f"host={self.config.pghost} port={self.config.pgport} user={self.config.pguser} password={self.config.pgpass} dbname={self.config.pgdb}"
                self.postgres_conn = await AsyncConnection.connect(conninfo)
            else:
                self.postgres_conn = await AsyncConnection.connect(
                    user=self.config.pguser,
                    dbname=self.config.pgdb,
                )
        except Exception as e:
            print(f"Error connecting to database: {e}")
        return None

    def process_qlever_results(self, ret):
        results = {"results": []}
        results["total"] = ret.get("resultSizeTotal", 0)
        results["time"] = ret.get("time", {}).get("total", "unknown")
        results["variables"] = ret.get("selected", [])
        if ret["status"] == "ERROR":
            results["error"] = ret["exception"]
        for r in ret.get("res", []):
            r2 = []
            for i in r:
                if i is None:
                    r2.append(None)
                elif i[0] == "<" and i[-1] == ">":
                    r2.append(i[1:-1])
                elif "^^<" in i and i[-1] == ">":
                    val, dt = i[:-1].rsplit("^^<", 1)
                    val = val[1:-1]
                    if dt.endswith("int"):
                        r2.append(int(val))
                    elif dt.endswith("decimal"):
                        r2.append(float(val))
                    else:
                        r2.append(val)
                else:
                    r2.append(i)
            results["results"].append(r2)
        return results

    def fetch_qlever_sparql(self, q):
        if self.sparql_client is None:
            self.connect_to_qlever()

        if self.open_requests > 16:
            print(f"!!! {self.open_requests} open requests")

        if self.config.use_httpx:
            return self.fetch_qlever_sparql_httpx(q)
        else:
            return self.fetch_qlever_sparql_aiohttp(q)

    @alru_cache(maxsize=500)
    async def fetch_qlever_sparql_aiohttp(self, q):
        try:
            self.open_requests += 1
            async with self.sparql_client.post(
                self.config.sparql_endpoint,
                data={"query": q, "send": 60},
                headers={"Accept": "application/qlever-results+json"},
            ) as response:
                ret = await response.json()
                self.open_requests -= 1
                return self.process_qlever_results(ret)
        except Exception as e:
            print("--- Qlever Exception ---")
            print(q)
            print(e)
            self.open_requests -= 1
            return {"total": 0, "results": [], "error": str(e), "status": response.status_code}

    @alru_cache(maxsize=500)
    async def fetch_qlever_sparql_httpx(self, q):
        try:
            self.open_requests += 1
            response = await self.sparql_client.post(
                self.config.sparql_endpoint,
                data={"query": q, "send": 60},
                headers={"Accept": "application/qlever-results+json"},
            )
            ret = response.json()
            self.open_requests -= 1
            return self.process_qlever_results(ret)
        except Exception as e:
            print("--- Qlever Exception ---")
            print(q)
            print(e)
            self.open_requests -= 1
            return {"total": 0, "results": [], "error": str(e), "status": response.status_code}

    def make_sparql_query(self, scope, q, page=1, pageLength=0, sort="relevance", order="DESC"):
        if pageLength < 1:
            pageLength = self.config.page_length
        offset = (page - 1) * pageLength
        soffset = (offset // 60) * 60
        q = q.replace(self.config.mt_uri, self.config.data_uri)
        try:
            jq = json.loads(q)
            assert type(jq) is dict
        except Exception:
            # fall back to trying to parse simple text query
            qp = self.query_parser.parse(q)
            qjs = qp.to_json()
            k = list(qjs.keys())[0]
            jq = {"_scope": scope}
            jq[k] = qjs[k]
        parsed = self.json_reader.read(jq, scope)
        try:
            spq = self.sparql_translator.translate_search(parsed, scope=scope, offset=soffset, sort=sort, order=order)
        except Exception as e:
            print(f"Error translating search: {e}")
            return None
        qt = spq.get_text()
        return qt

    async def do_hal_links(self, scope, identifier):
        if self.config.use_disk_hal_cache:
            fn = os.path.join(self.config.hal_cache_path, f"{identifier}.json")
            if os.path.exists(fn):
                with open(fn, "r") as f:
                    links = json.load(f)
                return links

        uri = f"{self.config.data_uri}data/{scope}/{identifier}"
        links = {}
        if scope in ["person", "group"]:
            hscope = "agent"
        elif scope in ["object", "digital"]:
            hscope = "item"
        elif scope in ["place", "set", "event", "concept"]:
            hscope = scope
        elif scope in ["period", "activity"]:
            hscope = "event"
        elif scope in ["text", "visual", "image"]:
            hscope = "work"
        else:
            print(f"MISSED SCOPE IN HAL: {scope}")
            hscope = scope

        # uuri = urllib.parse.quote(uri)
        for hal, qt in self.config.sparql_hal_queries[hscope].items():
            if type(qt) is dict:
                # related list
                scope = qt["scope"]
                rlname = qt["relatedList"]
                rtemplate = qt["template"]
                try:
                    qt = self.config.related_list_sparql[scope][rlname]
                except KeyError:
                    print(f"Missing related list for {scope} and {rlname}")
                    continue
                qt = qt.replace("V_TARGET_URI", uri)
                rtemplate = rtemplate.replace("{id}", uri)
            else:
                qt = qt.replace("URI-HERE", uri)
                rtemplate = None

            res = await self.fetch_qlever_sparql(qt)
            try:
                ttl = res["results"][0][0]
            except IndexError:
                ttl = 0
            if type(ttl) is not int:
                ttl = res["total"]
            if ttl > 0:
                if rtemplate is None:
                    info = self.config.hal_queries[hal]
                    template = info["template"]
                    qname = info["queryName"]
                    jq = self.config.queries[qname]
                    jqs = json.dumps(jq, separators=(",", ":"))
                    jqs = jqs.replace("URI-HERE", uri)
                    jqs = urllib.parse.quote(jqs)
                    href = template.replace("{q}", jqs)
                else:
                    href = rtemplate
                links[hal] = {"href": href, "_estimate": 1}

        # XXX FIXME: Don't write to file, but write back to a postgres table
        if self.config.use_disk_hal_cache:
            with open(fn, "w") as f:
                json.dump(links, f)
        elif self.config.use_postgres_hal_cache:
            await self.store_postgres_hal_cache(links)
        return links

    def get_primary_name(self, names):
        candidates = []
        for name in names:
            if name["type"] == "Name":
                langs = [x.get("equivalent", [{"id": None}])[0]["id"] for x in name.get("language", [])]
                cxns = [x.get("equivalent", [{"id": None}])[0]["id"] for x in name.get("classified_as", [])]
                if self.config.aat_english in langs and self.config.aat_primary in cxns:
                    return name
                elif self.config.aat_primary in cxns:
                    candidates.append(name)
        candidates.sort(key=lambda x: len(x.get("language", [])), reverse=True)
        return candidates[0] if candidates else None

    async def store_postgres_hal_cache(self, identifier, links):
        async with self.postgres_conn.cursor(row_factory=dict_row) as cursor:
            qry = f"INSERT INTO {self.config.pgtable_hal} (identifier, data) VALUES (%s, %s)"
            params = (identifier, links)
            await cursor.execute(qry, params)

    async def fetch_record_from_cache(self, identifier):
        if self.config.use_postgres_hal_cache:
            qry = f"SELECT doc.data, hal.data FROM {self.config.pgtable} AS doc LEFT JOIN {self.config.pgtable_hal} AS hal ON doc.identifier = hal.identifier WHERE doc.identifier = %s"
        else:
            qry = f"SELECT * FROM {self.config.pgtable} WHERE identifier = %s"
        params = (identifier,)
        row = None
        try:
            # this will fail at least the very first attempt before the connection is created
            async with self.postgres_conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(qry, params)
                row = await cursor.fetchone()
        except Exception:
            await self.connect_to_postgres()
            async with self.postgres_conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(qry, params)
                row = await cursor.fetchone()

        print(row)
        if row:
            return row["data"]
        else:
            return None

    # API Functions From Here

    async def do_get_config(self):
        # mt-config.luxql-config.lux-advsrch-config (!)
        return JSONResponse(content=self.config.lux_config.lux_config)

    async def do_search(
        self, scope: scopeEnum, q: str, page: int = 1, pageLength: int = 0, sort: str = "relevance:desc"
    ):
        """
        Given a search query in the q parameter, perform the search against the database and return the results.

        Parameters:
            - scope (scopeEnum): The scope of the search.
            - q (url encoded dict): The search query.
            - page (int): The page number for pagination of results
            - pageLength (int): The number of results per page
            - sort (str): How to sort the results, default by relevance to the query

        Returns:
            - dict: The search results in the ActivityStreams CollectionPage format
        """

        scope = scope.value
        page = int(page)
        pageLength = int(pageLength)
        offset = (page - 1) * pageLength
        sort = sort.strip()
        if pageLength < 1:
            pageLength = self.config.page_length
        if sort:
            try:
                sort, ascdesc = sort.split(":")
                ascdesc = ascdesc.upper().strip()
                sort = sort.strip()
            except Exception:
                ascdesc = "ASC"
        else:
            sort = "relevance"
            ascdesc = "DESC"
        pred = self.config.sorts[scope].get(sort, "relevance")
        uq = urllib.parse.quote(q)

        if scope == "multi":
            # Just write the queries sensibly
            qt = ""
        else:
            try:
                qt = self.make_sparql_query(scope, q, page, pageLength, pred, ascdesc)
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": str(e)})

        res = await self.fetch_qlever_sparql(qt)

        js = {
            "@context": "https://linked.art/ns/v1/search.json",
            "id": f"{self.config.mt_uri}api/search/{scope}?q={uq}&page=1",
            "type": "OrderedCollectionPage",
            "partOf": {
                "id": f"{self.config.mt_uri}api/search-estimate/{scope}?q={uq}",
                "type": "OrderedCollection",
                "label": {"en": ["Search Results"]},
                "summary": {"en": ["Description of Search Results"]},
                "totalItems": res["total"],
            },
            "orderedItems": [],
            "_timing": res["time"],
        }
        # FIXME: do next and prev

        for r in res["results"][offset % 60 : offset % 60 + pageLength]:
            js["orderedItems"].append(
                {
                    "id": r[0].replace(f"{self.config.data_uri}data/", f"{self.config.mt_uri}data/"),
                    "type": "Object",
                }
            )
        return JSONResponse(content=js)

    async def do_search_estimate(self, scope, q={}, page=1):
        uq = urllib.parse.quote(q)
        js = {
            "@context": "https://linked.art/ns/v1/search.json",
            "id": f"{self.config.mt_uri}api/search/{scope}?q={uq}",
            "type": "OrderedCollection",
            "label": {"en": ["Search Results"]},
            "summary": {"en": ["Description of Search Results"]},
            "totalItems": 0,
        }
        qt = self.make_sparql_query(scope, q)
        res = await self.fetch_qlever_sparql(qt)
        js["totalItems"] = res["total"]
        js["_timing"] = res["time"]
        return JSONResponse(content=js)

    async def do_facet(
        self, scope: scopeEnum, q: str, name: str, page: int = 1, sort: str = "", pageLength: int = 20
    ):
        """
        Retrieve facet values for a given facet name and query.

        Parameters:
            scope (scopeEnum): The scope of the search
            q (url encoded dict): The query
            name (str): The name of the facet
            page (int): The page number, defaults to 1
            sort (str): "asc" or "desc"
            pageLength (int): The number of items per page, defaults to configured value (20)

        Returns:
            - dict: The JSON response containing the facet values as an ActivityStream CollectionPage
        """
        if self.config.facet_delay:
            await asyncio.sleep(self.config.facet_delay / 1000)
        scope = scope.value

        offset = (int(page) - 1) * self.config.page_length
        soffset = (offset // 60) * 60
        sort = sort.strip()
        if sort:
            try:
                sort, ascdesc = sort.split(":")
                ascdesc = ascdesc.upper().strip()
                sort = sort.strip()
            except Exception:
                ascdesc = sort.upper().strip()
                sort = ""
        else:
            sort = ""
            ascdesc = ""

        q = q.replace(self.config.mt_uri, self.config.data_uri)
        jq = json.loads(q)
        parsed = self.json_reader.read(jq, scope)

        uq = urllib.parse.quote(q)
        js = {
            "@context": "https://linked.art/ns/v1/search.json",
            "id": f"{self.config.mt_uri}api/facets/{scope}?q={uq}&name={name}&page={page}&pageLength={self.config.page_length}",
            "type": "OrderedCollectionPage",
            "partOf": {"type": "OrderedCollection", "totalItems": 0},
            "orderedItems": [],
        }

        pname = None
        pname2 = None
        if name.endswith("RecordType"):
            pred = "a"
        elif name == "responsibleCollections":
            pred = "lux:itemMemberOfSet/lux:setCuratedBy"
        elif name == "responsibleUnits":
            pred = "lux:itemMemberOfSet/lux:setCuratedBy/lux:agentMemberOfGroup"
        else:
            pname = self.config.facets.get(name, None)
            if not pname:
                print(f" *** request for unknown facet {name} ***")
                pname2 = "MISSING"
            else:
                pname2 = pname["searchTermName"]
            pred = self.sparql_translator.get_predicate(pname2, scope)
            if pred == "lux:missed":
                pred = self.sparql_translator.get_leaf_predicate(pname2, scope)
                if type(pred) is list:
                    pred = pred[0]
                if pred == "missed":
                    pred = pname2
        if ":" not in pred and pred != "a":
            pred = f"lux:{pred}"

        spq = self.sparql_translator.translate_facet(
            parsed, pred, scope=scope, offset=soffset, sort=sort, order=ascdesc
        )
        qt = spq.get_text()
        # print(qt)
        res = await self.fetch_qlever_sparql(qt)
        js["partOf"]["totalItems"] = res["total"] + soffset
        js["_timing"] = res["time"]

        for r in res["results"][offset % 60 : offset % 60 + self.config.page_length]:
            val = r[0]
            ct = r[1]
            if type(val) is str and val.startswith("http"):
                # is a URI
                if pred == "a":
                    if val.startswith("https://lux.collections.yale.edu/ns/"):
                        continue
                    val = val.replace("https://linked.art/ns/terms/", "")
                else:
                    val = (
                        val.replace(f"{self.config.data_uri}data/", f"{self.config.mt_uri}data/")
                        .replace("https://lux.collections.yale.edu/ns/", "")
                        .replace("https://linked.art/ns/terms/", "")
                    )
                clause = {pname2: {"id": val}}
            else:
                clause = {pname2: val, "_comp": "=="}

            nq = {"AND": [clause, jq]}
            qstr = urllib.parse.quote(json.dumps(nq, separators=(",", ":")))
            js["orderedItems"].append(
                {
                    "id": f"{self.config.mt_uri}api/search-estimate/{scope}?q={qstr}",
                    "type": "OrderedCollection",
                    "value": val,
                    "totalItems": ct,
                }
            )
        if page > 1:
            js["prev"] = {
                "id": f"{self.config.mt_uri}api/facets/{scope}?q={uq}&name={name}&page={page - 1}&pageLength={self.config.page_length}",
                "type": "OrderedCollectionPage",
            }
        if offset + self.config.page_length < js["partOf"]["totalItems"]:
            js["next"] = {
                "id": f"{self.config.mt_uri}api/facets/{scope}?q={uq}&name={name}&page={page + 1}&pageLength={self.config.page_length}",
                "type": "OrderedCollectionPage",
            }

        return JSONResponse(content=js)

    async def do_related_list(self, scope: scopeEnum, name: str, uri: str, page: int = 1):
        """?name=relatedToAgent&uri=(uri-of-record)"""
        xuri = urllib.parse.quote(uri)
        js = {
            "@context": "https://linked.art/ns/v1/search.json",
            "id": f"{self.config.mt_uri}api/related-list/{scope}?name={name}&page={page}&uri={xuri}",
            "type": "OrderedCollectionPage",
            "orderedItems": [],
            "next": f"{self.config.mt_uri}api/related-list/{scope}?name={name}&page={page + 1}&uri={xuri}",
        }

        # get query from config's sparql cache and substitue in xuri
        spq = self.config.related_list_sparql[scope][name]
        spq = spq.replace("V_TARGET_URI", uri)
        # execute sparql query
        res = await self.fetch_qlever_sparql(spq)

        # make full response

        vars = [x[1:].replace("_", "-") for x in res["variables"]]
        for r in res["results"]:
            uri = r[0]
            if uri is None:
                print(res)
                continue
            luri = uri.replace(self.config.data_uri, self.config.mt_uri)
            rd = list(zip(vars[2:], [x if x else 0 for x in r][2:]))
            rd.sort(key=lambda x: x[1], reverse=True)
            for k, v in rd:
                if not v:
                    break
                label = self.config.related_list_names.get(k, f"UNKNOWN RELATED LIST: {k}")
                qscope = self.config.related_list_scopes[scope][name][k]

                # make json query string with target substitution
                qjstr = (
                    self.config.related_list_json[scope][name][k].replace("V_TO_URI", uri).replace("V_FROM_URI", xuri)
                )

                qjstr = urllib.parse.quote(qjstr)
                coll_id = f"{self.config.mt_uri}api/search-estimate/{qscope}?q={qjstr}"
                first_id = f"{self.config.mt_uri}api/search/{qscope}?page=1&q={qjstr}"
                entry = {
                    "id": coll_id,
                    "type": "OrderedCollection",
                    "totalItems": v,
                    "first": {"id": first_id, "type": "OrderedCollectionPage"},
                    "value": luri,
                    "name": label,
                }
                js["orderedItems"].append(entry)

        return JSONResponse(content=js)

    async def do_translate(self, scope: scopeEnum, q: str):
        """
        Translate a simple search query into a JSON query equivalent.

        Parameters:
            - scope (scopeEnum): The scope for the query
            - q (str): The simple search query

        Returns:
            - dict: The JSON query equivalent of the given query

        """
        # take simple search in text and return json query equivalent
        scope = scope.value
        js = {"_scope": scope}
        try:
            qp = self.query_parser.parse(q)
            # now translate AST into JSON query
            qjs = qp.to_json()
            k = list(qjs.keys())[0]
            js[k] = qjs[k]
        except Exception:
            js["AND"] = [{"text": q}]
        jqs = json.dumps(qjs, separators=(",", ":"))
        jqs = urllib.parse.quote(jqs)
        # js["_link"] = f"{self.config.mt_uri}api/search/{scope}?q={jqs}&page=1"
        return JSONResponse(content=js)

    async def do_get_record(self, scope: classEnum, identifier: UUID, profile: profileEnum = None):
        """
        Retrieve an individual record from the database.

        Parameters:
            - scope (str): The class of the record.
            - identifier (str): A UUID, the identifier of the record.
            - profile (str, optional): The profile of the record. Defaults to no profile, otherwise "name" of "results"

        Returns:
            - dict: The record.
        """
        # Check postgres cache
        scope = scope.value
        if profile is not None:
            profile = profile.value
        identifier = str(identifier)
        scope = str(scope)
        js = await self.fetch_record_from_cache(identifier)
        if js:
            if not profile:
                links = {
                    "curies": [
                        {"name": "lux", "href": f"{self.config.mt_uri}api/rels/{{rel}}", "templated": True},
                        {"name": "la", "href": "https://linked.art/api/1.0/rels/{rel}", "templated": True},
                    ],
                    "self": {"href": f"{self.config.mt_uri}data/{scope}/{identifier}"},
                }
                # Calculate _links here
                more_links = await self.do_hal_links(scope, identifier)
                links.update(more_links)
                jstr = json.dumps(js)
                jstr = jstr.replace(f"{self.config.data_uri}data/", f"{self.config.mt_uri}data/")
                js2 = json.loads(jstr)
                js2["_links"] = links
            else:
                js2 = {}
                js2["id"] = js["id"]
                js2["type"] = js["type"]
                js2["identified_by"] = [self.get_primary_name(js["identified_by"])]

                if profile == "results":
                    for fld in self.config.results_fields:
                        if fld in js:
                            js2[fld] = js[fld]
                    for nm in js["identified_by"]:
                        if nm["type"] == "Identifier":
                            js2["identified_by"].append(nm)
                jstr = json.dumps(js2)
                jstr = jstr.replace(f"{self.config.data_uri}data/", f"{self.config.mt_uri}data/")
                js2 = json.loads(jstr)

            return JSONResponse(content=js2)
        else:
            return JSONResponse(content={}, status_code=404)

    async def do_stats(self):
        """
        Get counts of each class in the database.

        Returns a dictionary of estimates in estimates.searchScopes where the key is the class and the value is the count.
        """

        if self.sparql_translator.portal is not None:
            spq = f"PREFIX lux: <https://lux.collections.yale.edu/ns/> SELECT ?class (COUNT(?class) as ?count) \
            WHERE {{?what a ?class ; lux:source lux:{self.sparql_translator.portal} . }} GROUP BY ?class"
        else:
            spq = "SELECT ?class (COUNT(?class) as ?count) {?what a ?class} GROUP BY ?class"
        # This will always be in the ALRU cache
        res = await self.fetch_qlever_sparql(spq)
        vals = {}
        for r in res["results"]:
            vals[r[0].rsplit("/")[-1].lower()] = r[1]
        cts = {}
        for s in self.config.lux_config.scopes:
            cts[s] = vals.get(s, 0)
        js = {"estimates": {"searchScopes": cts}}
        return JSONResponse(content=js)
