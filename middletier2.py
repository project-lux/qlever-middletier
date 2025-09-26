import os
import copy
import json

import aiohttp
import urllib
import psycopg2
from psycopg2.extras import RealDictCursor

from async_lru import alru_cache

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from enum import StrEnum
from uuid import UUID

import asyncio
import uvloop
from hypercorn.config import Config as HyperConfig
from hypercorn.asyncio import serve as hypercorn_serve

from qleverlux.middletier_config import cfg, rdr, st, sorts, facets, args
from qleverlux.middletier_config import hal_link_templates, hal_queries, sparql_hal_queries
from qleverlux.middletier_config import related_list_names, related_list_queries, related_list_sparql
from qleverlux.middletier_config import MY_URI, SPARQL_ENDPOINT, PAGE_LENGTH, DATA_URI, PG_TABLE
from qleverlux.middletier_config import ENGLISH, PRIMARY, RESULTS_FIELDS, FACET_DELAY

from qleverlux.bool_query_parser2 import BooleanQueryParser

# FIXME: This should go to config (obviously)
# st.portal = "YPM"

query_parser = BooleanQueryParser()
# Create a connection to the PostgreSQL database
conn = psycopg2.connect(user=args.user, dbname=args.db)

if not os.path.exists("hal_cache"):
    os.makedirs("hal_cache")

# FIXME: Should be in config
SPARQL_RESPONSE_COUNT = 20


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


class orderEnum(StrEnum):
    ASC = "ASC"
    DESC = "DESC"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@alru_cache(maxsize=500)
async def fetch_qlever_sparql(q):
    results = {"total": 0, "results": []}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                SPARQL_ENDPOINT,
                data={"query": q, "send": SPARQL_RESPONSE_COUNT},
                headers={"Accept": "application/qlever-results+json"},
            ) as response:
                ret = await response.json()
                results["total"] = ret.get("resultSizeTotal", 0)
                results["time"] = ret.get("time", {}).get("total", "unknown")
                results["variables"] = ret.get("selected", [])
                for r in ret["res"]:
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
    except Exception as e:
        print(q)
        print(e)
        # raise
        results["error"] = str(e)
    return results


@app.get("/api/advanced-search-config", operation_id="get_advanced_search_config")
async def do_get_config():
    return JSONResponse(content=cfg.lux_config)


def build_multi_query(jq):
    if "OR" in jq:
        if "memberOf" in jq["OR"][0] and "memberOf" in jq["OR"][1]:
            test_uri = jq["OR"][0]["memberOf"]["id"]
            qt = f"""
PREFIX lux: <https://lux.collections.yale.edu/ns/>\nSELECT ?uri WHERE {{
?uri lux:itemMemberOfSet|lux:setMemberOfSet <{test_uri}> .
?uri lux:setSortIdentifier|lux:itemSortIdentifier ?sortId .
FILTER(STRLEN(?sortId) >= 8)
FILTER(!CONTAINS(?sortId, " "))
}}
ORDER BY ASC(?sortId)
LIMIT {PAGE_LENGTH}"""
            return qt


def make_sparql_query(scope, q, page=1, pageLength=PAGE_LENGTH, sort="relevance", order="DESC"):
    offset = (page - 1) * pageLength
    soffset = (offset // 60) * 60
    q = q.replace(MY_URI, DATA_URI)
    try:
        jq = json.loads(q)
        assert type(jq) is dict
    except Exception:
        # fall back to trying to parse simple text query
        qp = query_parser.parse(q)
        qjs = qp.to_json()
        k = list(qjs.keys())[0]
        jq = {"_scope": scope}
        jq[k] = qjs[k]
    parsed = rdr.read(jq, scope)
    try:
        spq = st.translate_search(parsed, scope=scope, offset=soffset, sort=sort, order=order)
    except Exception as e:
        print(f"Error translating search: {e}")
        return None
    qt = spq.get_text()
    return qt


@app.get("/api/search/{scope}", operation_id="search")
async def do_search(
    scope: scopeEnum,
    q: str,
    page: int = 1,
    pageLength: int = PAGE_LENGTH,
    sort: str = "relevance",
    order: orderEnum = "DESC",
):
    """
    Given a search query in the q parameter, perform the search against the database and return the results.

    Parameters:
        - scope (scopeEnum): The scope of the search.
        - q (url encoded dict): The search query.
        - page (int): The page number for pagination of results
        - pageLength (int): The number of results per page
        - sort (str): How to sort the results, default by relevance to the query
        - order (orderEnum): The direction of the sort, ASC or DESC

    Returns:
        - dict: The search results in the ActivityStreams CollectionPage format
    """

    scope = scope.value
    order = order.value
    page = int(page)
    pageLength = int(pageLength)
    offset = (page - 1) * pageLength
    sort = sort.strip()
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
    pred = sorts[scope].get(sort, "relevance")
    uq = urllib.parse.quote(q)

    if scope == "multi":
        # Just write the queries sensibly
        qt = ""
    else:
        qt = make_sparql_query(scope, q, page, pageLength, pred, ascdesc)

    res = await fetch_qlever_sparql(qt)

    js = {
        "@context": "https://linked.art/ns/v1/search.json",
        "id": f"{MY_URI}api/search/{scope}?q={uq}&page=1",
        "type": "OrderedCollectionPage",
        "partOf": {
            "id": f"{MY_URI}api/search-estimate/{scope}?q={uq}",
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
                "id": r[0].replace(f"{DATA_URI}data/", f"{MY_URI}data/"),
                "type": "Object",
            }
        )
    return JSONResponse(content=js)


@app.get("/api/search-estimate/{scope}")
async def do_search_estimate(scope, q={}, page=1):
    uq = urllib.parse.quote(q)
    js = {
        "@context": "https://linked.art/ns/v1/search.json",
        "id": f"{MY_URI}api/search/{scope}?q={uq}",
        "type": "OrderedCollection",
        "label": {"en": ["Search Results"]},
        "summary": {"en": ["Description of Search Results"]},
        "totalItems": 0,
    }
    qt = make_sparql_query(scope, q)
    res = await fetch_qlever_sparql(qt)
    js["totalItems"] = res["total"]
    js["_timing"] = res["time"]
    return JSONResponse(content=js)


@app.get("/api/search-will-match")
async def do_search_match(q={}):
    if type(q) is str:
        q = json.loads(q)
    scope = q["_scope"]
    del q["_scope"]
    q = json.dumps(q)
    qt = make_sparql_query(scope, q)
    res = await fetch_qlever_sparql(qt)
    js = {
        "unnamed": {
            "hasOneOrMoreResult": 1 if res["total"] > 0 else 0,
            "isRelatedList": False,
            "_timing": res["time"],
        }
    }
    return JSONResponse(content=js)


@app.get("/api/facets/{scope}", operation_id="facet")
async def do_facet(scope: scopeEnum, q: str, name: str, page: int = 1):
    """
    Retrieve facet values for a given facet name and query.

    Parameters:
        scope (scopeEnum): The scope of the search
        q (url encoded dict): The query
        name (str): The name of the facet
        page (int): The page number, defaults to 1

    Returns:
        - dict: The JSON response containing the facet values as an ActivityStream CollectionPage
    """
    if FACET_DELAY:
        await asyncio.sleep(FACET_DELAY / 1000)
    scope = scope.value
    offset = (int(page) - 1) * PAGE_LENGTH
    soffset = (offset // 60) * 60
    q = q.replace(MY_URI, DATA_URI)
    jq = json.loads(q)
    parsed = rdr.read(jq, scope)

    uq = urllib.parse.quote(q)
    js = {
        "@context": "https://linked.art/ns/v1/search.json",
        "id": f"{MY_URI}api/facets/{scope}?q={uq}&name={name}&page={page}",
        "type": "OrderedCollectionPage",
        "partOf": {"type": "OrderedCollection", "totalItems": 1000},
        "orderedItems": [],
    }

    pname = None
    pname2 = None
    if name.endswith("RecordType"):
        pred = "a"
    elif name.endswith("IsOnline"):
        return JSONResponse(js)
    elif name == "responsibleCollections":
        pred = "lux:itemMemberOfSet/lux:setCuratedBy"
    elif name == "responsibleUnits":
        pred = "lux:itemMemberOfSet/lux:setCuratedBy/lux:agentMemberOfGroup"
    else:
        pname = facets.get(name, None)
        pname2 = pname["searchTermName"]
        pred = st.get_predicate(pname2, scope)
        if pred == "lux:missed":
            pred = st.get_leaf_predicate(pname2, scope)
            if type(pred) is list:
                pred = pred[0]
            if pred == "missed":
                pred = pname2
    if ":" not in pred and pred != "a":
        pred = f"lux:{pred}"

    spq = st.translate_facet(parsed, pred, offset=soffset)
    qt = spq.get_text()
    res = await fetch_qlever_sparql(qt)
    js["partOf"]["totalItems"] = res["total"] + soffset
    js["_timing"] = res["time"]

    for r in res["results"][offset % 60 : offset % 60 + PAGE_LENGTH]:
        val = r[0]
        ct = r[1]
        if type(val) is str and val.startswith("http"):
            # is a URI
            if pred == "a":
                val = val.replace("https://linked.art/ns/terms/", "")
            else:
                val = (
                    val.replace(f"{DATA_URI}data/", f"{MY_URI}data/")
                    .replace("https://lux.collections.yale.edu/ns/", "")
                    .replace("https://linked.art/ns/terms/", "")
                )
            clause = {pname2: {"id": val}}
        else:
            clause = {pname2: val}

        nq = {"AND": [clause, jq]}
        qstr = urllib.parse.quote(json.dumps(nq, separators=(",", ":")))
        js["orderedItems"].append(
            {
                "id": f"{MY_URI}api/search-estimate/{scope}?q={qstr}",
                "type": "OrderedCollection",
                "value": val,
                "totalItems": ct,
            }
        )
    return JSONResponse(content=js)


@app.get("/api/related-list/{scope}")
async def do_related_list(scope, name, uri, page=1):
    """?name=relatedToAgent&uri=(uri-of-record)"""
    uuri = urllib.parse.quote(uri)
    js = {
        "@context": "https://linked.art/ns/v1/search.json",
        "id": f"{MY_URI}api/related-list/{scope}?name={name}&uri={uuri}&page={page}",
        "type": "OrderedCollectionPage",
        "orderedItems": [],
    }
    entry = {
        "id": f"{MY_URI}api/search-estimate/{scope}?q=QUERY-HERE",
        "type": "OrderedCollection",
        "totalItems": 0,
        "first": {
            "id": f"{MY_URI}api/search/{scope}?q=QUERY-HERE",
            "type": "OrderedCollectionPage",
        },
        "value": "",
        "name": "",
    }

    all_res = {}
    cts = {}
    for name, spq in related_list_sparql[scope].items():
        qry = spq.replace("URI-HERE", uri)
        res = await fetch_qlever_sparql(qry)
        _timing = res["time"]
        for row in res["results"]:
            what = row[0]
            ct = int(row[1])
            try:
                cts[what] += ct
            except KeyError:
                cts[what] = ct
            sqry = related_list_queries[scope][name].replace("URI-HERE", uri)
            try:
                all_res[what].append((name, ct, sqry))
            except Exception:
                all_res[what] = [(name, ct, sqry)]

    # FIXME: These queries aren't complete
    # https://lux.collections.yale.edu/api/related-list/concept?&name=relatedToAgent
    #       &uri=https%3A%2F%2Flux.collections.yale.edu%2Fdata%2Fperson%2F66049111-383e-4526-9632-2e9b6b6302dd
    # vs
    # http://localhost:5001/api/related-list/concept?name=relatedToAgent
    #       &uri=https%3A//lux.collections.yale.edu/data/person/66049111-383e-4526-9632-2e9b6b6302dd
    # Need to include `what` in the query, as per facets

    all_sort = sorted(cts, key=cts.get, reverse=True)
    for what in all_sort[:25]:
        es = sorted(all_res[what], key=lambda x: x[1], reverse=True)
        for rel, ct, sqry in es:
            usqry = urllib.parse.quote(sqry)
            e = copy.deepcopy(entry)
            e["id"] = e["id"].replace("QUERY-HERE", usqry)
            e["value"] = what.replace(DATA_URI, MY_URI)
            e["totalItems"] = ct
            e["name"] = related_list_names[rel]
            e["first"]["id"] = e["first"]["id"].replace("QUERY-HERE", usqry)
            js["orderedItems"].append(e)
    js["orderedItems"].sort(
        key=lambda x: cts[x["value"].replace(MY_URI, DATA_URI)],
        reverse=True,
    )
    return JSONResponse(content=js)


@app.get("/api/translate/{scope}", operation_id="translate_string_query")
async def do_translate(scope: scopeEnum, q: str):
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
        qp = query_parser.parse(q)
        # now translate AST into JSON query
        qjs = qp.to_json()
        k = list(qjs.keys())[0]
        js[k] = qjs[k]
    except Exception:
        js["AND"] = [{"text": q}]
    return JSONResponse(content=js)


async def do_hal_links(scope, identifier):
    if os.path.exists(f"hal_cache/{identifier}.json"):
        with open(f"hal_cache/{identifier}.json", "r") as f:
            links = json.load(f)
        return links

    uri = f"{DATA_URI}data/{scope}/{identifier}"
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
    uuri = urllib.parse.quote(uri)
    for hal, spq in sparql_hal_queries[hscope].items():
        if type(spq) is str:
            # related-list ... just add it
            href = hal_link_templates[hal].replace("{id}", uuri)
            links[hal] = {"href": href, "_estimate": 1}
            continue
        qt = spq.get_text()
        qt = qt.replace("URI-HERE", uri)
        res = await fetch_qlever_sparql(qt)
        ttl = res["results"][0][0]
        if ttl > 0:
            jq = hal_queries[hscope][hal]
            jqs = json.dumps(jq, separators=(",", ":"))
            jqs = jqs.replace("URI-HERE", uri)
            jqs = urllib.parse.quote(jqs)
            href = hal_link_templates[hal].replace("{q}", jqs)
            links[hal] = {"href": href, "_estimate": 1}

    with open(f"hal_cache/{identifier}.json", "w") as f:
        json.dump(links, f)
    return links


def get_primary_name(names):
    candidates = []
    for name in names:
        if name["type"] == "Name":
            langs = [x.get("equivalent", [{"id": None}])[0]["id"] for x in name.get("language", [])]
            cxns = [x.get("equivalent", [{"id": None}])[0]["id"] for x in name.get("classified_as", [])]
            if ENGLISH in langs and PRIMARY in cxns:
                return name
            elif PRIMARY in cxns:
                candidates.append(name)
    candidates.sort(key=lambda x: len(x.get("language", [])), reverse=True)
    return candidates[0] if candidates else None


def fetch_record_from_cache(identifier):
    global conn
    qry = f"SELECT * FROM {PG_TABLE} WHERE identifier = %s"
    params = (identifier,)
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(qry, params)
        row = cursor.fetchone()
    except:
        # try re-connecting to the database
        try:
            conn = psycopg2.connect(user=args.user, dbname=args.db)
        except psycopg2.OperationalError as e:
            print(f"Error connecting to database: {e}")
            return None
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(qry, params)
        row = cursor.fetchone()

    if row:
        return row["data"]
    else:
        return None


@app.get("/data/{scope}/{identifier}", operation_id="get_record")
async def do_get_record(scope: classEnum, identifier: UUID, profile: profileEnum = None):
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
    js = fetch_record_from_cache(identifier)
    if js:
        if not profile:
            links = {
                "curies": [
                    {"name": "lux", "href": f"{MY_URI}api/rels/{{rel}}", "templated": True},
                    {"name": "la", "href": "https://linked.art/api/1.0/rels/{{rel}}", "templated": True},
                ],
                "self": {"href": f"{MY_URI}data/{scope}/{identifier}"},
            }
            # Calculate _links here
            more_links = await do_hal_links(scope, identifier)
            links.update(more_links)
            jstr = json.dumps(js)
            jstr = jstr.replace(f"{DATA_URI}data/", f"{MY_URI}data/")
            js2 = json.loads(jstr)
            js2["_links"] = links
        else:
            js2 = {}
            js2["id"] = js["id"]
            js2["type"] = js["type"]
            js2["identified_by"] = [get_primary_name(js["identified_by"])]

            if profile == "results":
                for fld in RESULTS_FIELDS:
                    if fld in js:
                        js2[fld] = js[fld]
                for nm in js["identified_by"]:
                    if nm["type"] == "Identifier":
                        js2["identified_by"].append(nm)
            jstr = json.dumps(js2)
            jstr = jstr.replace(f"{DATA_URI}data/", f"{MY_URI}data/")
            js2 = json.loads(jstr)

        return JSONResponse(content=js2)
    else:
        return JSONResponse(content={}, status_code=404)


class SearchScopes(BaseModel):
    searchScopes: dict[str, int]


class StatisticsResponse(BaseModel):
    estimates: SearchScopes


@app.get("/api/stats", response_model=StatisticsResponse, operation_id="get_statistics")
async def do_stats():
    """
    Get counts of each class in the database.

    Returns a dictionary of estimates in estimates.searchScopes where the key is the class and the value is the count.
    """

    if st.portal is not None:
        spq = f"PREFIX lux: <https://lux.collections.yale.edu/ns/> SELECT ?class (COUNT(?class) as ?count) WHERE {{?what a ?class ; lux:source lux:{st.portal} . }} GROUP BY ?class"
    else:
        spq = "SELECT ?class (COUNT(?class) as ?count) {?what a ?class} GROUP BY ?class"
    # This will always be in the ALRU cache
    res = await fetch_qlever_sparql(spq)
    vals = {}
    for r in res["results"]:
        vals[r[0].rsplit("/")[-1].lower()] = r[1]
    cts = {}
    for s in cfg.scopes:
        cts[s] = vals.get(s, 0)
    js = {"estimates": {"searchScopes": cts}}
    return JSONResponse(content=js)


async def main():
    uvloop.install()
    hconfig = HyperConfig()
    hconfig.bind = [f"0.0.0.0:{args.port}"]
    hconfig.loglevel = args.loglevel
    hconfig.accesslog = "-"
    hconfig.errorlog = "-"
    hconfig.certfile = f"files/{args.cert}.pem"
    hconfig.keyfile = f"files/{args.cert}-key.pem"
    hconfig.queue_size = 200
    hconfig.backlog = 200
    hconfig.read_timeout = 120
    hconfig.max_app_queue_size = 50
    hconfig.worker_class = "asyncio"
    # hconfig.workers = 10
    await hypercorn_serve(app, hconfig)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hypercorn https/2 server...")
    asyncio.run(main())
