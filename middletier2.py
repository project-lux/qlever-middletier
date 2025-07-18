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
from fastapi_mcp import FastApiMCP

from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from uuid import UUID

import asyncio
import uvloop
from hypercorn.config import Config as HyperConfig
from hypercorn.asyncio import serve as hypercorn_serve

from qleverlux.middletier_config import cfg, rdr, st, sorts, facets, args
from qleverlux.middletier_config import hal_link_templates, hal_queries, sparql_hal_queries
from qleverlux.middletier_config import related_list_names, related_list_queries, related_list_sparql
from qleverlux.middletier_config import MY_URI, SPARQL_ENDPOINT, PAGE_LENGTH, DATA_URI, PG_TABLE
from qleverlux.middletier_config import ENGLISH, PRIMARY, RESULTS_FIELDS, PORTAL_SOURCE

from qleverlux.boolean_query_parser import BooleanQueryParser


conn = psycopg2.connect(user=args.user, dbname=args.db)

### To do
#
# * run the multi scope set queries (sparql in comments below)
# * figure out how to create the related list per-entry queries
# * Add a "no" option for hasDigitalImage
# * get "quoted string" for anywhere to match in ref name
#

### Extensions
#
# * Allow fields and relationships in the text representation (not done)
# * Allow variables in the queries (done)
#


class scopeEnum(str, Enum):
    ITEM = "item"
    WORK = "work"
    AGENT = "agent"
    PLACE = "place"
    CONCEPT = "concept"
    SET = "set"
    EVENT = "event"


class classEnum(str, Enum):
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


class profileEnum(str, Enum):
    DEFAULT = None
    NAME = "name"
    RESULTS = "results"


class orderEnum(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_parser = BooleanQueryParser()

if not os.path.exists("hal_cache"):
    os.makedirs("hal_cache")


@alru_cache(maxsize=500)
async def fetch_qlever_sparql(q):
    results = {"total": 0, "results": []}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                SPARQL_ENDPOINT,
                data={"query": q, "send": 60},
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
    except Exception:
        # fall back to trying to parse simple text query
        qp = query_parser.parse(q)
        qjs = qp.to_json()
        k = list(qjs.keys())[0]
        jq = {"_scope": scope}
        jq[k] = qjs[k]
    parsed = rdr.read(jq, scope)
    spq = st.translate_search(parsed, scope=scope, offset=soffset, sort=sort, order=order)
    qt = spq.get_text()
    return qt


def make_simple_reference(identifier):
    if identifier.startswith("http"):
        identifier = identifier.rsplit("/", 1)[-1]
    rec = fetch_record_from_cache(identifier)
    if not rec:
        return None
    outrec = {}
    outrec["id"] = identifier
    outrec["type"] = rec["type"]
    outrec["name"] = get_primary_name(rec["identified_by"])["content"]
    return outrec, rec


def make_simple_record(uri):
    try:
        outrec, rec = make_simple_reference(uri)
    except Exception:
        return None
    if "classified_as" in rec:
        outrec["classifications"] = []
        for cxn in rec["classified_as"]:
            if "id" in cxn:
                try:
                    outrec["classifications"].append(make_simple_reference(cxn["id"])[0])
                except Exception:
                    continue
    if "referred_to_by" in rec:
        outrec["descriptions"] = []
        for stmt in rec["referred_to_by"]:
            if "language" in stmt:
                langs = [x.get("equivalent", [{"id": None}])[0]["id"] for x in stmt.get("language", [])]
                if ENGLISH not in langs:
                    continue
            desc = {"content": stmt["content"]}
            if "classified_as" in stmt:
                desc["classifications"] = []
                for cxn in stmt["classified_as"]:
                    if "id" in cxn:
                        try:
                            desc["classifications"].append(make_simple_reference(cxn["id"])[0])
                        except Exception:
                            continue
            outrec["descriptions"].append(desc)
    if "part_of" in rec:
        outrec["part_of"] = []
        for parent in rec["part_of"]:
            try:
                outrec["part_of"].append(make_simple_reference(parent["id"])[0])
            except Exception:
                continue
    elif "broader" in rec:
        outrec["part_of"] = []
        for parent in rec["broader"]:
            try:
                outrec["part_of"].append(make_simple_reference(parent["id"])[0])
            except Exception:
                continue
    if rec["type"] == "Person":
        if "born" in rec:
            # split into birthDate and birthPlace
            if "timespan" in rec["born"]:
                if "begin_of_the_begin" in rec["born"]["timespan"]:
                    outrec["birthDate"] = rec["born"]["timespan"]["begin_of_the_begin"]
            if "took_place_at" in rec["born"]:
                outrec["birthPlace"] = make_simple_reference(rec["born"]["took_place_at"][0]["id"])[0]
        if "died" in rec:
            # split into deathDate and deathPlace
            if "timespan" in rec["died"]:
                if "begin_of_the_begin" in rec["died"]["timespan"]:
                    outrec["deathDate"] = rec["died"]["timespan"]["begin_of_the_begin"]
            if "took_place_at" in rec["died"]:
                outrec["deathPlace"] = make_simple_reference(rec["died"]["took_place_at"][0]["id"])[0]
    elif rec["type"] == "Group":
        if "formed_by" in rec:
            # split into birthDate and birthPlace
            if "timespan" in rec["formed_by"]:
                if "begin_of_the_begin" in rec["formed_by"]["timespan"]:
                    outrec["foundingDate"] = rec["formed_by"]["timespan"]["begin_of_the_begin"]
            if "took_place_at" in rec["formed_by"]:
                outrec["foundingPlace"] = make_simple_reference(rec["formed_by"]["took_place_at"][0]["id"])[0]
            if "carried_out_by" in rec["formed_by"]:
                outrec["founder"] = [make_simple_reference(x["id"])[0] for x in rec["formed_by"]["carried_out_by"]]

        if "dissolved_by" in rec:
            # split into deathDate and deathPlace
            if "timespan" in rec["dissolved_by"]:
                if "begin_of_the_begin" in rec["dissolved_by"]["timespan"]:
                    outrec["dissolutionDate"] = rec["dissolved_by"]["timespan"]["begin_of_the_begin"]
            if "took_place_at" in rec["dissolved_by"]:
                outrec["dissolutionPlace"] = make_simple_reference(rec["dissolved_by"]["took_place_at"][0]["id"])[0]
            if "carried_out_by" in rec["dissolved_by"]:
                outrec["dissolver"] = make_simple_reference(rec["dissolved_by"]["carried_out_by"][0]["id"])[0]
    elif rec["type"] == "HumanMadeObject":
        # produced_by
        # encountered_by
        # made_of
        # carries/shows -- embed this
        # ignore: dimensions, current_owner etc
        pass
    elif rec["type"] in ["LinguisticObject", "VisualItem"]:
        # about, etc
        # embed the HMO somehow? Would require a search...
        pass

    if "member_of" in rec:
        outrec["member_of"] = []
        for parent in rec["member_of"]:
            try:
                outrec["memberOf"].append(make_simple_reference(parent["id"])[0])
            except Exception:
                continue
    return outrec


@app.get("/api/basic/{scope}", operation_id="search_by_name")
async def do_basic_name_search(scope: scopeEnum, name: str):
    """
    Search for the top 20 entities in the given scope by their exact name.
    The `id` fields of references within the records can be used with the get_by_id tool to retrieve their full records.

    Parameters:
        - name (str): The name of the entity to search for

    Returns:
        - List[Entity]: A list of entities matching the name.
    """
    name = name.lower()
    q = {"_scope": scope.value, "name": f'"{name}"', "_complete": True}
    qt = make_sparql_query(scope.value, json.dumps(q))
    res = await fetch_qlever_sparql(qt)

    recs = []
    for r in res["results"][:20]:
        uri = r[0]
        outrec = make_simple_record(uri)
        if outrec is not None:
            recs.append(outrec)

    return recs


@app.get("/api/basic/get/{identifier}", operation_id="get_by_id")
async def do_basic_fetch(identifier: UUID):
    """
    Fetch a single entity by its identifier from `id` within a record

    Parameters:
        - identifier (UUID): The identifier of the entity to fetch

    Returns:
        - Entity: The description of the entity.
    """
    identifier = str(identifier)
    outrec = make_simple_record(identifier)
    return outrec


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
    return JSONResponse(content=js)


@app.get("/api/search-will-match")
async def do_search_match(q={}):
    scope = q["_scope"]
    del q["_scope"]
    qt = make_sparql_query(scope, q)
    res = await fetch_qlever_sparql(qt)
    js = {
        "unnamed": {
            "hasOneOrMoreResult": 1 if res["total"] > 0 else 0,
            "isRelatedList": False,
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
    await asyncio.sleep(0.1)
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
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    qry = f"SELECT * FROM {PG_TABLE} WHERE identifier = %s"
    params = (identifier,)
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

    spq = "SELECT ?class (COUNT(?class) as ?count) {?what a ?class} GROUP BY ?class"
    res = await fetch_qlever_sparql(spq)
    vals = {}
    for r in res["results"]:
        vals[r[0].rsplit("/")[-1].lower()] = r[1]
    cts = {}
    for s in cfg.scopes:
        cts[s] = vals[s]
    js = {"estimates": {"searchScopes": cts}}
    return JSONResponse(content=js)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting hypercorn https/2 server...")
    uvloop.install()
    hconfig = HyperConfig()
    hconfig.bind = [f"0.0.0.0:{args.port}"]
    hconfig.loglevel = args.loglevel
    hconfig.accesslog = "-"
    hconfig.errorlog = "-"
    hconfig.certfile = f"files/{args.cert}.pem"
    hconfig.keyfile = f"files/{args.cert}-key.pem"
    mcp = FastApiMCP(
        app,
        name="LUX MCP Server",
        describe_all_responses=False,
        describe_full_response_schema=False,
        include_operations=[
            "get_statistics",
            "get_record",
            "translate_string_query",
            "search",
            "facet",
            "search_by_name",
            "get_by_id",
        ],
    )
    mcp.mount()
    asyncio.run(hypercorn_serve(app, hconfig))
