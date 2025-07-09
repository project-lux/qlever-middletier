from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import json
import uvicorn
import os
import copy
import aiohttp
import urllib

import psycopg2
from psycopg2.extras import RealDictCursor

from middletier_config import cfg, rdr, st, sorts, facets, args
from middletier_config import hal_link_templates, hal_queries, sparql_hal_queries
from middletier_config import related_list_names, related_list_queries, related_list_sparql
from middletier_config import MY_URI, SPARQL_ENDPOINT, PAGE_LENGTH, DATA_URI, PG_TABLE
from middletier_config import ENGLISH, PRIMARY, RESULTS_FIELDS, PORTAL_SOURCE

from boolean_query_parser import BooleanQueryParser

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


async def fetch_sparql(spq):
    if type(spq) is str:
        q = spq
    else:
        q = spq.get_text()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                SPARQL_ENDPOINT,
                data={"query": q},
                headers={"Accept": "application/sparql-results+json"},
            ) as response:
                ret = await response.json()
                if "results" in ret:
                    results = [r for r in ret["results"]["bindings"]]
                else:
                    # print("Broke...")
                    # print(q)
                    # print(json.dumps(ret, indent=2))
                    results = []
    except Exception as e:
        print(q)
        print(e)
        results = []
    return results


@app.get("/api/advanced-search-config")
async def do_get_config():
    return JSONResponse(content=cfg.lux_config)


@app.get("/api/search/{scope}")
async def do_search(scope, q={}, page=1, pageLength=PAGE_LENGTH, sort=""):
    if scope == "multi":
        # Handle multi scope search for archiveSort
        #
        # objectOrSetMemberOfSet is just:
        # ?uri lux:itemMemberOfSet|lux:setMemberOfSet <set-uri>
        #
        # setCurrentHierarchyPage:
        # the other entries in that are part of the archive that I am part of
        #
        # ?what lux:itemMemberOfSet|lux:setMemberOfSet ?parent .
        # <set-uri> lux:setMemberOfSet ?parent .
        # ?parent lux:setClassification <https://lux.collections.yale.edu/data/concept/0c8e015e-8ead-43e7-ad8c-c06c08448019>

        return JSONResponse({})

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

    q = q.replace(MY_URI, DATA_URI)
    jq = json.loads(q)
    parsed = rdr.read(jq, scope)
    spq = st.translate_search(parsed, scope=scope, limit=pageLength, offset=offset, sort=pred, order=ascdesc)
    qt = spq.get_text()
    print(qt)
    res = await fetch_sparql(qt)

    spq2 = st.translate_search_count(parsed, scope=scope)
    qt2 = spq2.get_text()
    ttl_res = await fetch_sparql(qt2)
    try:
        ttl = ttl_res[0]["count"]["value"]
    except Exception:
        ttl = 0
    uq = urllib.parse.quote(q)

    js = {
        "@context": "https://linked.art/ns/v1/search.json",
        "id": f"{MY_URI}api/search/{scope}?q={uq}&page=1",
        "type": "OrderedCollectionPage",
        "partOf": {
            "id": f"{MY_URI}api/search-estimate/{scope}?q={uq}",
            "type": "OrderedCollection",
            "label": {"en": ["Search Results"]},
            "summary": {"en": ["Description of Search Results"]},
            "totalItems": ttl,
        },
        "orderedItems": [],
    }
    # do next and prev

    for r in res:
        js["orderedItems"].append(
            {
                "id": r["uri"]["value"].replace(f"{DATA_URI}data/", f"{MY_URI}data/"),
                "type": "Object",
            }
        )
    return JSONResponse(content=js)


@app.get("/api/search-estimate/{scope}")
async def do_search_estimate(scope, q={}, page=1):
    q = q.replace(MY_URI, DATA_URI)
    jq = json.loads(q)
    uq = urllib.parse.quote(q)
    js = {
        "@context": "https://linked.art/ns/v1/search.json",
        "id": f"{MY_URI}api/search/{scope}?q={uq}",
        "type": "OrderedCollection",
        "label": {"en": ["Search Results"]},
        "summary": {"en": ["Description of Search Results"]},
        "totalItems": 0,
    }
    try:
        parsed = rdr.read(jq, scope)
    except ValueError as e:
        return JSONResponse(content=js)
    spq2 = st.translate_search_count(parsed, scope=scope)
    qt2 = spq2.get_text()
    ttl_res = await fetch_sparql(qt2)
    ttl = ttl_res[0]["count"]["value"]
    js["totalItems"] = int(ttl)
    return JSONResponse(content=js)


@app.get("/api/search-will-match")
async def do_search_match(q={}):
    scope = q["_scope"]
    del q["_scope"]

    q = q.replace(MY_URI, DATA_URI)
    jq = json.loads(q)
    parsed = rdr.read(jq, scope)
    spq2 = st.translate_search_count(parsed, scope=scope)
    qt2 = spq2.get_text()
    ttl_res = await fetch_sparql(qt2)
    ttl = ttl_res[0]["count"]["value"]
    js = {
        "unnamed": {
            "hasOneOrMoreResult": 1 if ttl > 0 else 0,
            "isRelatedList": False,
        }
    }
    return JSONResponse(content=js)


@app.get("/api/facets/{scope}")
async def do_facet(scope, q={}, name="", page=1):
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
    # print(f"{name} {pname} {pname2} {pred}")

    spq = st.translate_facet(parsed, pred)
    res = await fetch_sparql(spq)
    if res:
        spq2 = st.translate_facet_count(parsed, pred)
        res2 = await fetch_sparql(spq2)
        if res2 and "count" in res2[0]:
            ttl = int(res2[0]["count"]["value"])
            js["partOf"]["totalItems"] = ttl
        else:
            ttl = 0

    for r in res:
        # Need to know type of facet (per datatype below)
        # and what query to AND based on the predicate
        # e.g:
        # AND: [(query), {"rel": {"id": "val"}}]

        if r["facet"]["type"] == "uri":
            clause = {pname2: {"id": r["facet"]["value"]}}
            if pred == "a":
                val = r["facet"]["value"]
                if val.startswith(DATA_URI):
                    continue
                else:
                    val = val.replace("https://linked.art/ns/terms/", "")
            else:
                val = (
                    r["facet"]["value"]
                    .replace(f"{DATA_URI}data/", f"{MY_URI}data/")
                    .replace("https://lux.collections.yale.edu/ns/", "")
                    .replace("https://linked.art/ns/terms/", "")
                )

        elif r["facet"]["datatype"].endswith("int") or r["facet"]["datatype"].endswith("decimal"):
            val = int(r["facet"]["value"])
            clause = {pname2: val}
        elif r["facet"]["datatype"].endswith("float"):
            val = float(r["facet"]["value"])
            clause = {pname2: val}

        elif r["facet"]["datatype"].endswith("dateTime"):
            val = r["facet"]["value"]
            clause = {pname2: val}
        else:
            raise ValueError(r)

        nq = {"AND": [clause, jq]}
        qstr = urllib.parse.quote(json.dumps(nq, separators=(",", ":")))
        js["orderedItems"].append(
            {
                "id": f"{MY_URI}api/search-estimate/{scope}?q={qstr}",
                "type": "OrderedCollection",
                "value": val,
                "totalItems": int(r["facetCount"]["value"]),
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
    # scope is the type of records to find
    # name gives related list type (relatedToAgent)
    # uri is the anchoring entity

    all_res = {}
    cts = {}
    for name, spq in related_list_sparql[scope].items():
        qry = spq.replace("URI-HERE", uri)
        res = await fetch_sparql(qry)
        for row in res:
            what = row["uri"]["value"]
            ct = int(row["count"]["value"])
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


@app.get("/api/translate/{scope}")
async def do_translate(scope, q={}):
    # take simple search in text and return json query equivalent

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
        res = await fetch_sparql(qt)
        ttl = int(res[0]["count"]["value"])
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


@app.get("/data/{scope}/{identifier}")
async def do_get_record(scope, identifier, profile=None):
    # Check postgres cache
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    qry = f"SELECT * FROM {PG_TABLE} WHERE identifier = %s"
    params = (identifier,)
    cursor.execute(qry, params)
    row = cursor.fetchone()
    if row:
        js = row["data"]

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


@app.get("/api/stats")
async def do_stats():
    """Fetch counts of each class"""
    spq = "SELECT ?class (COUNT(?class) as ?count) {?what a ?class}  GROUP  BY  ?class"
    res = await fetch_sparql(spq)
    vals = {}
    for r in res:
        vals[r["class"]["value"].rsplit("/")[-1].lower()] = int(r["count"]["value"])
    cts = {}
    for s in cfg.scopes:
        cts[s] = vals[s]
    js = {"estimates": {"searchScopes": cts}}
    return JSONResponse(content=js)


geo_search = """
PREFIX lux: <https://lux.collections.yale.edu/ns/>
PREFIX ogc: <http://www.opengis.net/rdf#>
PREFIX osmrel: <https://www.openstreetmap.org/relation/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX qlss: <https://qlever.cs.uni-freiburg.de/spatialSearch/>

SELECT ?where ?coords WHERE {
  BIND( "POINT(174.763336 -36.848461)"^^geo:wktLiteral AS ?akl )

  SERVICE qlss: {
    _:config  qlss:algorithm qlss:s2 ;
              qlss:left ?akl ;
              qlss:right ?coords ;
              qlss:numNearestNeighbors 20 ;
              qlss:maxDistance 5000 ;
              qlss:bindDistance ?dist_left_right ;
              qlss:payload ?where  .
    {
      ?where lux:placeDefinedBy ?coords .
    }
  }
}
"""

geo_search2 = """
PREFIX lux: <https://lux.collections.yale.edu/ns/>
PREFIX ogc: <http://www.opengis.net/rdf#>
PREFIX osmrel: <https://www.openstreetmap.org/relation/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX qlss: <https://qlever.cs.uni-freiburg.de/spatialSearch/>
SELECT ?where ?centroid WHERE {
  BIND( "POINT (-1.5 53.608273)"^^geo:wktLiteral AS ?akl )
  SERVICE qlss: {
    _:config qlss:algorithm qlss:s2 ;
              qlss:left ?akl ;
              qlss:right ?centroid ;
              qlss:numNearestNeighbors 10000 ;
              qlss:maxDistance 70000 ;
              qlss:bindDistance ?dist_left_right ;
              qlss:payload ?where, ?coords .
    {
      # Any subquery, that selects ?right_geometry, ?payloadA and ?payloadB
?where lux:placeDefinedBy ?coords .
BIND(geof:centroid(?coords) AS ?centroid)

    }
  }
}
"""


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.loglevel)
