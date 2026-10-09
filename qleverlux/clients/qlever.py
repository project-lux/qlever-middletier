"""The QLever SPARQL client.

One class wrapping both HTTP backends. httpx (HTTP/2) is the default; aiohttp
is kept because it was the original and is useful for comparison. Callers only
see ``await client.query(text)`` and never learn which one is in use.

Two behaviours worth knowing about:

* Every query is memoized with ``alru_cache`` keyed on the query text. This is
  why pagination is issued at offsets rounded down to a multiple of 60 - nearby
  pages then share a cache entry and the exact page is sliced out in Python.
* Requests are shed once more than ``max_qlever_requests`` are in flight,
  returning a 504-ish stub rather than queueing. HAL generation passes
  ``sheddable=False`` so it cannot be dropped halfway through building a
  record's links.
"""

from __future__ import annotations

import aiohttp
import httpx
from async_lru import alru_cache


def process_qlever_results(ret):
    """Unwrap ``application/qlever-results+json`` into a plain result dict.

    ``<uri>`` brackets are stripped and typed literals coerced to int/float, so
    the rest of the code never sees SPARQL syntax.
    """
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


#: Returned instead of querying when too many requests are already in flight.
SHED_RESPONSE = {
    "total": -1,
    "results": [],
    "error": "Too many open requests",
    "status": 504,
}


class QLeverClient:
    """Talks SPARQL to QLever over httpx or aiohttp."""

    def __init__(self, settings):
        self.settings = settings
        self.endpoint = settings.sparql_endpoint
        self.use_httpx = settings.use_httpx
        self.max_open = settings.max_qlever_requests
        self.open_requests = 0
        self.client = None

    def connect(self):
        """Open the connection pool. Must run inside the event loop."""
        if self.client is not None:
            return
        settings = self.settings
        print(f"Connecting to QLever: {self.endpoint}")
        if self.use_httpx:
            limits = httpx.Limits(max_connections=settings.max_qlever_connections)
            timeout = httpx.Timeout(
                settings.qlever_timeout,
                connect=2,
                read=settings.qlever_timeout - 1,
            )
            self.client = httpx.AsyncClient(
                http2=True, verify=False, timeout=timeout, limits=limits
            )
        else:
            timeout = aiohttp.ClientTimeout(
                connect=2,
                total=settings.qlever_timeout,
                sock_read=settings.qlever_timeout - 1,
            )
            self.client = aiohttp.ClientSession(timeout=timeout)

    async def aclose(self):
        if self.client is None:
            return
        if self.use_httpx:
            await self.client.aclose()
        else:
            await self.client.close()
        self.client = None

    def query(self, q, sheddable=True):
        """Run a SPARQL query. Returns a coroutine; results are cached."""
        if self.client is None:
            self.connect()
        if self.use_httpx:
            return self._query_httpx(q, sheddable)
        return self._query_aiohttp(q, sheddable)

    def _failure(self, q, e, response=None):
        print("--- Qlever Exception ---")
        print(q)
        print(e)
        self.open_requests -= 1
        if response is not None:
            return {
                "total": 0,
                "results": [],
                "error": str(e),
                "status": response.status_code,
            }
        return {"total": 0, "results": [], "error": str(e), "status": 0}

    @alru_cache(maxsize=500)
    async def _query_aiohttp(self, q, sheddable=True):
        response = None
        if sheddable and self.open_requests > self.max_open:
            return dict(SHED_RESPONSE)
        try:
            self.open_requests += 1
            async with self.client.post(
                self.endpoint,
                data={"query": q, "send": 60},
                headers={"Accept": "application/qlever-results+json"},
            ) as response:
                ret = await response.json()
                self.open_requests -= 1
                return process_qlever_results(ret)
        except Exception as e:
            return self._failure(q, e, response)

    @alru_cache(maxsize=500)
    async def _query_httpx(self, q, sheddable=True):
        response = None
        if sheddable and self.open_requests > self.max_open:
            return dict(SHED_RESPONSE)
        if self.client is None:
            self.connect()
        try:
            self.open_requests += 1
            response = await self.client.post(
                self.endpoint,
                data={"query": q, "send": 60},
                headers={"Accept": "application/qlever-results+json"},
            )
            ret = response.json()
            self.open_requests -= 1
            return process_qlever_results(ret)
        except Exception as e:
            return self._failure(q, e, response)
