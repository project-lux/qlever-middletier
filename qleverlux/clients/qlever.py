"""The QLever SPARQL client.

One class wrapping both HTTP backends. httpx (HTTP/2) is the default; aiohttp
is kept because it was the original and is useful for comparison. Callers only
see ``await client.query(text)`` and never learn which one is in use.

Two behaviours worth knowing about:

* Every successful query is memoized with ``alru_cache`` keyed on the query
  text. This is why pagination is issued at offsets rounded down to a multiple
  of 60 - nearby pages then share a cache entry and the exact page is sliced
  out in Python. Failures (timeouts, QLever errors, shed requests) are never
  cached: one slow first attempt must not fail that query until it is evicted.
* Requests are shed once more than ``max_qlever_requests`` are in flight,
  returning a 504-ish stub rather than queueing. ``sheddable=False`` exempts a
  call; nothing in the middle tier uses it now (HAL generation once did, and
  so bypassed the limit for every uncached record).
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


class _QueryFailed(Exception):
    """Carries a failure result out of the cached call, so it is not cached."""

    def __init__(self, result):
        super().__init__(result.get("error"))
        self.result = result


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
        """Run a SPARQL query. Returns a coroutine; successful results are cached."""
        if self.client is None:
            self.connect()
        return self._query(q, sheddable)

    async def _query(self, q, sheddable):
        # Shedding is decided per call, outside the cache: a query shed once
        # under load must not keep answering "Too many open requests".
        if sheddable and self.open_requests > self.max_open:
            return dict(SHED_RESPONSE)
        fetch = self._query_httpx if self.use_httpx else self._query_aiohttp
        try:
            return await fetch(q)
        except _QueryFailed as failed:
            return failed.result

    def _failure(self, q, e, response=None):
        print("--- Qlever Exception ---")
        print(q)
        # httpx timeouts carry no message, which used to surface as "error": ""
        message = str(e) or f"{type(e).__name__} after {self.settings.qlever_timeout}s"
        # QLever answers some refusals in plain text, e.g. a timeout above its own TIMEOUT;
        # the JSON decode error alone would hide why (httpx only: aiohttp's text is a coroutine)
        body = getattr(response, "text", None)
        if isinstance(body, str) and body.strip():
            message = f"{message}: {body.strip()[:500]}"
        print(message)
        status = response.status_code if response is not None else 0
        return _QueryFailed({"total": 0, "results": [], "error": message, "status": status})

    @staticmethod
    def _checked(results):
        """QLever answered, but with an error (bad query, a SERVICE that failed): not cached either."""
        if "error" in results:
            raise _QueryFailed(results)
        return results

    def _form(self, q):
        """The POST body. ``timeout`` makes QLever itself stop when we stop waiting: without it a
        query we have given up on runs on to the server's own TIMEOUT, and a retry runs beside it.
        QLever refuses a timeout above its configured TIMEOUT unless given the access token, so
        QLMT_QLEVER_TIMEOUT must not exceed the Qleverfile's."""
        return {"query": q, "send": 60, "timeout": f"{self.settings.qlever_timeout}s"}

    # open_requests is decremented in ``finally``: a request cancelled mid-flight (async-lru
    # cancels a shared call once every caller has gone; a client disconnect can cancel the
    # handler) raises CancelledError, which ``except Exception`` does not see. Missed, each one
    # left the count a step higher until every sheddable query was shed.

    @alru_cache(maxsize=500)
    async def _query_aiohttp(self, q):
        response = None
        self.open_requests += 1
        try:
            async with self.client.post(
                self.endpoint,
                data=self._form(q),
                headers={"Accept": "application/qlever-results+json"},
            ) as response:
                ret = await response.json()
        except Exception as e:
            raise self._failure(q, e, response) from e
        finally:
            self.open_requests -= 1
        return self._checked(process_qlever_results(ret))

    @alru_cache(maxsize=500)
    async def _query_httpx(self, q):
        response = None
        self.open_requests += 1
        try:
            response = await self.client.post(
                self.endpoint,
                data=self._form(q),
                headers={"Accept": "application/qlever-results+json"},
            )
            ret = response.json()
        except Exception as e:
            raise self._failure(q, e, response) from e
        finally:
            self.open_requests -= 1
        return self._checked(process_qlever_results(ret))
