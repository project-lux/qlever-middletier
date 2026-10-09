"""Turning a failed QLever round-trip into an HTTP response.

Every handler used to repeat the same try/except plus ``if "error" in res``
block. Instead ``run_query`` raises ``QLeverError`` and one exception handler
renders it, so the services only deal with results that worked.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

#: Used when QLever reports a failure without a usable HTTP status.
DEFAULT_ERROR_STATUS = 504


class QLeverError(Exception):
    """A SPARQL query failed, timed out, or was shed under load."""

    def __init__(self, payload, status=None):
        self.payload = payload
        self.status = status or DEFAULT_ERROR_STATUS
        super().__init__(payload.get("error", "QLever request failed"))


class BadQuery(Exception):
    """The query could not be parsed or translated. Surfaces as a 400."""


async def run_query(client, sparql, sheddable=True):
    """Run a query and return its results, raising ``QLeverError`` on failure."""
    try:
        res = await client.query(sparql, sheddable=sheddable)
    except Exception as e:
        res = {"error": str(e), "results": [], "status": 0}
    if "error" in res:
        raise QLeverError(res, res.get("status", 0))
    return res


def install_error_handlers(app):
    @app.exception_handler(QLeverError)
    async def _qlever_error(request: Request, exc: QLeverError):
        return JSONResponse(content=exc.payload, status_code=exc.status)

    @app.exception_handler(BadQuery)
    async def _bad_query(request: Request, exc: BadQuery):
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    return app
