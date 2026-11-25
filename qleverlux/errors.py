"""Turning a failed QLever round-trip into an HTTP response.

Every handler used to repeat the same try/except plus ``if "error" in res``
block. Instead ``run_query`` raises ``QLeverError`` and one exception handler
renders it, so the services only deal with results that worked.

A failure's detail - QLever's message, an exception's text - goes to the log, and to the client
only on a deployment with QLMT_DEBUG_ERRORS: it can name the query, internal hosts and paths.
Validation errors (BadQuery) are the caller's own mistake and are shown as they are.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

#: Used when QLever reports a failure without a usable HTTP status.
DEFAULT_ERROR_STATUS = 504

#: What a client is told when a query fails. QLever's own message goes to the log (and to the
#: response only with QLMT_DEBUG_ERRORS): it can quote the whole query and internal addresses -
#: the vector sidecar's, a SERVICE's.
PUBLIC_MESSAGES = {504: "The query timed out or the server is busy; please try again."}
DEFAULT_PUBLIC_MESSAGE = "The query could not be run."


def debug_errors(request: Request) -> bool:
    """Whether this deployment shows clients the full error (QLMT_DEBUG_ERRORS)."""
    mt = getattr(request.app.state, "mt", None)
    return bool(getattr(getattr(mt, "settings", None), "debug_errors", False))


def error_content(settings, message: str, detail=None) -> dict:
    """``{"error": message}``, plus ``"detail"`` when ``settings`` show full errors."""
    content = {"error": message}
    if detail is not None and getattr(settings, "debug_errors", False):
        content["detail"] = str(detail)
    return content


def error_response(request: Request, status: int, message: str, detail=None) -> JSONResponse:
    """``{"error": message}``, plus ``"detail"`` when the deployment shows full errors."""
    content = {"error": message}
    if detail is not None and debug_errors(request):
        content["detail"] = str(detail)
    return JSONResponse(content=content, status_code=status)


class QLeverError(Exception):
    """A SPARQL query failed, timed out, or was shed under load."""

    def __init__(self, payload, status=None):
        self.payload = payload
        self.status = status or DEFAULT_ERROR_STATUS
        super().__init__(payload.get("error", "QLever request failed"))


class BadQuery(Exception):
    """The query could not be parsed or translated. Surfaces as a 400."""


async def run_query(client, sparql, sheddable=True, fmt=None):
    """Run a query and return its results, raising ``QLeverError`` on failure. ``fmt`` is the
    client's result format (``clients.qlever.JSON`` / ``TSV``); None leaves it the client's default."""
    try:
        res = await client.query(sparql, sheddable=sheddable, **({"fmt": fmt} if fmt else {}))
    except Exception as e:
        res = {"error": str(e), "results": [], "status": 0}
    if "error" in res:
        raise QLeverError(res, res.get("status", 0))
    return res


def install_error_handlers(app):
    @app.exception_handler(QLeverError)
    async def _qlever_error(request: Request, exc: QLeverError):
        detail = exc.payload.get("error")
        print(f"QLever error {exc.status} for {request.url.path}: {detail}")
        message = PUBLIC_MESSAGES.get(exc.status, DEFAULT_PUBLIC_MESSAGE)
        return error_response(request, exc.status, message, detail)

    @app.exception_handler(BadQuery)
    async def _bad_query(request: Request, exc: BadQuery):
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    return app
