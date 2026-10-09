"""Access to the wired-up middle tier from a route.

The application object is built once by the lifespan handler and kept on
``app.state``; routes ask for it through ``Depends(get_middletier)``. This
replaces the module-level ``mt`` global that every entry point had to remember
to assign after constructing it.
"""

from __future__ import annotations

from fastapi import Request


def get_middletier(request: Request):
    return request.app.state.mt
