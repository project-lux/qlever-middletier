"""HTTP routes. One module per group of endpoints; no logic beyond
validating parameters and handing off to a service."""

from fastapi import APIRouter

from qleverlux.api import cms, facets, meta, records, related, search

router = APIRouter()
for module in (search, facets, related, records, meta, cms):
    router.include_router(module.router)

__all__ = ["router"]
