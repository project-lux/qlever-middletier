"""Facet endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from qleverlux.api.deps import get_middletier
from qleverlux.enums import scopeEnum

router = APIRouter()


@router.get("/api/facets/{scope}", operation_id="get_facet")
async def get_facet(
    scope: scopeEnum,
    q: str,
    name: str,
    page: int = 1,
    sort: str = "",
    mt=Depends(get_middletier),
):
    """Retrieve facet values for a given facet name and query.

    Parameters:
        scope (scopeEnum): The scope of the search
        q (url encoded dict): The query
        name (str): The name of the facet
        page (int): The page number, defaults to 1
        sort (str): "asc" or "desc"

    Returns:
        - dict: The facet values as an ActivityStreams CollectionPage
    """
    return await mt.facets.facet(scope, q, name, page, sort)
