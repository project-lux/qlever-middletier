"""Search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from qleverlux.api.deps import get_middletier
from qleverlux.enums import scopeEnum, searchScopeEnum

router = APIRouter()


@router.get("/api/search/{scope}", operation_id="get_search")
async def get_search(
    scope: searchScopeEnum,
    q: str,
    page: int = 1,
    pageLength: int = 0,
    sort: str = "relevance:DESC",
    mt=Depends(get_middletier),
):
    """Search for records matching a query.

    Parameters:
        - scope (searchScopeEnum): The scope of the search. "multi" runs the
          query's OR branches, each in its own scope, and merges the results.
        - q (url encoded dict): The search query.
        - page (int): The page number for pagination of results
        - pageLength (int): The number of results per page
        - sort (str): How to sort the results, default by relevance to the query

    Returns:
        - dict: The search results in the ActivityStreams CollectionPage format
    """
    return await mt.search.search(scope, q, page, pageLength, sort)


@router.get("/api/search-estimate/{scope}", operation_id="get_estimate")
async def get_search_estimate(
    scope: searchScopeEnum, q={}, page=1, mt=Depends(get_middletier)
):
    """The number of records a query matches."""
    return await mt.search.estimate(scope, q, page)


@router.get("/api/translate/{scope}", operation_id="translate_string_query")
async def get_translate(scope: scopeEnum, q: str, mt=Depends(get_middletier)):
    """Translate a simple string query into its JSON query equivalent.

    Parameters:
        - scope (scopeEnum): The scope for the query
        - q (str): The simple search query

    Returns:
        - dict: The JSON query equivalent of the given query
    """
    return JSONResponse(content=mt.search.translate_string_query(scope.value, q))
