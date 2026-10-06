"""Configuration, statistics and AI translation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from qleverlux.api.deps import get_middletier
from qleverlux.enums import scopeEnum
from qleverlux.models import StatisticsResponse

router = APIRouter()


@router.get("/api/advanced-search-config", operation_id="get_config")
async def get_search_config(mt=Depends(get_middletier)):
    """The advanced search configuration the front end builds its UI from."""
    return JSONResponse(content=mt.catalogue.lux_config.lux_config)


@router.get(
    "/api/stats", response_model=StatisticsResponse, operation_id="get_statistics"
)
async def get_statistics(mt=Depends(get_middletier)):
    """Counts of each class in the database."""
    return await mt.stats.stats()


@router.get("/api/ai-translate", operation_id="ai_translate_string_query")
async def get_ai_translate(
    q: str, prevQuery: str = "", mt=Depends(get_middletier)
):
    """Translate a natural language question into candidate LUX queries.

    Parameters:
        - q (str): The question, or with prevQuery, the change to make to it
        - prevQuery (str, optional): A LUX JSON query to improve rather than
          starting from scratch

    Returns:
        - list: One entry per option the model offered, each with the model's
          natural language reading of it and the query itself, whose "_scope"
          says which scope to search
    """
    return await mt.ai.translate(q, prevQuery)


@router.get(
    "/api/ai-translate/{scope}", operation_id="ai_translate_string_query_scoped"
)
async def get_ai_translate_scoped(
    scope: scopeEnum, q: str, prevQuery: str = "", mt=Depends(get_middletier)
):
    """As /api/ai-translate, for clients that put the scope in the path.

    The scope is accepted but not used: the model picks a scope per option, and
    reports it as "_scope" on each returned query.
    """
    return await mt.ai.translate(q, prevQuery)
