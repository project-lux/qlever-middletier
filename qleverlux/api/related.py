"""Related-list endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from qleverlux.api.deps import get_middletier
from qleverlux.enums import scopeEnum

router = APIRouter()


@router.get("/api/related-list/{scope}", operation_id="get_related_list")
async def get_related_list(
    scope: scopeEnum, name: str, uri: str, page: int = 1, mt=Depends(get_middletier)
):
    """The records related to a given record, grouped by how they relate."""
    return await mt.related.related_list(scope, name, uri, page)
