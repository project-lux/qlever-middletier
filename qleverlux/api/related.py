"""Related-list endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from qleverlux.api.deps import get_middletier
from qleverlux.enums import scopeEnum

router = APIRouter()

#: The most entries one page may ask for, as on the MarkLogic backend.
MAX_RELATED_PAGE_LENGTH = 1000


@router.get("/api/related-list/{scope}", operation_id="get_related_list")
async def get_related_list(
    scope: scopeEnum,
    name: str,
    uri: str,
    page: int = Query(1, ge=1),
    pageLength: int | None = Query(None, ge=1, le=MAX_RELATED_PAGE_LENGTH),
    mt=Depends(get_middletier),
):
    """The records related to a given record, grouped by how they relate.

    Parameters:
        - name (str): The related list, e.g. relatedToAgent
        - uri (str): The record to find related records for
        - page (int): The page, from 1
        - pageLength (int, optional): Entries per page, at most 1000; default
          QLMT_RELATED_PAGELENGTH (25). An entry is one related record and one
          way it relates, so a record related in several ways spans several
          adjacent entries.

    Returns:
        - dict: An ActivityStreams OrderedCollectionPage, with ``next`` only when
          there is a further page
    """
    return await mt.related.related_list(scope, name, uri, page, pageLength)
