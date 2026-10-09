"""Record retrieval endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from qleverlux.api.deps import get_middletier
from qleverlux.enums import classEnum, profileEnum

router = APIRouter()


@router.get("/data/{scope}/{identifier}", operation_id="get_record")
async def get_record(
    scope: classEnum,
    identifier: str,
    profile: profileEnum = None,
    mt=Depends(get_middletier),
):
    """Retrieve an individual record.

    Parameters:
        - scope (str): The class of the record.
        - identifier (str): The identifier of the record: a UUID, or for a
          store with QLMT_LMDB_KEY_FORMAT=qid a Wikidata Q-id.
        - profile (str, optional): "name" or "results"; default is the full record.

    Returns:
        - dict: The record.
    """
    return await mt.records.get_record(scope, identifier, profile)
