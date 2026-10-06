"""CMS stub: the Drupal JSON:API paths the front end reads.

Point the front end at it with
``REACT_APP_CMS_API_BASE_URL=<mt_uri>jsonapi/``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from qleverlux.api.deps import get_middletier

router = APIRouter()


def _int_param(request, name, default):
    try:
        return int(request.query_params[name])
    except (KeyError, ValueError):
        return default


@router.get("/jsonapi/node/{node_type}", operation_id="get_cms_collection")
async def get_cms_collection(
    node_type: str, request: Request, mt=Depends(get_middletier)
):
    """Every node of a type, e.g. ``node/faq?page[limit]=100``."""
    # page[limit] / page[offset] are not valid Python identifiers, so they are
    # read off the request rather than declared as parameters
    return mt.cms.collection(
        node_type,
        limit=_int_param(request, "page[limit]", None),
        offset=_int_param(request, "page[offset]", 0),
    )


@router.get("/jsonapi/node/{node_type}/{uuid}", operation_id="get_cms_resource")
async def get_cms_resource(node_type: str, uuid: str, mt=Depends(get_middletier)):
    """One node, e.g. a content page or a results-page overlay."""
    return mt.cms.resource(node_type, uuid)
