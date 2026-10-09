"""A stand-in for the Drupal CMS the front end reads its editorial content from.

lux-frontend points ``REACT_APP_CMS_API_BASE_URL`` at a Drupal JSON:API
(``https://lux-cms.collections.yale.edu/jsonapi/``) and asks it for two shapes
of thing: a collection, ``node/<type>?page[limit]=100``, and a single resource,
``node/<type>/<uuid>``. It only ever reads ``data`` - ``attributes`` for the
text and images, and ``id`` - so the documents here reproduce Drupal's
envelope but not its links, filters, sparse fieldsets or includes.

Content comes from ``<cms_path>/node/<type>.json``, each a list of JSON:API
resource objects as written by ``files/snapshot_cms.py``. A type with no file is
an empty collection, not an error, so the front end renders without that
section rather than failing.

Edits on disk show up without a restart: every request compares the files'
modification times with those last loaded and rereads them if anything
changed. That is a handful of stat calls, and it works under ``workers.py``,
where each process holds its own copy and a reload endpoint would reach only
the one worker that took the request. A file caught half-written or left as
invalid JSON keeps the previous content in service until it parses.
"""

from __future__ import annotations

import glob
import os

import ujson as json
from fastapi.responses import JSONResponse

MEDIA_TYPE = "application/vnd.api+json"
JSONAPI = {
    "version": "1.0",
    "meta": {"links": {"self": {"href": "http://jsonapi.org/format/1.0/"}}},
}
#: Drupal's page size when the request does not give page[limit]
DEFAULT_PAGE_LIMIT = 50


class CmsService:
    def __init__(self, settings):
        self.settings = settings
        self.base = f"{settings.mt_uri}jsonapi/"
        self.pattern = os.path.join(settings.cms_path, "node", "*.json")
        self.nodes = {}
        self.by_id = {}
        self.signature = None
        self.reload_if_changed()
        if not self.nodes:
            print(f"CMS: no content under {settings.cms_path}/node; /jsonapi is empty")

    def _signature(self):
        """What the files on disk look like now: (path, mtime, size) for each."""
        sig = []
        for path in sorted(glob.glob(self.pattern)):
            try:
                st = os.stat(path)
            except OSError:
                continue  # removed between the glob and the stat
            sig.append((path, st.st_mtime_ns, st.st_size))
        return tuple(sig)

    def reload_if_changed(self):
        """Reread the content if any file was added, removed or modified.

        Returns True if new content was loaded.
        """
        sig = self._signature()
        if sig == self.signature:
            return False
        # recorded even on failure, so a broken file is reported once rather
        # than on every request; fixing it changes the signature again
        self.signature = sig
        nodes = {}
        try:
            for path, _, _ in sig:
                node_type = os.path.splitext(os.path.basename(path))[0]
                with open(path) as fh:
                    nodes[node_type] = json.load(fh)
            by_id = {
                (node_type, r["id"]): r
                for node_type, resources in nodes.items()
                for r in resources
            }
        except (OSError, ValueError, TypeError, KeyError) as e:
            print(f"CMS: not reloading, {path} is unreadable ({e}); keeping previous content")
            return False
        if self.nodes:
            print(f"CMS: reloaded {len(by_id)} resources from {len(nodes)} files")
        self.nodes, self.by_id = nodes, by_id
        return True

    def _response(self, doc, status=200):
        return JSONResponse(
            content={"jsonapi": JSONAPI, **doc},
            status_code=status,
            media_type=MEDIA_TYPE,
        )

    def _error(self, status, title, detail):
        return self._response(
            {"errors": [{"title": title, "status": str(status), "detail": detail}]},
            status,
        )

    def collection(self, node_type, limit=None, offset=0):
        """``node/<type>``, paginated the way Drupal does it."""
        self.reload_if_changed()
        resources = self.nodes.get(node_type, [])
        # the LUX CMS honours page[limit]=100 - the front end relies on getting
        # every hero image and featured block in one request
        limit = DEFAULT_PAGE_LIMIT if limit is None else max(0, limit)
        offset = max(0, offset)
        page = resources[offset : offset + limit]

        url = f"{self.base}node/{node_type}"
        links = {"self": {"href": url}}
        if offset + limit < len(resources):
            links["next"] = {
                "href": f"{url}?page[offset]={offset + limit}&page[limit]={limit}"
            }
        return self._response(
            {"data": page, "meta": {"count": len(resources)}, "links": links}
        )

    def resource(self, node_type, uuid):
        """``node/<type>/<uuid>``."""
        self.reload_if_changed()
        r = self.by_id.get((node_type, uuid))
        if r is None:
            return self._error(
                404, "Not Found", f"No node--{node_type} with id {uuid}."
            )
        return self._response(
            {"data": r, "links": {"self": {"href": f"{self.base}node/{node_type}/{uuid}"}}}
        )
