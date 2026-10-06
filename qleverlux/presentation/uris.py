"""URI rewriting between the data's URIs and this deployment's.

The data says ``https://lux.collections.yale.edu/...`` whatever machine it is
served from, so queries coming in are rewritten to data URIs before
translation, and records going out are rewritten to this instance's ``mt_uri``.
Every new endpoint needs both directions, which is why they live together here
rather than as ``.replace()`` calls scattered through the handlers.
"""

from __future__ import annotations

import ujson as json


class UriRewriter:
    """Translates URIs between the data namespace and this deployment's."""

    def __init__(self, data_uri: str, mt_uri: str):
        self.data_uri = data_uri
        self.mt_uri = mt_uri
        self.data_records = f"{data_uri}data/"
        self.mt_records = f"{mt_uri}data/"

    def inbound(self, text: str) -> str:
        """Query text as received -> data URIs, ready to translate."""
        return text.replace(self.mt_uri, self.data_uri)

    def outbound(self, uri: str) -> str:
        """A data URI -> this deployment's equivalent."""
        return uri.replace(self.data_uri, self.mt_uri)

    def outbound_record(self, uri: str) -> str:
        """Rewrite only record URIs, leaving vocabulary URIs untouched."""
        return uri.replace(self.data_records, self.mt_records)

    def outbound_json(self, obj):
        """Rewrite every record URI inside a whole document."""
        jstr = json.dumps(obj, escape_forward_slashes=False)
        jstr = jstr.replace(self.data_records, self.mt_records)
        return json.loads(jstr)
