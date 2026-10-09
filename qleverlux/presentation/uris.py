"""URI rewriting between the data's URIs and this deployment's.

The data says ``https://lux.collections.yale.edu/...`` whatever machine it is
served from, so queries coming in are rewritten to data URIs before
translation, and records going out are rewritten to this instance's ``mt_uri``.
Every new endpoint needs both directions, which is why they live together here
rather than as ``.replace()`` calls scattered through the handlers.

This deployment always serves a record at ``<mt_uri>data/<class>/<id>``. Where
it sits in the data is ``data_uri`` + ``record_path``. LUX's own data has the
class in it too (``data/{class}/{id}``), so going out is a prefix swap. Data
that does not (Wikidata's ``{id}``) cannot be rewritten by string alone: the
class comes from the record's Linked Art type, so only a URI whose type is
known is rewritten, and any other is left as the data URI. Inbound always
works, because the class is in the URI being rewritten.
"""

from __future__ import annotations

import re

import ujson as json

from qleverlux.enums import TYPE_TO_CLASS

#: One path segment of a URI embedded in text.
_SEGMENT = r"[^/?#\"'\s<>]+"


class UriRewriter:
    """Translates URIs between the data namespace and this deployment's."""

    def __init__(self, data_uri: str, mt_uri: str, record_path="data/{class}/{id}"):
        self.data_uri = data_uri
        self.mt_uri = mt_uri
        self.record_path = record_path
        #: Whether a data URI carries the record class, so a string prefix
        #: swap suffices in both directions.
        self.classed = "{class}" in record_path
        if self.classed:
            prefix = record_path.split("{class}", 1)[0]
            self.data_records = f"{data_uri}{prefix}"
            self.mt_records = f"{mt_uri}data/"
        self.data_pattern = self._pattern(data_uri, record_path)
        self.mt_pattern = self._pattern(mt_uri, "data/{class}/{id}")

    @staticmethod
    def _pattern(base, path):
        rx = re.escape(base + path)
        rx = rx.replace(re.escape("{class}"), f"(?P<cls>{_SEGMENT})")
        rx = rx.replace(re.escape("{id}"), f"(?P<id>{_SEGMENT})")
        return re.compile(rx)

    def data_record(self, cls: str, identifier: str) -> str:
        """The data URI of a record, as QLever knows it."""
        return self.data_uri + self.record_path.format(**{"class": cls, "id": identifier})

    def mt_record(self, cls: str, identifier: str) -> str:
        """The URI this deployment serves a record at."""
        return f"{self.mt_uri}data/{cls}/{identifier}"

    def inbound(self, text: str) -> str:
        """Query text as received -> data URIs, ready to translate."""
        if not self.classed:
            text = self.mt_pattern.sub(
                lambda m: self.data_record(m["cls"], m["id"]), text
            )
        return text.replace(self.mt_uri, self.data_uri)

    def outbound(self, uri: str, la_type: str | None = None) -> str:
        """A data URI -> this deployment's equivalent."""
        if self.classed:
            return uri.replace(self.data_uri, self.mt_uri)
        return self.outbound_record(uri, la_type)

    def outbound_record(self, uri: str, la_type: str | None = None) -> str:
        """Rewrite only record URIs, leaving vocabulary URIs untouched.

        ``la_type`` is the record's Linked Art type, e.g. ``Person``; it is
        only needed, and only read, when the data URI has no class in it.
        """
        if self.classed:
            return uri.replace(self.data_records, self.mt_records)
        m = self.data_pattern.fullmatch(uri)
        cls = TYPE_TO_CLASS.get(la_type)
        if m is None or cls is None:
            return uri
        return self.mt_record(cls, m["id"])

    def outbound_json(self, obj):
        """Rewrite every record URI inside a whole document."""
        if self.classed:
            jstr = json.dumps(obj, escape_forward_slashes=False)
            jstr = jstr.replace(self.data_records, self.mt_records)
            return json.loads(jstr)
        return self._outbound_typed(obj)

    def _outbound_typed(self, obj):
        """Rewrite the ``id`` of every node that says what ``type`` it is."""
        if isinstance(obj, list):
            return [self._outbound_typed(x) for x in obj]
        if not isinstance(obj, dict):
            return obj
        out = {k: self._outbound_typed(v) for k, v in obj.items()}
        if isinstance(out.get("id"), str) and isinstance(out.get("type"), str):
            out["id"] = self.outbound_record(out["id"], out["type"])
        return out
