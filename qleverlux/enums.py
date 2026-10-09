"""Path and query parameter enums shared by the routes.

``scopeEnum`` is the seven search scopes the query language works in;
``classEnum`` is the finer set of record classes that appear in URL paths and
gets folded back onto scopes when building HAL links.
"""

from __future__ import annotations

from enum import StrEnum


class scopeEnum(StrEnum):
    ITEM = "item"
    WORK = "work"
    AGENT = "agent"
    PLACE = "place"
    CONCEPT = "concept"
    SET = "set"
    EVENT = "event"


#: The scope name for a search that spans several scopes at once.
MULTI_SCOPE = "multi"


class searchScopeEnum(StrEnum):
    """Scopes a search may be issued against: the seven, plus ``multi``.

    Only search and search-estimate accept ``multi`` - a facet, related list or
    sort is defined per scope and has no meaning across a union of them.
    """

    ITEM = "item"
    WORK = "work"
    AGENT = "agent"
    PLACE = "place"
    CONCEPT = "concept"
    SET = "set"
    EVENT = "event"
    MULTI = MULTI_SCOPE


class classEnum(StrEnum):
    OBJECT = "object"
    DIGITAL = "digital"
    TEXT = "text"
    VISUAL = "visual"
    PLACE = "place"
    PERSON = "person"
    GROUP = "group"
    SET = "set"
    CONCEPT = "concept"
    EVENT = "event"
    PERIOD = "period"
    ACTIVITY = "activity"


class profileEnum(StrEnum):
    DEFAULT = ""
    NAME = "name"
    RESULTS = "results"


#: Record class -> the search scope its HAL links are defined against.
CLASS_TO_SCOPE = {
    "person": "agent",
    "group": "agent",
    "object": "item",
    "digital": "item",
    "place": "place",
    "set": "set",
    "event": "event",
    "concept": "concept",
    "period": "event",
    "activity": "event",
    "text": "work",
    "visual": "work",
    "image": "work",
}


def scope_for_class(scope):
    """Fold a record class onto its search scope, warning if it is unknown."""
    hscope = CLASS_TO_SCOPE.get(scope)
    if hscope is None:
        print(f"MISSED SCOPE IN HAL: {scope}")
        return scope
    return hscope


#: Record class -> the Linked Art type a record of that class has. Where several
#: types share a class (concept), this is the one a store keyed by type uses.
CLASS_TO_TYPE = {
    "object": "HumanMadeObject",
    "digital": "DigitalObject",
    "text": "LinguisticObject",
    "visual": "VisualItem",
    "person": "Person",
    "group": "Group",
    "place": "Place",
    "concept": "Type",
    "event": "Event",
    "period": "Period",
    "activity": "Activity",
    "set": "Set",
}

#: Linked Art type -> the record class in its URL path.
TYPE_TO_CLASS = {t: c for c, t in CLASS_TO_TYPE.items()} | {
    "Language": "concept",
    "Material": "concept",
    "Currency": "concept",
    "MeasurementUnit": "concept",
}
