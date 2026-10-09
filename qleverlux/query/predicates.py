"""The QLever predicate vocabulary.

Pure data plus the rules for resolving a LUX search-term name to a ``lux:``
predicate. Three things read this: the translator, the related-list query
builder, and ``files/derive_from_upstream.py``, which drops any generated
relation whose hops have no predicate here. It lives on its own so all three
can import it rather than reach into the translator (the derive script used to
recover it by AST-parsing the translator's source).

Property path notation used in the values below:

* ``^elt`` is an inverse path (from object to subject)
* ``elt*`` is zero or more, ``elt+`` one or more, ``elt?`` zero or one

``MISSED`` is the sentinel for a term with no predicate in this scope.
"""

from __future__ import annotations

#: Returned when a search term has no predicate in the given scope. A term
#: mapped to "" here is one LUX knows but QLever cannot express directly, and
#: resolves to this too.
MISSED = "missed"

#: Date families expand to a (startOf..., endOf...) pair rather than one predicate.
_DATE_SUFFIXES = {
    "startDate": "Beginning",
    "producedDate": "Beginning",
    "createdDate": "Beginning",
    "endDate": "Ending",
    "activeDate": "Activity",
    "publishedDate": "Publication",
    "encounteredDate": "Encounter",
}

#: Leaf fields that are plain numeric measurements.
_DIMENSIONS = ("height", "width", "depth", "weight", "dimension")

#: Scope -> the Linked Art classes a record in it can have. Every record is
#: typed twice, ``lux:<Scope>`` and ``la:<Class>`` (the data pipeline's
#: PREFIX_BY_TYPE, inverted); search returns the latter as the result's type.
SCOPE_TYPES = {
    "item": ("HumanMadeObject", "DigitalObject"),
    "work": ("LinguisticObject", "VisualItem"),
    "agent": ("Person", "Group"),
    "place": ("Place",),
    "concept": ("Type", "Language", "Material", "Currency", "MeasurementUnit"),
    "event": ("Activity", "Event", "Period"),
    "set": ("Set",),
}


#: Relationship terms: scope -> LUX search term -> lux: predicate name.
SCOPE_FIELDS: dict[str, dict[str, str]] = {
    "agent": {
        "startAt": "placeOfAgentBeginning",
        "endAt": "placeOfAgentEnding",
        "foundedBy": "agentOfAgentBeginning",
        "gender": "gender",
        "occupation": "occupation",
        "nationality": "nationality",
        "professionalActivity": "typeOfAgentActivity",
        "activeAt": "placeOfAgentActivity",
        "createdSet": "^agentOfSetBeginning",
        "produced": "^agentOfItemBeginning",
        "created": "^agentOfWorkBeginning",
        "carriedOut": "^eventCarriedOutBy",
        "curated": "^setCuratedBy",
        "encountered": "^agentOfItemEncounter",
        "founded": "^agentOfAgentBeginning",
        "memberOfInverse": "^agentMemberOfGroup",
        "influencedProduction": "^agentInfluenceOfItemBeginning",
        "influencedCreation": "^agentInfluenceOfWorkBeginning",
        "publishedSet": "^agentOfSetPublication",
        "published": "^agentOfWorkPublication",
        "subjectOfSet": "^setAboutAgent",
        "subjectOfWork": "^workAboutAgent",
        "classification": "agentClassification",
    },
    "item": {
        "producedAt": "placeOfItemBeginning",
        "producedBy": "agentOfItemBeginning",
        "producedUsing": "typeOfItemBeginning",
        "productionInfluencedBy": "agentInfluenceOfItemBeginning",
        "encounteredAt": "placeOfItemEncounter",
        "encounteredBy": "agentOfItemEncounter",
        "carries": "carries",
        "material": "material",
        "subjectOfSet": "^setAboutItem",
        "subjectOfWork": "^workAboutItem",
        "memberOf": "itemMemberOfSet",
        "creationCausedBy": "causeOfProduction",
        "usedForEvent": "",
        "classification": "itemClassification",
    },
    "concept": {
        "broader": "broader",
        "broaderPlus": "broader+",
        "classificationOfSet": "^setClassification",
        "classificationOfConcept": "^conceptClassification",
        "classificationOfEvent": "^eventClassification",
        "classificationOfItem": "^itemClassification",
        "classificationOfAgent": "^agentClassification",
        "classificationOfPlace": "^placeClassification",
        "classificationOfWork": "^workClassification",
        "genderOf": "^gender",
        "languageOf": "^workLanguage",
        "languageOfSet": "^setLanguage",
        "materialOfItem": "^material",
        "influencedByAgent": "agentInfluenceOfConceptBeginning",
        "influencedByConcept": "conceptInfluenceOfConceptBeginning",
        "influencedByEvent": "eventInfluenceOfConceptBeginning",
        "influencedByPlace": "placeInfluenceOfConceptBeginning",
        "narrower": "^broader",
        "nationalityOf": "^nationality",
        "occupationOf": "^occupation",
        "professionalActivityOf": "^typeOfAgentActivity",
        "subjectOfSet": "^setAboutConcept",
        "subjectOfWork": "^workAboutConcept",
        "usedToProduce": "^typeOfItemBeginning",
        "classification": "conceptClassification",
    },
    "event": {
        "carriedOutBy": "agentOfEvent",
        "tookPlaceAt": "placeOfEvent",
        "used": "eventUsedSet",
        "causeOfEvent": "causeOfEvent",
        "causedCreationOf": "^causeOfWorkBeginning",
        "subjectOfSet": "^setAboutEvent",
        "subjectOfWork": "^workAboutEvent",
        "classification": "eventClassification",
    },
    "place": {
        "partOf": "placePartOf",
        "partOfPlus": "placePartOf+",
        "activePlaceOfAgent": "^placeOfAgentActivity",
        "startPlaceOfAgent": "^placeOfAgentBeginning",
        "producedHere": "^placeOfItemBeginning",
        "createdHere": "^placeOfWorkBeginning",
        "setCreatedHere": "^placeOfSetBeginning",
        "endPlaceOfAgent": "^placeOfAgentEnding",
        "encounteredHere": "^placeOfItemEncounter",
        "placeOfEvent": "^placeOfEvent",
        "setPublishedHere": "^placeOfSetPublication",
        "publishedHere": "^placeOfWorkPublication",
        "subjectOfSet": "^setAboutPlace",
        "subjectOfWork": "^workAboutPlace",
        "classification": "placeClassification",
    },
    "set": {
        "aboutConcept": "setAboutConcept",
        "aboutEvent": "setAboutEvent",
        "aboutItem": "setAboutItem",
        "aboutAgent": "setAboutAgent",
        "aboutPlace": "setAboutPlace",
        "aboutWork": "setAboutWork",
        "createdAt": "placeOfSetBeginning",
        "createdBy": "agentOfSetBeginning",
        "creationCausedBy": "causeOfSetBeginning",
        "creationInfluencedBy": "agentInfluenceOfSetBeginning",
        "curatedBy": "setCuratedBy",
        "publishedAt": "placeOfSetPublication",
        "publishedBy": "agentOfSetPublication",
        "containingSet": "^setMemberOfSet",
        "containingItem": "^itemMemberOfSet",
        "usedForEvent": "^eventUsedSet",
        "memberOf": "setMemberOfSet",
        "classification": "setClassification",
    },
    "work": {
        "aboutConcept": "workAboutConcept",
        "aboutEvent": "workAboutEvent",
        "aboutItem": "workAboutItem",
        "aboutAgent": "workAboutAgent",
        "aboutPlace": "workAboutPlace",
        "aboutWork": "workAboutWork",
        "createdAt": "placeOfWorkBeginning",
        "createdBy": "agentOfWorkBeginning",
        "creationCausedBy": "causeOfWorkBeginning",
        "creationInfluencedBy": "agentInfluenceOfWorkBeginning",
        "publishedAt": "placeOfWorkPublication",
        "publishedBy": "agentOfWorkPublication",
        "language": "workLanguage",
        "partOfWork": "workPartOf",
        "subjectOfSet": "^setAboutWork",
        "subjectOfWork": "^workAboutWork",
        "carriedBy": "^carries",
        "containsWork": "^workPartOf",
        "classification": "workClassification",
    },
}

#: Leaf-value terms: scope -> LUX search term -> lux: predicate name.
SCOPE_LEAF_FIELDS: dict[str, dict[str, str]] = {
    "agent": {},
    "concept": {},
    "event": {},
    "place": {},
    "set": {},
    "work": {},
    "item": {},
}


def get_predicate(rel: str, scope: str) -> str:
    """Resolve a relationship term to a prefixed predicate, inverses included."""
    if rel == "classification":
        return f"lux:{scope}{rel[0].upper()}{rel[1:]}"
    elif rel == "memberOf":
        typ = "Group" if scope == "agent" else "Set"
        return f"lux:{scope}{rel[0].upper()}{rel[1:]}{typ}"
    else:
        # `or MISSED`, not a default: a few terms map to "" to record that LUX
        # has the search term but QLever has no single predicate for it (an
        # item is used for an event only via the set it belongs to). Treat that
        # exactly like an unmapped term instead of indexing off the end of it.
        p = SCOPE_FIELDS[scope].get(rel) or MISSED
        if p[0] == "^":
            return f"^lux:{p[1:]}"
        else:
            return f"lux:{p}"


def get_leaf_predicate(field: str, scope: str) -> str | list[str]:
    """Resolve a leaf term to a predicate, or to a [start, end] pair for dates."""
    if field in _DIMENSIONS:
        return f"lux:{field}"

    if field in _DATE_SUFFIXES:
        suffix = _DATE_SUFFIXES[field]
        if scope == "event" and suffix in ("Beginning", "Ending"):
            # An event's timespan is the event itself: the data has
            # startOfEvent/endOfEvent, not the Beginning/Ending pair every
            # other scope uses. The compared predicate comes first.
            if suffix == "Ending":
                return ["lux:endOfEvent", "lux:startOfEvent"]
            return ["lux:startOfEvent", "lux:endOfEvent"]
        return [
            f"lux:startOf{scope.title()}{suffix}",
            f"lux:endOf{scope.title()}{suffix}",
        ]
    elif "CreationOrPublicationDate" in field:
        # "Creation" is the record's Beginning in the data; there is no
        # startOf<Scope>Creation predicate.
        return [
            f"lux:startOf{scope.title()}Beginning|lux:startOf{scope.title()}Publication",
            f"lux:endOf{scope.title()}Beginning|lux:endOf{scope.title()}Publication",
        ]
    elif field in ("hasDigitalImage", "isOnline"):
        return f"lux:{scope}{field[0].upper()}{field[1:]}"
    elif field in (f"{scope}HasDigitalImage", f"{scope}IsOnline"):
        return f"lux:{field}"

    return SCOPE_LEAF_FIELDS[scope].get(field, MISSED)
