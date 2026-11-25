"""Stage 2: one concise natural-language card per record, by record class.

A card is ``(title, text)``; the encoder formats it as ``title: {title} | text: {text}``, the
document form EmbeddingGemma 2 recommends. Identity fields come first, long notes last and capped,
so mean pooling is not diluted by boilerplate. What each class takes follows the design doc:

=========  ===================================================  ======================================
Class      From the record (and 1-2 hop references)             From Stage 1 (incoming facets)
=========  ===================================================  ======================================
item       type, materials, technique, maker/place/date,        -
           depicted subjects (via the work it shows/carries),
           sets, description
work       type, creator/place/date, subjects, language,        creators, materials, object types,
           description                                           places and dates of carrying objects
set        type, creator, curator, parent set, scope note       top member creators, types, subjects
agent      kind, nationality, occupation, gender, birth/death   active places, object types, mediums,
           or formation/dissolution, residence, groups, bio      what their works are about and what their
                                                                 objects depict (tf-idf), active span,
                                                                 when their works were published
place      type, part-of chain, notes                           - (never incoming objects)
concept    type, broader chain, scope note                      - (never incoming objects)
event      type, date, place, participants, part of, notes      -
=========  ===================================================  ======================================

Excluded on purpose: identifiers, dimensions, provenance, credit lines, member/carrier lists.
"""

from __future__ import annotations

from .labels import LabelStore
from .linkedart import (
    GENDER,
    aat_ids,
    NATIONALITY,
    OCCUPATION,
    alternate_names,
    as_list,
    classes_of,
    date_phrase,
    event_parts,
    primary_name,
    ref_ids,
    statements,
)
from .records import SCOPE_OF_CLASS, RecordStore

LIST = 6  # values per field
# Blanket classes LUX adds beside the specific type: "works (art)", and "Collection Item", which is on
# 99.8% of objects (it showed up in cards as "items").
GENERIC_TYPES = {"300133025", "300404024"}
CHAIN = 6  # ancestors per hierarchy


def _join(values) -> str:
    return ", ".join(values)


class CardBuilder:
    def __init__(self, records: RecordStore, labels: LabelStore, max_note_words: int = 300,
                 skip_name_only: bool = False):
        self.records, self.labels, self.max_note_words = records, labels, max_note_words
        # No card for a record whose card would hold only its name and other names: on full LUX
        # that is half of all agents, whose vectors could only match on how a name looks.
        self.skip_name_only = skip_name_only
        self._parents: dict[str, list[str]] = {}
        self._generics: dict[str, bool] = {}

    # -- shared pieces ---------------------------------------------------------------------------

    def _labels(self, value, limit: int = LIST) -> list[str]:
        return self.labels.labels(ref_ids(value), limit)

    def _types(self, value, limit: int = LIST) -> list[str]:
        """Labels of classifications, the blanket ones dropped."""
        specific = [cx for cx in as_list(value) if isinstance(cx, dict) and not aat_ids(cx) & GENERIC_TYPES]
        return self._labels(specific, limit)

    def _activity(self, event: dict | None) -> dict[str, list[str] | str]:
        """Agents, places, techniques and the date of an event and its parts."""
        agents, places, techniques, date = [], [], [], None
        for part in event_parts(event):
            agents += list(ref_ids(part.get("carried_out_by")))
            places += list(ref_ids(part.get("took_place_at")))
            techniques += list(ref_ids(part.get("technique")))
            date = date or date_phrase(part.get("timespan"))
        return {
            "agents": self.labels.labels(agents, LIST),
            "places": self.labels.labels(places, LIST),
            "techniques": self.labels.labels(techniques, LIST),
            "date": date,
        }

    def _chain(self, iri: str, prop: str) -> list[str]:
        """Labels of the first-parent ancestor chain along ``prop`` (broader / part_of), nearest first."""
        chain, seen, current = [], {iri}, iri
        while len(chain) < CHAIN:
            parents = self._parents.get(current)
            if parents is None:
                record = self.records.get(current) or {}
                parents = self._parents[current] = list(ref_ids(record.get(prop)))[:3]
                if len(self._parents) > 500_000:
                    self._parents.clear()
            nxt = next((p for p in parents if p not in seen), None)
            if nxt is None:
                break
            seen.add(nxt)
            label = self.labels.label(nxt)
            if label:
                chain.append(label)
            current = nxt
        return chain

    def _depicted_subjects(self, record: dict) -> list[str]:
        """Two hops: the subjects of the works an object shows or carries."""
        subjects = []
        for work in list(ref_ids(record.get("shows"))) + list(ref_ids(record.get("carries"))):
            work_record = self.records.get(work) or {}
            subjects += list(ref_ids(work_record.get("about"))) + list(ref_ids(work_record.get("represents")))
            if len(subjects) > 3 * LIST:
                break
        return self.labels.labels(subjects, LIST)

    @staticmethod
    def _field(parts: list[str], name: str, values) -> None:
        if isinstance(values, str):
            values = [values] if values else []
        if values:
            parts.append(f"{name}: {_join(values)}.")

    def _facets(self, parts: list[str], facets: dict, fields: list[tuple[str, str]]) -> None:
        for kind, name in fields:
            values = facets.get(kind, [])
            if kind in ("object type", "work type"):
                values = [v for v in values if not self._generic(v)]
            self._field(parts, name, self.labels.labels(values, LIST))

    def _span(self, parts: list[str], facets: dict, span: str, name: str) -> None:
        """A (start year, end year) facet as "1650" or "1650–1702"."""
        if span in facets:
            start, end = facets[span]
            self._field(parts, name, start if start == end else f"{start}–{end}")

    def _generic(self, concept: str) -> bool:
        """Whether a facet value is one of the blanket GENERIC_TYPES, by the AAT its record names."""
        key = concept.rsplit("/", 1)[-1]
        if key not in self._generics:
            self._generics[key] = bool(aat_ids(self.records.get(key) or {}) & GENERIC_TYPES)
        return self._generics[key]

    def _holding_only(self, r: dict) -> bool:
        """A holding with nothing of its own: it carries a work (a text: the library case) and has no
        maker, place, technique, date, material or statement. Its card would only repeat the work's
        (library books are Books + Collection Item + a library), so it gets none, and the work card
        says "Realized as: Books". Objects that *show* a work (prints, specimens, coins) are the
        thing itself and always keep their card."""
        if not any(ref_ids(r.get("carries"))) or any(ref_ids(r.get("made_of"))):
            return False
        for part in event_parts(r.get("produced_by")):
            if part.get("timespan") or any(any(ref_ids(part.get(key)))
                                           for key in ("carried_out_by", "took_place_at", "technique")):
                return False
        return not statements(r, 1)

    # -- per class -------------------------------------------------------------------------------

    def card(self, record: dict, facets: dict | None = None) -> tuple[str, str] | None:
        title = primary_name(record)
        scope = SCOPE_OF_CLASS.get(record.get("type", ""))
        if not title or not scope:
            return None
        if scope == "item" and self._holding_only(record):
            return None
        parts: list[str] = []
        self._field(parts, "Also known as", alternate_names(record, title, 3))
        names_only = len(parts)
        getattr(self, f"_card_{scope}")(record, parts, facets or {})
        notes = statements(record, self.max_note_words)
        if notes:
            parts.append(" ".join(notes))
        if self.skip_name_only and len(parts) == names_only:
            return None
        return title, " ".join(parts) or title

    def _card_item(self, r, parts, facets):
        self._field(parts, "Type", self._types(r.get("classified_as")))
        self._field(parts, "Made of", self._labels(r.get("made_of")))
        made = self._activity(r.get("produced_by"))
        self._field(parts, "Technique", made["techniques"])
        self._field(parts, "Made by", made["agents"])
        self._field(parts, "Made in", made["places"])
        self._field(parts, "Date", made["date"])
        self._field(parts, "Subjects", self._depicted_subjects(r))
        self._field(parts, "Part of", self._labels(r.get("member_of"), 3))

    def _card_work(self, r, parts, facets):
        self._field(parts, "Type", self._types(r.get("classified_as")))
        made = self._activity(r.get("created_by"))
        self._field(parts, "Created by", made["agents"])
        self._field(parts, "Created in", made["places"])
        self._field(parts, "Date", made["date"])
        subjects = list(ref_ids(r.get("about"))) + list(ref_ids(r.get("represents")))
        self._field(parts, "Subjects", self.labels.labels(subjects, LIST))
        self._field(parts, "Language", self._labels(r.get("language"), 3))
        # Incoming: the objects carrying this work, where MDS keeps maker, place and date.
        self._facets(parts, facets, [("creator", "Realized by"), ("object type", "Realized as"),
                                     ("material", "Materials"), ("place", "Made in")])
        if "dates" in facets and not made["date"]:
            start, end = facets["dates"]
            self._field(parts, "Date", start if start == end else f"{start}–{end}")

    def _card_set(self, r, parts, facets):
        self._field(parts, "Type", self._types(r.get("classified_as")))
        made = self._activity(r.get("created_by"))
        self._field(parts, "Created by", made["agents"])
        curators = [a for use in as_list(r.get("used_for")) for a in ref_ids(use.get("carried_out_by"))]
        self._field(parts, "Curated by", self.labels.labels(curators, 3))
        self._field(parts, "Part of", self._labels(r.get("member_of"), 3))
        if len(" ".join(statements(r, 60)).split()) < 50:  # members speak for sets with sparse notes
            self._facets(parts, facets, [("creator", "Includes works by"), ("object type", "Includes"),
                                         ("subject", "Subjects")])

    def _card_agent(self, r, parts, facets):
        kinds = {"nationality": [], "occupation": [], "gender": [], "other": []}
        for cx in as_list(r.get("classified_as")):
            meta = classes_of(cx)
            slot = ("nationality" if NATIONALITY in meta else "occupation" if OCCUPATION in meta
                    else "gender" if GENDER in meta else "other")
            kinds[slot] += list(ref_ids(cx))
        self._field(parts, "Kind", self.labels.labels(kinds["other"], 3))
        self._field(parts, "Nationality", self.labels.labels(kinds["nationality"], 3))
        self._field(parts, "Occupation", self.labels.labels(kinds["occupation"], LIST))
        self._field(parts, "Gender", self.labels.labels(kinds["gender"], 1))
        begin, end = (r.get("born"), r.get("died")) if r.get("type") == "Person" else (r.get("formed_by"), r.get("dissolved_by"))
        b, e = self._activity(begin), self._activity(end)
        begun, ended = ("Born", "Died") if r.get("type") == "Person" else ("Formed", "Dissolved")
        self._field(parts, begun, _join(x for x in (b["date"], _join(b["places"])) if x))
        self._field(parts, ended, _join(x for x in (e["date"], _join(e["places"])) if x))
        self._field(parts, "Residence", self._labels(r.get("residence"), 3))
        self._field(parts, "Member of", self._labels(r.get("member_of"), 3))
        # The two kinds of subject are named for the relation, so an author on Edo prints does not read
        # like an Edo print designer; "subject" is the single kind of facets extracted before the split.
        self._facets(parts, facets, [("place", "Active in"), ("object type", "Associated with"),
                                     ("material", "Mediums"), ("work subject", "Works about"),
                                     ("depicted subject", "Depicted"), ("subject", "Subjects"),
                                     ("work type", "Works")])
        self._span(parts, facets, "dates", "Active")
        self._span(parts, facets, "published", "Works published")

    def _card_place(self, r, parts, facets):
        self._field(parts, "Type", self._types(r.get("classified_as")))
        self._field(parts, "Part of", " > ".join(self._chain(r["id"], "part_of")))

    def _card_concept(self, r, parts, facets):
        kind = {"Material": "material", "Language": "language", "Currency": "currency",
                "MeasurementUnit": "unit of measurement"}.get(r.get("type"))
        self._field(parts, "Kind", kind or "")
        self._field(parts, "Type", self._types(r.get("classified_as")))
        self._field(parts, "Broader", " > ".join(self._chain(r["id"], "broader")))

    def _card_event(self, r, parts, facets):
        self._field(parts, "Type", self._types(r.get("classified_as")))
        date = date_phrase(r.get("timespan"))
        happened = self._activity(r)
        self._field(parts, "Date", date or happened["date"])
        self._field(parts, "Place", happened["places"])
        self._field(parts, "Participants", happened["agents"])
        self._field(parts, "Part of", self._labels(r.get("part_of"), 3))
