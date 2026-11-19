"""Linked Art helpers: names, statements, references and timespans, without trusting ``_label``.

``_label`` is informative only (per Linked Art), so names come from ``identified_by`` Name nodes and
classifications are recognised through the AAT ``equivalent`` every LUX reference embeds.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Iterator

AAT = "http://vocab.getty.edu/aat/"

PRIMARY_NAME = "300404670"
SORT_NAME = "300451544"
ALTERNATE_NAME = "300264273"
DISPLAY_NAME = "300404669"  # also the "display date" of a timespan
ENGLISH = "300388277"

# Statement (referred_to_by) kinds that are boilerplate rather than description, by the AAT terms
# Linked Art documents for them. MDS carries only Description (300435416), Biography (300435422) and
# Note (300027200); run `python -m lux_embed kinds` on a new dataset to see what else it holds.
SKIPPED_STATEMENTS = {
    "300026687",  # credit line (acknowledgments)
    "300435434",  # copyright/licensing statement
    "300444174",  # provenance statement
    "300435430",  # dimensions statement
    "300311705",  # citation
}

# Meta-classifications of an agent's classified_as entries.
NATIONALITY = "300379842"
GENDER = "300055147"
OCCUPATION = "300263369"

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def aat_ids(node: dict) -> set[str]:
    """The AAT numbers a reference stands for: its own id, or the AAT ``equivalent`` it embeds."""
    found = set()
    for candidate in [node, *as_list(node.get("equivalent"))]:
        ident = candidate.get("id", "") if isinstance(candidate, dict) else ""
        if ident.startswith(AAT):
            found.add(ident[len(AAT) :])
    return found


def classes_of(node: dict) -> set[str]:
    """AAT numbers of everything ``node`` is classified as."""
    found: set[str] = set()
    for cx in as_list(node.get("classified_as")):
        found |= aat_ids(cx)
    return found


def languages_of(node: dict) -> set[str]:
    found: set[str] = set()
    for lang in as_list(node.get("language")):
        found |= aat_ids(lang)
        if lang.get("notation") == "en":
            found.add(ENGLISH)
    return found


def clean(text: str) -> str:
    """Strips HTML and collapses whitespace."""
    if "<" in text:
        text = _TAG.sub(" ", text)
    if "&" in text:
        text = html.unescape(text)
    return _SPACE.sub(" ", text).strip()


def _english_or_untagged(node: dict) -> bool:
    langs = languages_of(node)
    return not langs or ENGLISH in langs


def names(record: dict) -> list[dict]:
    return [n for n in as_list(record.get("identified_by")) if n.get("type") == "Name" and n.get("content")]


def primary_name(record: dict) -> str | None:
    """The record's name as LUX displays it: the English primary name, else an untagged primary name,
    else any primary name, else an English or untagged name, else the first name. The pipeline
    guarantees one primary name per language, so the first two steps are unambiguous."""
    all_names = names(record)
    if not all_names:
        return None
    primary = [n for n in all_names if PRIMARY_NAME in classes_of(n)]
    for pool in (
        [n for n in primary if ENGLISH in languages_of(n)],
        [n for n in primary if not languages_of(n)],
        primary,
        [n for n in all_names if _english_or_untagged(n)],
        all_names,
    ):
        if pool:
            return clean(pool[0]["content"])
    return None


def alternate_names(record: dict, primary: str | None, limit: int = 4) -> list[str]:
    """Distinct English/untagged names other than the primary, sort names excluded."""
    seen = {primary.casefold()} if primary else set()
    out = []
    for n in names(record):
        if SORT_NAME in classes_of(n) and PRIMARY_NAME not in classes_of(n):
            continue
        if not _english_or_untagged(n):
            continue
        text = clean(n["content"])
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
            if len(out) == limit:
                break
    return out


def statements(record: dict, max_words: int = 300) -> list[str]:
    """English or untagged ``referred_to_by`` texts, boilerplate kinds skipped, capped in words."""
    out, words, seen = [], 0, set()
    for st in as_list(record.get("referred_to_by")):
        content = st.get("content")
        if not content or not _english_or_untagged(st) or classes_of(st) & SKIPPED_STATEMENTS:
            continue
        text = clean(content)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        tokens = text.split()
        if words + len(tokens) > max_words:
            tokens = tokens[: max(0, max_words - words)]
            if not tokens:
                break
            text = " ".join(tokens) + " …"
        out.append(text)
        words += len(tokens)
        if words >= max_words:
            break
    return out


def ref_ids(value) -> Iterator[str]:
    """IRIs of the LUX records a property points at."""
    for node in as_list(value):
        if isinstance(node, dict):
            ident = node.get("id", "")
            if ident.startswith("https://lux.collections.yale.edu/data/"):
                yield ident


def year(stamp: str | None) -> int | None:
    if not stamp:
        return None
    match = re.match(r"(-?\d{1,4})", stamp)
    return int(match.group(1)) if match else None


def date_phrase(timespan: dict | None) -> str | None:
    """A display date: the timespan's own name if it has one, else its years."""
    if not timespan:
        return None
    for n in as_list(timespan.get("identified_by")):
        if n.get("content"):
            return clean(n["content"])
    begin, end = year(timespan.get("begin_of_the_begin")), year(timespan.get("end_of_the_end"))
    if begin is None and end is None:
        return None
    if begin is None or end is None or begin == end:
        return str(begin if begin is not None else end)
    return f"{begin}–{end}"


def event_parts(event: dict | None) -> Iterable[dict]:
    """The event and its ``part`` sub-events, where Linked Art splits a production among roles."""
    if not event:
        return []
    return [event, *[p for p in as_list(event.get("part")) if isinstance(p, dict)]]
