"""Natural-language query translation.

The model answers with an ``options`` array; each option carries a ``scope``
and a compact query tree - ``f`` field, ``v`` value, ``c`` comparator, ``p``
boolean children, ``r`` relationship child - which this expands into the LUX
JSON queries the rest of the API speaks, coercing the values whose types the
model cannot be trusted to get right.

Two flows, matching the system prompts and ``lux-ai-query-builder``:

* **build** - a question in, candidate queries out.
* **improve** - an existing query plus a requested change in, revised
  candidates out. The prompt format is fixed by the improve system prompt and
  is built by ``build_improve_prompt``.

The reference CLI returns only the first option; this returns all of them, with
the model's own natural-language reading of each, so a caller can offer the
alternatives the model was asked to produce and label them.
"""

from __future__ import annotations

import ujson as json
from fastapi.responses import JSONResponse

from qleverlux.enums import scopeEnum

#: Fields whose values must be coerced out of whatever the model produced.
_FLOAT_FIELDS = ("height", "width", "depth", "dimension")
_INT_FIELDS = ("hasDigitalImage",)

#: The scopes a relationship can lead into. A term's "relation" is either one
#: of these or a leaf type ("text", "date", "float", "boolean").
_SCOPES = frozenset(s.value for s in scopeEnum)


def build_improve_prompt(previous_query, change):
    """The improve prompt, in the exact shape the improve system prompt expects."""
    return f"Query: \n{json.dumps(previous_query, indent=2)}\n\nImprovement: {change}"


class AiTranslateService:
    def __init__(self, settings, catalogue, backend):
        self.settings = settings
        self.catalogue = catalogue
        self.backend = backend

    def relation_scope(self, field, scope):
        """The scope a relationship leads into, or ``scope`` if it leads nowhere.

        Only a relation naming one of the seven scopes is a scope change; a
        term whose relation is a leaf type leaves the scope alone. The old code
        indexed a global ``scopes`` mapping that was never defined, so every
        query containing a relationship raised ``NameError`` here.
        """
        if scope is None:
            return scope
        terms = self.catalogue.lux_config.lux_config["terms"].get(scope, {})
        relation = terms.get(field, {}).get("relation")
        return relation if relation in _SCOPES else scope

    def post_process(self, query, scope=None):
        """The model's compact tree -> a LUX JSON query."""
        new = {}
        if "p" in query:
            # BOOL
            new[query["f"]] = [self.post_process(x, scope) for x in query["p"]]
        elif "r" in query:
            # Change scope
            scope = self.relation_scope(query["f"], scope)
            new[query["f"]] = self.post_process(query["r"], scope)
        else:
            if query["f"] in _FLOAT_FIELDS:
                query["v"] = float(query["v"])
            elif query["f"] in _INT_FIELDS:
                query["v"] = int(query["v"])
            elif query["f"].lower() == "recordtype":
                query["v"] = query["v"].lower()
            new[query["f"]] = query["v"]
            if "c" in query:
                new["_comp"] = query["c"]
        return new

    def to_lux(self, js):
        """Expand the model's options into the disambiguation list the API returns.

        One entry per option, each carrying the model's own description of what
        it took the question to mean alongside the query itself::

            [{"natural": ..., "parsed": ..., "query": {..., "_scope": ...}}]

        ``_scope`` goes *inside* the query because that is where the client
        reads it from, and strips it before running the search. ``natural`` is
        produced by both prompts; ``parsed`` only by the improve prompt, so it
        defaults to empty rather than being left out.
        """
        if type(js) is list:
            js = {"options": js}
        try:
            options = []
            for option in js["options"]:
                scope = option["scope"]
                query = self.post_process(option["query"], scope)
                query["_scope"] = scope
                options.append(
                    {
                        "natural": option.get("natural", ""),
                        "parsed": option.get("parsed", ""),
                        "query": query,
                    }
                )
            return options
        except Exception:
            print("Failed to process:")
            print(json.dumps(js, indent=2))
            raise

    def _attempt(self, prompt, which):
        """One round trip. Returns (options, error); options is None if unusable.

        A backend exception is re-raised: a connection refused or a timeout will
        not come good on a second try, and retrying a timeout just doubles how
        long the caller waits.
        """
        qjs = self.backend.generate(prompt, which)
        if qjs is None:
            return None, "no usable response from the AI backend"
        print(qjs)
        try:
            options = self.to_lux(qjs)
        except Exception as e:
            return None, f"could not read the AI response: {e}"
        if not options:
            return None, "the AI backend returned no options"
        return options, None

    async def translate(self, q, prevQuery=""):
        """Translate ``q``, or improve ``prevQuery`` according to ``q``.

        A model at temperature 0.8 occasionally answers with an empty
        ``options`` array, or with something that will not parse, so one retry
        is allowed before giving up - the same single retry the reference proxy
        did. Returning an empty list instead would be worse than an error: the
        client indexes ``[0]`` when there is only one option.
        """
        if not self.backend.enabled:
            return JSONResponse(content={}, status_code=404)
        if not q:
            return JSONResponse(content={}, status_code=400)

        which = "build"
        prompt = q
        if prevQuery:
            try:
                previous = json.loads(prevQuery)
            except Exception:
                return JSONResponse(
                    content={"error": "prevQuery is not valid JSON"}, status_code=400
                )
            which = "improve"
            prompt = build_improve_prompt(previous, q)

        try:
            options, error = self._attempt(prompt, which)
            if options is None:
                print(f"AI translate: {error}; retrying once")
                options, error = self._attempt(prompt, which)
        except Exception as e:
            print(f"AI translate failed: {e}")
            return JSONResponse(
                content={"error": f"AI backend failed: {e}"}, status_code=502
            )

        if options is None:
            return JSONResponse(content={"error": error}, status_code=502)
        return JSONResponse(content=options)
