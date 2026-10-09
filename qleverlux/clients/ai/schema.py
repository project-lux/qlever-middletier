"""The JSON schema the system prompts describe.

Backends that support schema-constrained output (LM Studio's structured
prediction, OpenAI's ``json_schema`` response format) can be handed this so the
model cannot produce a shape the service will reject. It mirrors the contract
in ``config/system-prompt-translate.txt``: an ``options`` array, each entry a
scope plus a recursive query tree.

The tree is recursive, so the schema uses ``$defs``/``$ref``. Not every server
can compile that - LM Studio converts schemas to GBNF grammars - so backends
must treat a rejection as "no schema available" rather than an error.
"""

from __future__ import annotations

from qleverlux.enums import scopeEnum

#: One node of the compact query tree the model emits.
QUERY_NODE = {
    "type": "object",
    "properties": {
        "f": {"type": "string"},
        "v": {"type": ["string", "number", "boolean"]},
        "c": {"type": "string", "enum": [">", ">=", "=", "<", "<="]},
        "p": {"type": "array", "items": {"$ref": "#/$defs/query"}},
        "r": {"$ref": "#/$defs/query"},
        "d": {"type": "boolean"},
    },
    "required": ["f"],
}

#: The whole reply: the options array the prompts ask for.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "natural": {"type": "string"},
                    "parsed": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": [s.value for s in scopeEnum],
                    },
                    "query": {"$ref": "#/$defs/query"},
                },
                "required": ["scope", "query"],
            },
        }
    },
    "required": ["options"],
    "$defs": {"query": QUERY_NODE},
}
