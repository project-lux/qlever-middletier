"""What every AI translation backend has to provide.

A backend turns one prompt into the JSON the system prompts ask for: an
object with an ``options`` array, each entry carrying a ``scope`` and a
compact ``query`` tree. Which model produced it is not the service's problem,
so everything model-specific - auth, generation settings, how JSON mode is
requested - stays behind ``generate()``.

Two system prompts are in play and every backend has to honour both: "build"
turns a natural-language question into candidate queries, "improve" takes an
existing query plus a requested change and returns revised candidates.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

import ujson as json


def parse_model_json(text, source="model"):
    """Parse a model's reply, tolerating the wrappers models add anyway.

    The prompts demand bare JSON and Gemini honours that when asked for
    ``application/json``, but local and OpenAI-compatible models routinely wrap
    it in a markdown code fence or put a sentence around it. Returns None if
    nothing parseable comes back; the caller decides what that means.
    """
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    if text.startswith("```"):
        # ```json\n{...}\n```  ->  {...}
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # last resort: the outermost {...} or [...] in the reply
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue

    print(f"--- unparseable {source} response ---")
    print(text)
    sys.stdout.flush()
    return None


class AiBackend(ABC):
    """One model backend. Construct through ``clients.ai.build_ai_backend``."""

    #: Name used in settings and log messages.
    name = "unnamed"

    def __init__(self, settings, catalogue):
        self.settings = settings
        self.prompts = {
            "build": catalogue.system_prompt_translate,
            "improve": catalogue.system_prompt_improve,
        }
        self.model = settings.ai_translate_model
        self.client = None

    def system_prompt(self, which="build"):
        return self.prompts["improve" if which == "improve" else "build"]

    @property
    def enabled(self) -> bool:
        """Whether translation requests should be attempted at all."""
        return self.settings.ai_translate_enabled and self.client is not None

    @abstractmethod
    def generate(self, prompt, which="build"):
        """Run one prompt. Returns parsed JSON, or None if it was unusable."""

    def close(self):
        """Release anything the backend holds open."""


class NullBackend(AiBackend):
    """Stands in when translation is off or nothing is configured."""

    name = "none"

    def __init__(self, settings, catalogue, reason=""):
        super().__init__(settings, catalogue)
        self.reason = reason

    @property
    def enabled(self) -> bool:
        return False

    def generate(self, prompt, which="build"):
        return None
