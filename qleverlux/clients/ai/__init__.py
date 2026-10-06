"""AI query translation backends.

Pick one with ``QLMT_AI_TRANSLATE_BACKEND``:

``gemini``
    Gemini on Vertex AI. Needs ``QLMT_AI_TRANSLATE_PROJECT``; this is what the
    system prompts were tuned against.
``lmstudio``
    A local model through the LM Studio SDK.
``openai``
    Any OpenAI-compatible ``/v1/chat/completions`` endpoint, set with
    ``QLMT_AI_TRANSLATE_API_ENDPOINT`` and ``QLMT_AI_TRANSLATE_API_KEY``.

Left empty it is inferred, so existing deployments keep working: a model name
starting with ``gemini`` means Gemini, otherwise an endpoint means the
OpenAI-compatible backend, otherwise translation is off. LM Studio is never
inferred - it has to be asked for by name.
"""

from __future__ import annotations

from qleverlux.clients.ai.base import AiBackend, NullBackend, parse_model_json
from qleverlux.clients.ai.gemini import GeminiBackend
from qleverlux.clients.ai.lmstudio import LmStudioBackend
from qleverlux.clients.ai.openai_compat import OpenAiCompatBackend

BACKENDS = {
    GeminiBackend.name: GeminiBackend,
    LmStudioBackend.name: LmStudioBackend,
    OpenAiCompatBackend.name: OpenAiCompatBackend,
}

__all__ = [
    "BACKENDS",
    "AiBackend",
    "GeminiBackend",
    "LmStudioBackend",
    "NullBackend",
    "OpenAiCompatBackend",
    "build_ai_backend",
    "infer_backend",
    "parse_model_json",
]


def infer_backend(settings):
    """Which backend the settings imply when none is named."""
    if settings.ai_translate_model.startswith("gemini"):
        return GeminiBackend.name
    if settings.ai_translate_api_endpoint:
        return OpenAiCompatBackend.name
    return ""


def build_ai_backend(settings, catalogue):
    """The configured backend, or a disabled stand-in with the reason why."""
    if not settings.ai_translate_enabled:
        return NullBackend(settings, catalogue, "QLMT_AI_TRANSLATE is false")

    kind = settings.ai_translate_backend.strip().lower() or infer_backend(settings)
    if not kind:
        reason = "no QLMT_AI_TRANSLATE_BACKEND, and nothing to infer one from"
        print(f"AI translate disabled: {reason}")
        return NullBackend(settings, catalogue, reason)

    cls = BACKENDS.get(kind)
    if cls is None:
        reason = f"unknown backend {kind!r}, expected one of {sorted(BACKENDS)}"
        print(f"AI translate disabled: {reason}")
        return NullBackend(settings, catalogue, reason)

    backend = cls(settings, catalogue)
    if backend.enabled:
        print(f"AI translate: {backend.name} / {settings.ai_translate_model}")
    else:
        print(f"AI translate: {backend.name} configured but not usable")
    return backend
