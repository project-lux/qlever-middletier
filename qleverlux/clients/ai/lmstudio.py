"""LM Studio, over its native SDK.

For running the query builder against a local model. The SDK talks to the
LM Studio server (``localhost:1234`` unless ``QLMT_AI_TRANSLATE_API_ENDPOINT``
says otherwise) and loads the model named by ``QLMT_AI_TRANSLATE_MODEL`` on
first use, so the first request after a cold start is slow.

The system prompt goes in as the chat's initial prompt rather than being
prepended to the user text, so the model sees the same split it would from
Gemini's ``system_instruction``.

Generation is constrained to ``schema.RESPONSE_SCHEMA``. That matters more here
than elsewhere: unlike LM Studio's own OpenAI endpoint, ``respond()`` hands back
whatever the model emitted, so a reasoning model will happily spend the entire
token budget thinking in prose and never produce any JSON.
"""

from __future__ import annotations

from qleverlux.clients.ai.base import AiBackend, parse_model_json
from qleverlux.clients.ai.schema import RESPONSE_SCHEMA

try:
    import lmstudio
except ImportError:  # the package is optional
    lmstudio = None

#: How long a loaded model stays resident between requests, in seconds.
MODEL_TTL = 3600


class LmStudioBackend(AiBackend):
    name = "lmstudio"

    def __init__(self, settings, catalogue):
        super().__init__(settings, catalogue)
        if lmstudio is None:
            print("AI translate: the lmstudio package is not installed")
            return
        if not self.model:
            print("AI translate: lmstudio needs QLMT_AI_TRANSLATE_MODEL")
            return

        endpoint = settings.ai_translate_api_endpoint.strip()
        try:
            if endpoint:
                # the SDK wants host:port, so tolerate a full URL being given
                host = endpoint.split("://", 1)[-1].rstrip("/")
                self.lms_client = lmstudio.Client(api_host=host)
            else:
                self.lms_client = lmstudio.Client()
            # Client.llm is a session namespace, not a factory; .model() is what
            # returns a handle (and loads the model if it is not resident).
            self.client = self.lms_client.llm.model(self.model, ttl=MODEL_TTL)
        except Exception as e:
            print(f"AI translate: could not reach LM Studio: {e}")
            self.client = None

    def _prediction_config(self):
        settings = self.settings
        return {
            "temperature": settings.ai_translate_temperature,
            "topPSampling": settings.ai_translate_top_p,
            "maxTokens": settings.ai_translate_max_tokens,
        }

    def generate(self, prompt, which="build"):
        chat = lmstudio.Chat(self.system_prompt(which))
        chat.add_user_message(prompt)
        kwargs = {"config": self._prediction_config()}
        if self.settings.ai_translate_json_mode:
            # LM Studio's JSON mode is schema-based: a bare {"type": "json"} is
            # rejected. Constraining generation to the schema also stops a
            # reasoning model from spending the whole token budget thinking out
            # loud and never reaching the JSON.
            kwargs["response_format"] = {
                "type": "json",
                "jsonSchema": RESPONSE_SCHEMA,
            }
        try:
            result = self.client.respond(chat, **kwargs)
        except Exception as e:
            if not self.settings.ai_translate_json_mode:
                raise
            # not every model served by LM Studio supports constrained JSON
            print(f"AI translate: lmstudio rejected JSON mode ({e}); retrying without")
            result = self.client.respond(chat, config=self._prediction_config())

        parsed = getattr(result, "parsed", None)
        if isinstance(parsed, (dict, list)):
            return parsed
        return parse_model_json(getattr(result, "content", None), source="lmstudio")

    def close(self):
        client = getattr(self, "lms_client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
