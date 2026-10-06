"""Any OpenAI-compatible chat completions endpoint.

Covers hosted OpenAI, a gateway in front of other models, vLLM, llama.cpp's
server, Ollama's compatibility layer and LM Studio's own OpenAI endpoint. The
only things that change are the base URL, the key and the model name:

    QLMT_AI_TRANSLATE_BACKEND=openai
    QLMT_AI_TRANSLATE_API_ENDPOINT=https://host/v1
    QLMT_AI_TRANSLATE_API_KEY=...
    QLMT_AI_TRANSLATE_MODEL=...

``response_format={"type": "json_object"}`` is sent when JSON mode is on, but
plenty of compatible servers reject the parameter outright, so a failure that
mentions it is retried once without it and JSON mode is then left off for the
rest of the process.
"""

from __future__ import annotations

from qleverlux.clients.ai.base import AiBackend, parse_model_json

try:
    import openai
except ImportError:  # the package is optional
    openai = None

#: Substrings that mean "this server does not support response_format".
_JSON_MODE_REJECTIONS = ("response_format", "json_object", "json mode")


class OpenAiCompatBackend(AiBackend):
    name = "openai"

    def __init__(self, settings, catalogue):
        super().__init__(settings, catalogue)
        if openai is None:
            print("AI translate: the openai package is not installed")
            return
        if not self.model:
            print("AI translate: openai backend needs QLMT_AI_TRANSLATE_MODEL")
            return

        self.json_mode = settings.ai_translate_json_mode
        kwargs = {
            # a local server usually wants no key, but the client insists on one
            "api_key": settings.ai_translate_api_key or "not-needed",
            "timeout": settings.ai_translate_timeout,
        }
        if settings.ai_translate_api_endpoint:
            kwargs["base_url"] = settings.ai_translate_api_endpoint
        try:
            self.client = openai.OpenAI(**kwargs)
        except Exception as e:
            print(f"AI translate: could not build the OpenAI client: {e}")
            self.client = None

    def _create(self, prompt, which, json_mode):
        settings = self.settings
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt(which)},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.ai_translate_temperature,
            "top_p": settings.ai_translate_top_p,
            "max_tokens": settings.ai_translate_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def generate(self, prompt, which="build"):
        try:
            response = self._create(prompt, which, self.json_mode)
        except Exception as e:
            if not self.json_mode or not any(
                s in str(e).lower() for s in _JSON_MODE_REJECTIONS
            ):
                raise
            print(f"AI translate: server rejected JSON mode ({e}); disabling it")
            self.json_mode = False
            response = self._create(prompt, which, False)

        if not response.choices:
            print("AI translate: no choices in the response")
            return None
        return parse_model_json(response.choices[0].message.content, source=self.model)

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
