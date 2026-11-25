"""Gemini on Vertex AI.

The settings here match ``lux-ai-query-builder/query-cli.py`` exactly -
temperature, top_p, token budget, the four safety categories switched off and
the thinking budget - because the system prompts were tuned against them.
"""

from __future__ import annotations

from qleverlux.clients.ai.base import AiBackend, parse_model_json, time_left

try:
    from google import genai
    from google.genai import types
except ImportError:  # the package is optional
    genai = None
    types = None

#: Safety categories the query builder turns off: a catalogue search should not
#: be refused because of what is in the collection.
SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_HARASSMENT",
)


class GeminiBackend(AiBackend):
    name = "gemini"

    def __init__(self, settings, catalogue):
        super().__init__(settings, catalogue)
        if genai is None:
            print("AI translate: google-genai is not installed")
            return
        self.configs = {
            which: self._config(self.system_prompt(which))
            for which in ("build", "improve")
        }
        self.client = genai.Client(
            vertexai=True,
            project=settings.ai_translate_project,
            location=settings.ai_translate_region or "global",
        )

    def _config(self, system_prompt):
        settings = self.settings
        return types.GenerateContentConfig(
            temperature=settings.ai_translate_temperature,
            top_p=settings.ai_translate_top_p,
            max_output_tokens=settings.ai_translate_max_tokens,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category=c, threshold="OFF")
                for c in SAFETY_CATEGORIES
            ],
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                thinking_budget=settings.ai_thinking_budget
            ),
            system_instruction=[types.Part.from_text(text=system_prompt)],
        )

    def generate(self, prompt, which="build", deadline=None):
        """One streamed generation, stopped at ``deadline``.

        The SDK had no timeout here at all. The time left goes in as the
        request's ``http_options.timeout`` (which also asks Vertex to stop, via
        ``X-Server-Timeout``), but on a stream that bounds each read, not the
        whole response, so the deadline is also checked between chunks. The
        SDK does not retry unless asked to.
        """
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        ]
        config = self.configs["improve" if which == "improve" else "build"]
        left = time_left(deadline)
        if left is not None:
            config = config.model_copy(
                update={"http_options": types.HttpOptions(timeout=max(1, int(left * 1000)))}
            )
        output = []
        stream = self.client.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )
        try:
            for chunk in stream:
                output.append(chunk.text or "")
                time_left(deadline)
        finally:
            stream.close()  # stopping early closes the response rather than reading it out
        return parse_model_json("".join(output), source="gemini")
