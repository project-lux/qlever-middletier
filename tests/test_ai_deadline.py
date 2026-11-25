"""Every AI translation ends by QLMT_AI_TRANSLATE_TIMEOUT, retry included.

The translation runs on a worker thread that holds a concurrency slot until it finishes, and a
thread cannot be stopped from outside, so each backend has to stop itself. Before this only the
OpenAI-compatible client had a timeout, and per request, with the SDK's own two retries on top;
Gemini had none, and LM Studio only gave up after a minute with no message - never, while the
model kept generating. A stalled backend could hold every slot and answer everyone "busy".
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from qleverlux.clients.ai.base import time_left
from qleverlux.services.ai_translate import AiTranslateService
from qleverlux.settings import load_settings

ANSWER = {"options": [{"scope": "item", "natural": "paintings", "query": {"f": "text", "v": "paintings"}}]}
CATALOGUE = SimpleNamespace(system_prompt_translate="build prompt", system_prompt_improve="improve prompt")


@pytest.fixture
def settings():
    s = load_settings([])
    s.ai_translate_model = "test-model"
    return s


def soon(seconds):
    return time.monotonic() + seconds


# -- the service -----------------------------------------------------------------


class RecordingBackend:
    enabled = True

    def __init__(self, answers):
        self.answers = list(answers)
        self.deadlines = []

    def generate(self, prompt, which, deadline=None):
        self.deadlines.append(deadline)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(deadline)
        return answer


def translate(settings, backend, **overrides):
    for name, value in overrides.items():
        setattr(settings, name, value)
    return asyncio.run(AiTranslateService(settings, None, backend).translate("paintings"))


def test_one_deadline_covers_the_attempt_and_its_retry(settings):
    backend = RecordingBackend([None, ANSWER])
    started = time.monotonic()
    response = translate(settings, backend, ai_translate_timeout=30)
    assert response.status_code == 200
    first, retry = backend.deadlines
    assert first == retry
    assert started + 29 < first <= time.monotonic() + 30


def test_no_retry_once_the_deadline_has_passed(settings):
    def slow_and_useless(deadline):
        time.sleep(1.1)
        return None

    backend = RecordingBackend([slow_and_useless, ANSWER])
    response = translate(settings, backend, ai_translate_timeout=1)
    assert response.status_code == 502
    assert len(backend.deadlines) == 1


def test_a_timeout_is_a_504_and_any_other_failure_a_502(settings):
    assert translate(settings, RecordingBackend([TimeoutError("late")])).status_code == 504
    assert translate(settings, RecordingBackend([ConnectionError("refused")])).status_code == 502


def test_a_backend_that_stops_at_the_deadline_frees_its_slot(settings):
    class Dawdling:
        enabled = True

        def generate(self, prompt, which, deadline=None):
            started = time.monotonic()
            while True:  # a model that never finishes, as a streaming backend sees it
                time_left(deadline)
                if time.monotonic() - started > 5:  # given no deadline: fail, don't hang the suite
                    raise RuntimeError("never stopped")
                time.sleep(0.02)

    service = AiTranslateService(settings, None, Dawdling())
    settings.ai_translate_timeout = 1
    settings.ai_translate_max_concurrent = 1

    async def run():
        first = await service.translate("paintings")
        await asyncio.sleep(0.05)  # the slot is released by a callback on the loop
        return first.status_code, service._slots.locked()

    started = time.monotonic()
    assert asyncio.run(run()) == (504, False)
    assert time.monotonic() - started < 3


def test_zero_means_no_deadline(settings):
    backend = RecordingBackend([ANSWER])
    assert translate(settings, backend, ai_translate_timeout=0).status_code == 200
    assert backend.deadlines == [None]


# -- OpenAI-compatible -----------------------------------------------------------


def test_openai_gets_the_time_left_and_does_not_retry_on_its_own(settings):
    pytest.importorskip("openai")
    from qleverlux.clients.ai.openai_compat import OpenAiCompatBackend

    backend = OpenAiCompatBackend(settings, CATALOGUE)
    assert backend.client.max_retries == 0
    calls = []
    message = SimpleNamespace(content='{"options": []}')
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: calls.append(kw) or SimpleNamespace(choices=[SimpleNamespace(message=message)]))))
    backend.generate("paintings", deadline=soon(10))
    assert 9 < calls[0]["timeout"] <= 10
    with pytest.raises(TimeoutError):
        backend.generate("paintings", deadline=soon(-1))
    assert len(calls) == 1


# -- Gemini ---------------------------------------------------------------------


def gemini_backend(settings, chunks, gap):
    pytest.importorskip("google.genai")
    from qleverlux.clients.ai.gemini import GeminiBackend

    backend = GeminiBackend.__new__(GeminiBackend)  # no Vertex client: the stream is faked
    backend.settings = settings
    backend.prompts = {"build": "build prompt", "improve": "improve prompt"}
    backend.model = "test-model"
    backend.configs = {w: backend._config(backend.prompts[w]) for w in ("build", "improve")}
    seen = SimpleNamespace(timeout=None, closed=False, sent=0)

    def stream(model, contents, config):
        seen.timeout = config.http_options.timeout if config.http_options else None
        try:
            for text in chunks:
                time.sleep(gap)
                seen.sent += 1
                yield SimpleNamespace(text=text)
        finally:
            seen.closed = True

    backend.client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream))
    return backend, seen


def test_gemini_passes_the_time_left_as_the_request_timeout(settings):
    backend, seen = gemini_backend(settings, ['{"options": ', "[]}"], 0)
    assert backend.generate("paintings", deadline=soon(10)) == {"options": []}
    assert 9000 < seen.timeout <= 10000


def test_gemini_stops_reading_a_stream_at_the_deadline(settings):
    backend, seen = gemini_backend(settings, ["x"] * 100, 0.05)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.generate("paintings", deadline=soon(0.3))
    assert time.monotonic() - started < 1
    assert seen.closed and seen.sent < 100


# -- LM Studio ------------------------------------------------------------------


class FakePrediction:
    """respond_stream()'s result: fragments until cancelled, prompt progress first."""

    def __init__(self, fragments, gap, progress_steps, on_progress):
        self.fragments, self.gap = fragments, gap
        self.progress_steps, self.on_progress = progress_steps, on_progress
        self.cancelled = False
        self.sent = 0

    def __iter__(self):
        for step in range(self.progress_steps):
            time.sleep(self.gap)
            self.on_progress(step / self.progress_steps)
            if self.cancelled:
                return
        for fragment in self.fragments:
            if self.cancelled:
                return
            time.sleep(self.gap)
            self.sent += 1
            yield fragment

    def cancel(self):
        self.cancelled = True

    def result(self):
        return SimpleNamespace(parsed=ANSWER, content=None)


def lmstudio_backend(settings, fragments=(), gap=0.0, progress_steps=0):
    pytest.importorskip("lmstudio")
    from qleverlux.clients.ai.lmstudio import LmStudioBackend

    backend = LmStudioBackend.__new__(LmStudioBackend)  # no LM Studio server: the model is faked
    backend.settings = settings
    backend.prompts = {"build": "build prompt", "improve": "improve prompt"}
    backend.model = "test-model"
    made = []

    def respond_stream(chat, on_prompt_processing_progress=None, **kwargs):
        made.append(FakePrediction(list(fragments), gap, progress_steps, on_prompt_processing_progress))
        return made[-1]

    backend.client = SimpleNamespace(respond_stream=respond_stream)
    return backend, made


def test_lmstudio_answers_within_the_deadline(settings):
    backend, _ = lmstudio_backend(settings, fragments=["a", "b"])
    assert backend.generate("paintings", deadline=soon(10)) == ANSWER


def test_lmstudio_cancels_a_generation_that_runs_past_the_deadline(settings):
    backend, made = lmstudio_backend(settings, fragments=["x"] * 100, gap=0.05)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        backend.generate("paintings", deadline=soon(0.3))
    assert time.monotonic() - started < 1
    # cancelled once, not retried without JSON mode as an ordinary failure would be
    assert len(made) == 1 and made[0].cancelled and made[0].sent < 100


def test_lmstudio_cancels_during_a_long_prompt_before_any_token(settings):
    backend, made = lmstudio_backend(settings, fragments=["x"], gap=0.05, progress_steps=100)
    with pytest.raises(TimeoutError):
        backend.generate("paintings", deadline=soon(0.3))
    assert made[0].cancelled and made[0].sent == 0
