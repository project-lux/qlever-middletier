"""Limits that keep one caller from taking the server down, and an overload from lasting.

* The QLever client's in-flight count once missed cancelled requests (``CancelledError`` is not an
  ``Exception``), creeping up until every sheddable query was shed; and it sent no ``timeout``, so
  QLever ran abandoned queries on to its own, longer, limit.
* HAL generation was exempt from shedding, so every uncached record pushed its 6-25 queries past
  the limit, and a probe that failed was taken for "no link" and cached for good.
* AI translation called a blocking SDK on the event loop: one request for a model that takes a
  minute stopped the process for that minute. Nor was there a cap on concurrency or input size.
* The facets ``page`` was unchecked: 0 became ``OFFSET -60``, reported as a QLever failure.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from qleverlux.clients.qlever import QLeverClient
from qleverlux.presentation.uris import UriRewriter
from qleverlux.services.ai_translate import AiTranslateService
from qleverlux.services.hal import HalService
from qleverlux.settings import load_settings


@pytest.fixture
def settings():
    return load_settings([])


# -- QLever client -----------------------------------------------------------


class HangingHttp:
    """An httpx stand-in whose POST waits until released, recording each form sent."""

    def __init__(self):
        self.forms = []
        self.release = asyncio.Event()

    async def post(self, endpoint, data, headers):
        self.forms.append(data)
        await self.release.wait()
        return SimpleNamespace(json=lambda: {"status": "OK", "res": [], "selected": []})


def test_a_cancelled_query_does_not_leave_the_in_flight_count_raised(settings):
    async def run():
        qlever = QLeverClient(settings)
        qlever.client = HangingHttp()
        task = asyncio.ensure_future(qlever.query("SELECT ?cancelled WHERE {}"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert qlever.open_requests == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        return qlever.open_requests

    assert asyncio.run(run()) == 0


def test_a_query_asks_qlever_to_stop_when_the_client_stops_waiting(settings):
    async def run():
        qlever = QLeverClient(settings)
        qlever.client = http = HangingHttp()
        http.release.set()
        res = await qlever.query("SELECT ?answered WHERE {}")
        return res, http.forms, qlever.open_requests

    res, forms, open_requests = asyncio.run(run())
    assert "error" not in res
    assert forms[0]["timeout"] == f"{settings.qlever_timeout}s"
    assert open_requests == 0


def test_results_are_asked_for_uncompressed(settings):
    """httpx asks for gzip/deflate/br by default and QLever obliges: it compressed every result
    and we decompressed it, on a local link, for results capped at ``send`` rows. The request
    goes through a real httpx client here, so its default header is what is being overridden."""
    import httpx

    seen = []

    def answer(request):
        seen.append(request.headers)
        return httpx.Response(200, json={"status": "OK", "res": [], "selected": []})

    async def run():
        qlever = QLeverClient(settings)
        qlever.client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
        try:
            return await qlever.query("SELECT ?uncompressed WHERE {}")
        finally:
            await qlever.client.aclose()

    assert "error" not in asyncio.run(run())
    assert seen[0]["accept-encoding"] == "identity"
    assert seen[0]["accept"] == "application/qlever-results+json"


def test_a_plain_text_refusal_from_qlever_reaches_the_log_and_the_count_recovers(settings, capsys):
    refusal = "User submitted timeout was higher than what is currently allowed by this instance (60s)."

    class RefusingHttp:
        async def post(self, endpoint, data, headers):
            def not_json():
                raise ValueError("Expecting value: line 1 column 1 (char 0)")

            return SimpleNamespace(json=not_json, text=refusal, status_code=400)

    async def run():
        qlever = QLeverClient(settings)
        qlever.client = RefusingHttp()
        return await qlever.query("SELECT ?refused WHERE {}"), qlever.open_requests

    res, open_requests = asyncio.run(run())
    assert refusal in res["error"]
    assert res["status"] == 400
    assert open_requests == 0


# -- HAL ---------------------------------------------------------------------

UUID = "00000000-0000-0000-0000-000000000007"


class ScriptedQLever:
    """Answers each query by the first marker it contains; records how it was asked."""

    def __init__(self, answers):
        self.answers = answers
        self.sheddable = []

    async def query(self, sparql, sheddable=True):
        self.sheddable.append(sheddable)
        for marker, answer in self.answers.items():
            if marker in sparql:
                return answer
        return {"results": [], "total": 0}


class MemoryHalCache:
    def __init__(self):
        self.stored = {}

    def get(self, key):
        return self.stored.get(key)

    async def put(self, key, links):
        self.stored[key] = links


FOUND = {"results": [[3]], "total": 1}
FAILED = {"results": [], "total": 0, "error": "Too many open requests", "status": 504}


def hal_service(settings, answers):
    catalogue = SimpleNamespace(
        sparql_hal_queries={
            "item": {
                "lux:countA": "SELECT countA URI-HERE",
                "lux:countB": "SELECT countB URI-HERE",
                "lux:listC": {
                    "scope": "item", "relatedList": "listC", "template": "related?uri={id}",
                },
            }
        },
        hal_related_list_tests={"item": {"listC": ["ASK probe1 V_TARGET_URI", "ASK probe2 V_TARGET_URI"]}},
        hal_queries={
            "lux:countA": {"queryName": "qA", "template": "search?q={q}"},
            "lux:countB": {"queryName": "qB", "template": "search?q={q}"},
        },
        queries={"qA": {"id": "URI-HERE"}, "qB": {"id": "URI-HERE"}},
    )
    uris = UriRewriter(settings.data_uri, settings.mt_uri, settings.record_path)
    qlever = ScriptedQLever(answers)
    cache = MemoryHalCache()
    return HalService(settings, catalogue, qlever, cache, uris), qlever, cache


def test_hal_links_with_every_query_answered_are_cached(settings):
    hal, qlever, cache = hal_service(settings, {"countA": FOUND, "probe2": FOUND})
    links = asyncio.run(hal.links("object", UUID))
    assert set(links) == {"lux:countA", "lux:listC"}
    assert cache.stored[UUID] == links


def test_hal_links_are_served_but_not_cached_when_a_query_failed(settings):
    hal, _, cache = hal_service(settings, {"countA": FOUND, "countB": FAILED})
    links = asyncio.run(hal.links("object", UUID))
    assert set(links) == {"lux:countA"}
    assert cache.stored == {}


def test_a_related_list_miss_after_a_failed_probe_is_not_cached(settings):
    hal, _, cache = hal_service(settings, {"probe1": FAILED})
    asyncio.run(hal.links("object", UUID))
    assert cache.stored == {}


def test_a_related_list_hit_after_a_failed_probe_is_conclusive(settings):
    hal, _, cache = hal_service(settings, {"probe1": FAILED, "probe2": FOUND})
    links = asyncio.run(hal.links("object", UUID))
    assert "lux:listC" in links
    assert cache.stored[UUID] == links


def test_hal_queries_are_sheddable(settings):
    hal, qlever, _ = hal_service(settings, {})
    asyncio.run(hal.links("object", UUID))
    assert qlever.sheddable and all(qlever.sheddable)


# -- AI translation ----------------------------------------------------------

ANSWER = {"options": [{"scope": "item", "natural": "paintings", "query": {"f": "text", "v": "paintings"}}]}


class BlockingBackend:
    """A backend whose generate() blocks its thread until released, as the real SDKs do."""

    enabled = True

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls = 0

    def generate(self, prompt, which, deadline=None):
        self.calls += 1
        self.started.set()
        assert self.release.wait(5), "never released"
        return ANSWER


def ai_service(settings, backend, **overrides):
    for name, value in overrides.items():
        setattr(settings, name, value)
    return AiTranslateService(settings, catalogue=None, backend=backend)


async def _wait_started(backend):
    while not backend.started.is_set():
        await asyncio.sleep(0.005)


def test_a_translation_does_not_block_the_event_loop(settings):
    backend = BlockingBackend()
    ai = ai_service(settings, backend)

    async def run():
        task = asyncio.ensure_future(ai.translate("paintings"))
        await _wait_started(backend)
        # the loop is still free while the model "thinks"
        ticks = 0
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks += 1
        backend.release.set()
        return ticks, await task

    ticks, response = asyncio.run(run())
    assert ticks == 3
    assert response.status_code == 200


def test_translations_past_the_limit_are_a_503(settings):
    backend = BlockingBackend()
    ai = ai_service(settings, backend, ai_translate_max_concurrent=1)

    async def run():
        first = asyncio.ensure_future(ai.translate("paintings"))
        await _wait_started(backend)
        second = await ai.translate("sculpture")
        backend.release.set()
        return second, await first

    second, first = asyncio.run(run())
    assert second.status_code == 503
    assert first.status_code == 200
    assert backend.calls == 1


def test_a_slot_is_held_until_an_abandoned_translation_finishes(settings):
    backend = BlockingBackend()
    ai = ai_service(settings, backend, ai_translate_max_concurrent=1)

    async def run():
        first = asyncio.ensure_future(ai.translate("paintings"))
        await _wait_started(backend)
        first.cancel()  # the client went away; the thread carries on
        await asyncio.sleep(0.01)
        while_running = (await ai.translate("sculpture")).status_code
        backend.release.set()
        for _ in range(100):  # let the thread finish and its callback release the slot
            if not ai._slots.locked():
                break
            await asyncio.sleep(0.01)
        backend.started.clear()
        after = (await ai.translate("sculpture")).status_code
        return while_running, after

    assert asyncio.run(run()) == (503, 200)


@pytest.mark.parametrize("q, prev", [("x" * 1001, ""), ("paintings", "{" + " " * 20000 + "}")], ids=["q", "prevQuery"])
def test_overlong_input_is_a_400_and_never_reaches_the_model(settings, q, prev):
    backend = BlockingBackend()
    ai = ai_service(settings, backend)
    response = asyncio.run(ai.translate(q, prev))
    assert response.status_code == 400
    assert backend.calls == 0


# -- facets ------------------------------------------------------------------


@pytest.mark.parametrize("page", [0, -1])
def test_a_facet_page_below_one_is_rejected(page):
    from fastapi.testclient import TestClient

    from qleverlux.app import create_app

    async def nothing():
        pass

    mt = SimpleNamespace(facets=None, start=nothing, aclose=nothing)
    with TestClient(create_app(middletier=mt)) as client:
        response = client.get(f"/api/facets/item?name=itemTypeId&q=%7B%7D&page={page}")
    assert response.status_code == 422
