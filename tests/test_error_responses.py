"""Failures tell a client what went wrong, not how the deployment is put together.

A failed query once returned QLever's whole result object, whose message can quote the SPARQL
and internal addresses (the vector sidecar's, a SERVICE's); a record cache failure and an AI
backend failure returned their exception text (paths, hosts, a provider's error). They now
return a plain message with the same status, and the detail goes to the log - and to the
response, as "detail", only on a deployment with QLMT_DEBUG_ERRORS.

A complete name search, an exact match on the whole name, stripped double quotes but not a
trailing backslash or a newline, which broke the query; it is now escaped like any literal.
"""

import json
import urllib.parse
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("luxql")

SECRET = "SERVICE <http://127.0.0.1:8081/sparql> failed; /data/lmdbs/lux-recap/vectors.lmdb"


class FailingQLever:
    def __init__(self, status):
        self.status = status

    async def query(self, sparql, sheddable=True, fmt=None):
        return {"results": [], "total": 0, "error": SECRET, "status": self.status}


class FailingCache:
    async def fetch(self, identifier, scope):
        raise OSError(SECRET)


class FailingBackend:
    enabled = True

    def generate(self, prompt, which, deadline=None):
        raise ConnectionError(SECRET)


@pytest.fixture(scope="module")
def catalogue():
    from qleverlux.query.catalogue import QueryCatalogue
    from qleverlux.settings import load_settings

    return QueryCatalogue(load_settings([]))


def client_for(catalogue, debug, status=400):
    from fastapi.testclient import TestClient

    from qleverlux.app import create_app
    from qleverlux.presentation.uris import UriRewriter
    from qleverlux.services.ai_translate import AiTranslateService
    from qleverlux.services.records import RecordService
    from qleverlux.services.search import SearchService
    from qleverlux.settings import load_settings

    settings = load_settings(["--debug-errors"] if debug else [])
    uris = UriRewriter(settings.data_uri, settings.mt_uri, settings.record_path)

    async def nothing():
        pass

    mt = SimpleNamespace(
        settings=settings,
        search=SearchService(settings, catalogue, FailingQLever(status), uris),
        records=RecordService(settings, FailingCache(), None, uris),
        ai=AiTranslateService(settings, catalogue, FailingBackend()),
        start=nothing, aclose=nothing,
    )
    return TestClient(create_app(middletier=mt))


SEARCH = "/api/search/item?q=" + urllib.parse.quote(json.dumps({"text": "x"}))
RECORD = f"/data/object/{uuid.UUID(int=7)}"
AI = "/api/ai-translate?q=paintings"


@pytest.mark.parametrize("path, status, message", [
    (SEARCH, 400, "The query could not be run."),
    (RECORD, 500, "The record could not be read."),
    (AI, 502, "The AI translation service is unavailable."),
])
def test_a_failure_says_what_went_wrong_and_nothing_about_the_deployment(catalogue, path, status, message):
    with client_for(catalogue, debug=False) as client:
        response = client.get(path)
    assert response.status_code == status
    assert response.json() == {"error": message}
    assert "127.0.0.1" not in response.text and "lmdbs" not in response.text


def test_a_timeout_keeps_its_status_and_says_to_try_again(catalogue):
    with client_for(catalogue, debug=False, status=504) as client:
        response = client.get(SEARCH)
    assert response.status_code == 504
    assert "try again" in response.json()["error"] and "detail" not in response.json()


@pytest.mark.parametrize("path", [SEARCH, RECORD, AI])
def test_a_debug_deployment_adds_the_detail(catalogue, path):
    with client_for(catalogue, debug=True) as client:
        body = client.get(path).json()
    assert "127.0.0.1:8081" in body["detail"] and body["error"]


@pytest.mark.parametrize("name, literal", [
    ("o'brien", '"o\'brien"'),
    ('the "rock"', '"the \\"rock\\""'),  # kept, escaped: names are stored with their quotes
    ("trailing\\", '"trailing\\\\"'),  # once ended the literal's closing quote
    ("two\nlines", '"two\\nlines"'),  # once broke the literal
])
def test_a_complete_name_is_escaped_like_any_literal(catalogue, name, literal):
    parsed = catalogue.json_reader.read({"name": name, "_complete": True}, "agent")
    sparql = catalogue.translator.translate_search(parsed, scope="agent").get_text()
    assert f"lux:agentName {literal}" in sparql
