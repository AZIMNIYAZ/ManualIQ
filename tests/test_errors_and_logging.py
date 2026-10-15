import json
import logging
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.api.search import get_aliases  # noqa: E402
from app.core import logging as app_logging  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.errors import register_secret  # noqa: E402
from app.core.logging import JsonFormatter, setup_logging  # noqa: E402
from app.database.qdrant import get_qdrant  # noqa: E402
from app.main import app  # noqa: E402
from app.models.chat import ChatRequest  # noqa: E402
from app.services import chat_service  # noqa: E402
from app.services import generation_service as gen  # noqa: E402
from app.services.context_builder import ContextSource  # noqa: E402
from app.services.embedding_service import get_embedder  # noqa: E402
from app.services.retrieval_service import InvalidBrand, VectorDatabaseUnavailable  # noqa: E402
from tests.test_chat import NETGEAR, FakeGenerator, result  # noqa: E402

FAKE_KEY = "xai-TESTKEY0123456789abcdefghijkl"


class Captured(logging.Handler):
    """Collects formatted JSON log lines (as the real handlers would write them)."""

    def __init__(self):
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))

    def events(self, name):
        return [json.loads(line) for line in self.lines if json.loads(line)["message"] == name]


@pytest.fixture
def logs():
    handler = Captured()
    root = logging.getLogger()
    root.addHandler(handler)
    yield handler
    root.removeHandler(handler)


def raising(error):
    def dependency():
        raise error
    return dependency


@pytest.fixture
def api():
    # No app startup in these tests: stand-ins for the shared clients (tests override what they need)
    app.dependency_overrides.update({get_qdrant: lambda: None, get_embedder: lambda: None, get_aliases: lambda: ()})
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


def assert_error(r, status, code):
    body = r.json()
    assert r.status_code == status, body
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message", "request_id", "details"}
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == r.headers["X-Request-ID"]  # body and header always match
    return body["error"]


# ---------------------------------------------------------------- one error contract

def test_validation_error_has_structured_details(api):
    err = assert_error(api.post("/api/v1/search", json={"query": "  ", "top_k": 50}), 422, "validation_error")
    assert {"field": "body.query", "problem": "query must not be blank"} in err["details"]
    assert any(d["field"] == "body.top_k" for d in err["details"])


def test_malformed_json_uses_the_same_shape(api):
    r = api.post("/api/v1/search", content="{not json", headers={"Content-Type": "application/json"})
    err = assert_error(r, 422, "validation_error")
    assert err["details"][0]["problem"] == "request body is not valid JSON"


def test_404_and_405_stay_distinct(api):
    assert_error(api.get("/api/v1/nope"), 404, "route_not_found")
    r = api.get("/api/v1/chat")
    err = assert_error(r, 405, "method_not_allowed")
    assert err["details"] is None and "POST" in r.headers["Allow"]


def test_service_errors_and_retry_after(api):
    cases = [
        (InvalidBrand("Unknown brand ['linksys']"), 422, "invalid_brand", None),
        (VectorDatabaseUnavailable("Vector database unavailable"), 503, "vector_db_unavailable", "5"),
        (gen.LLMRateLimited("slow down"), 429, "llm_rate_limited", "10"),
        (gen.LLMRateLimited("slow down", retry_after=7), 429, "llm_rate_limited", "7"),  # xAI's own hint wins
        (gen.LLMAuthenticationError("check XAI_API_KEY"), 502, "llm_authentication_failed", None),
        (gen.GenerationNotConfigured("not configured"), 503, "generation_not_configured", None),
    ]
    for error, status, code, retry in cases:
        app.dependency_overrides[get_aliases] = raising(error)
        r = api.post("/api/v1/search", json={"query": "reset"})
        err = assert_error(r, status, code)
        assert err["details"] is None and r.headers.get("Retry-After") == retry, code


# ---------------------------------------------------------------- unexpected crashes

def test_crash_is_generic_traceable_and_logged(api, logs):
    """Injected through a dependency override: production code is not broken for the test."""
    app.dependency_overrides[get_qdrant] = raising(RuntimeError(f"boom at C:\\secret\\path.py with {FAKE_KEY}"))
    app.dependency_overrides[get_embedder] = lambda: None
    app.dependency_overrides[get_aliases] = lambda: ()
    r = api.post("/api/v1/search", json={"query": "reset"}, headers={"X-Request-ID": "crash-test-1"})
    err = assert_error(r, 500, "internal_error")
    assert err["message"] == "An unexpected error occurred" and err["request_id"] == "crash-test-1"
    assert "boom" not in r.text and "secret" not in r.text and FAKE_KEY not in r.text
    crash = logs.events("unhandled_error")
    assert crash and crash[0]["request_id"] == "crash-test-1" and "RuntimeError: boom" in crash[0]["exception"]
    assert all(FAKE_KEY not in line for line in logs.lines)  # the key-shaped text is redacted from logs too


def test_registered_secret_is_redacted_from_errors_and_logs(api, logs):
    register_secret("my-very-secret-api-key-value")
    app.dependency_overrides[get_aliases] = raising(gen.LLMUnavailable("bad key my-very-secret-api-key-value"))
    r = api.post("/api/v1/search", json={"query": "reset"})
    assert assert_error(r, 502, "llm_unavailable")["message"] == "bad key [REDACTED]"
    assert all("my-very-secret-api-key-value" not in line for line in logs.lines)


@pytest.mark.parametrize("sent, kept", [("abc-123_X", True), ("bad id\nwith newline", False), ("x" * 65, False)])
def test_client_request_id_is_only_kept_if_safe(api, sent, kept):
    r = api.get("/api/v1/nope", headers={"X-Request-ID": sent})
    assert (r.headers["X-Request-ID"] == sent) is kept


# ---------------------------------------------------------------- chat_completed event

@pytest.fixture
def fake_search(monkeypatch):
    from app.models.search import SearchFilters, SearchResponse
    response = SearchResponse(query="q", collection="rag_350_60", query_tokens=20, query_truncated=False,
                              filters=SearchFilters(brands=["netgear"], brand_source="detected", include_regulatory=False),
                              results=NETGEAR, timings_ms={"checks": 1.0, "embed": 60.0, "search": 8.0, "total": 70.0})
    monkeypatch.setattr(chat_service.retrieval_service, "search", lambda *a, **k: response)


@pytest.mark.parametrize("log_user_text", [False, True])
def test_one_complete_chat_event_with_privacy(fake_search, logs, log_user_text):
    settings = Settings(_env_file=None, log_user_text=log_user_text)
    chat_service.answer(None, None, FakeGenerator("Click Add [1]."), settings,
                        ChatRequest(message="How do I set up port forwarding?"), ())
    events = logs.events("chat_completed")
    assert len(events) == 1  # exactly one record per request, never duplicated
    e = events[0]
    for key in ("session_id", "user_message", "retrieved", "cited", "found_in_documents", "warnings", "tokens",
                "timings_ms", "models", "versions", "brands", "brand_source", "history_used"):
        assert key in e, key
    assert set(e["timings_ms"]) == {"rewrite", "checks", "embed", "vector_search", "generate", "total"}
    assert e["answer"] == {"chars": len("Click Add [1].")}  # answers are never logged in full
    expected = "How do I set up port forwarding?" if log_user_text else {"chars": 32}
    assert e["user_message"] == expected


def test_log_file_written_once_and_rotated(tmp_path, monkeypatch):
    log_file = tmp_path / "logs" / "app.jsonl"
    try:
        setup_logging("INFO", str(log_file))
        setup_logging("INFO", str(log_file))  # calling twice must not duplicate records
        logging.getLogger("app").info("hello", extra={"fields": {"n": 1}})
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["n"] == 1

        monkeypatch.setattr(app_logging, "LOG_FILE_MAX_BYTES", 2000)
        setup_logging("INFO", str(log_file))
        for i in range(100):
            logging.getLogger("app").info("filler", extra={"fields": {"i": i, "pad": "x" * 50}})
        assert (tmp_path / "logs" / "app.jsonl.1").exists()  # rotated, at most 3 backups kept
        assert not (tmp_path / "logs" / "app.jsonl.4").exists()
    finally:
        setup_logging("INFO", None)  # release the file (Windows) and restore console logging


# ---------------------------------------------------------------- citation merging

def test_citations_merged_by_document_and_pages_in_first_citation_order():
    a1 = result(1, "first chunk on page 139", brand="netgear", pages=(139, 139))
    other_manual = result(2, "tenda text on page 139", brand="tenda", pages=(139, 139))
    a2 = result(3, "second chunk on page 139", brand="netgear", pages=(139, 139))
    b = result(4, "page 140 text", brand="netgear", pages=(140, 140))
    sources = [ContextSource(n=i + 1, result=r) for i, r in enumerate([a1, other_manual, a2, b])]
    merged = chat_service.cited_sources([4, 1, 3, 2], sources)
    assert [(s.document_id, s.refs) for s in merged] == [
        ("netgear_doc", [4]), ("netgear_doc", [1, 3]), ("tenda_doc", [2])]  # same pages, other manual: separate
    assert merged[1].n == 1 and merged[1].excerpt == "first chunk on page 139"
