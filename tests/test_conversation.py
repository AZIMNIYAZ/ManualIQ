import os
import socket
from datetime import datetime, timezone

import pytest
import redis as redis_lib
from fastapi.testclient import TestClient

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import Settings  # noqa: E402
from app.models.chat import ChatRequest  # noqa: E402
from app.models.search import SearchFilters, SearchResponse  # noqa: E402
from app.models.session import StoredMessage  # noqa: E402
from app.services import chat_service  # noqa: E402
from app.services.brand_detection import build_alias_table  # noqa: E402
from app.services.generation_service import GenerationResult  # noqa: E402
from app.services.query_rewriter import format_history, rewrite  # noqa: E402
from app.services.session_service import SessionNotFound, SessionStore, SessionUnavailable  # noqa: E402
from tests.test_chat import NETGEAR, FakeGenerator, result  # noqa: E402

ALIASES = build_alias_table({"netgear": ["Netgear", "Nighthawk"], "asus": ["ASUS"], "tenda": ["Tenda"]})
BRANDS = ["asus", "netgear", "tenda"]


def msg(role, content, **extra):
    return StoredMessage(role=role, content=content, timestamp=datetime.now(timezone.utc), **extra)


HISTORY = [msg("user", "How do I set up port forwarding on my Netgear router?"),
           msg("assistant", "1. Enter http://www.routerlogin.net [1]. 2. Select ADVANCED > Advanced Setup [1]. " * 10,
               rewritten_query=None, brands=["netgear"])]


class FakeRewriter(FakeGenerator):
    model = "fake-rewriter"


# ---------------------------------------------------------------- rewriter

def test_history_shortens_answers_and_drops_citations():
    text = format_history(HISTORY, answer_chars=80)
    assert text.startswith("User: How do I set up port forwarding on my Netgear router?\nAssistant: ")
    assert "[1]" not in text and text.endswith(" ...") and len(text.splitlines()[1]) < 110


@pytest.mark.parametrize("output", [
    '{"query": "How do I set up port forwarding on an ASUS router?", "brands": ["asus"]}',
    '```json\n{"query": "How do I set up port forwarding on an ASUS router?", "brands": ["asus"]}\n```',
])
def test_valid_rewrite(output):
    r = rewrite("What about on ASUS?", HISTORY, FakeRewriter(output), BRANDS, ALIASES)
    assert r.rewritten and r.query == "How do I set up port forwarding on an ASUS router?" and r.brands == ["asus"]
    assert r.warnings == []


def test_brand_switch_guard_corrects_the_model():
    """The reasoning model returned both brands for 'What about on ASUS?': the message names ASUS, so ASUS wins."""
    out = '{"query": "port forwarding on ASUS router", "brands": ["netgear", "asus"]}'
    r = rewrite("What about on ASUS?", HISTORY, FakeRewriter(out), BRANDS, ALIASES)
    assert r.brands == ["asus"] and r.warnings[0]["type"] == "rewrite_brand_corrected"


def test_carry_over_and_unknown_brand_dropped():
    out = '{"query": "How do I edit a port forwarding rule on a Netgear router?", "brands": ["netgear", "linksys"]}'
    r = rewrite("Can I edit the rule later?", HISTORY, FakeRewriter(out), BRANDS, ALIASES)
    assert r.brands == ["netgear"] and r.warnings == [{"type": "rewrite_unknown_brand", "detail": "dropped ['linksys']"}]


@pytest.mark.parametrize("output, reason", [
    ("Sure! Here is the query: port forwarding", "invalid rewrite output"),
    ('{"query": "", "brands": []}', "invalid rewrite output"),
    ('{"brands": ["asus"]}', "invalid rewrite output"),
    ('{"query": "1. Open the page\\n2. Click Add", "brands": []}', "looks like an answer"),
])
def test_bad_rewrite_falls_back_to_the_message(output, reason):
    r = rewrite("What about on ASUS?", HISTORY, FakeRewriter(output), BRANDS, ALIASES)
    assert not r.rewritten and r.query == "What about on ASUS?" and r.brands is None
    assert r.warnings[0]["type"] == "rewrite_failed" and reason in r.warnings[0]["detail"]


def test_rewrite_call_error_falls_back():
    r = rewrite("And on ASUS?", HISTORY, FakeRewriter(error=ConnectionError("down")), BRANDS, ALIASES)
    assert not r.rewritten and "rewrite call failed" in r.warnings[0]["detail"]


# ---------------------------------------------------------------- Redis session store (real Redis)

def redis_running() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", 6379)) == 0


needs_redis = pytest.mark.skipif(not redis_running(), reason="Redis not running (docker compose up -d)")


@pytest.fixture
def store():
    r = redis_lib.Redis(host="127.0.0.1", port=6379, decode_responses=True)
    yield SessionStore(r, prefix="test_session", ttl_seconds=100, max_messages=4)
    for key in r.scan_iter("test_session:*"):
        r.delete(key)


@needs_redis
def test_turns_are_stored_in_order_capped_and_ttl_refreshed(store):
    for i in range(3):
        store.append_turn("abcdefgh", msg("user", f"q{i}"), msg("assistant", f"a{i}", sources=[f"x.pdf, p. {i}"]))
    messages, ttl = store.get("abcdefgh")
    assert [m.content for m in messages] == ["q1", "a1", "q2", "a2"]  # capped at 4, oldest dropped
    assert messages[-1].sources == ["x.pdf, p. 2"] and 90 < ttl <= 100
    store.redis.expire(store.key("abcdefgh"), 5)
    store.append_turn("abcdefgh", msg("user", "q3"), msg("assistant", "a3"))
    assert store.redis.ttl(store.key("abcdefgh")) > 90  # every turn restarts the expiry
    assert [m.content for m in store.history("abcdefgh", last=2)] == ["q3", "a3"]


@needs_redis
def test_unknown_session(store):
    assert store.history("neverseen", last=8) == []
    with pytest.raises(SessionNotFound):
        store.get("neverseen")
    store.delete("neverseen")  # idempotent


def test_redis_down_is_session_unavailable():
    dead = SessionStore(redis_lib.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2),
                        "test_session", 100, 4)
    with pytest.raises(SessionUnavailable):
        dead.history("abcdefgh", last=8)


# ---------------------------------------------------------------- chat flow (search faked)

@pytest.fixture
def fake_search(monkeypatch):
    calls = []

    def search(client, embedder, settings, request, aliases, detect_brand=True, brand_source="request"):
        calls.append((request, brand_source))
        brands = request.brand or []
        return SearchResponse(query=request.query, collection="rag_350_60", query_tokens=20, query_truncated=False,
                              filters=SearchFilters(brands=brands, brand_source=brand_source if brands else "none",
                                                    include_regulatory=False),
                              results=NETGEAR + [result(3, "ASUS port forwarding steps", brand="asus")],
                              timings_ms={"total": 1.0})

    monkeypatch.setattr(chat_service.retrieval_service, "search", search)
    monkeypatch.setattr(chat_service.retrieval_service, "collection_brands", lambda c, n: {b: 1 for b in BRANDS})
    return calls


SETTINGS = Settings(_env_file=None)


def chat(generator, rewriter, sessions, message, session_id=None, brand=None):
    return chat_service.answer(None, None, generator, SETTINGS,
                               ChatRequest(message=message, session_id=session_id, brand=brand), ALIASES,
                               sessions=sessions, rewriter=rewriter)


@needs_redis
def test_first_message_no_rewrite_then_follow_up_rewritten(fake_search, store):
    generator = FakeGenerator("Select ADVANCED > Advanced Setup > Port Forwarding/Port Triggering [1].")
    rewriter = FakeRewriter('{"query": "How do I set up port forwarding on an ASUS router?", "brands": ["asus"]}')

    first = chat(generator, rewriter, store, "How do I set up port forwarding on my Netgear router?")
    assert len(first.session_id) == 32 and first.history_used == 0
    assert rewriter.calls == [] and first.rewrite is None and first.rewritten_query is None

    second = chat(generator, rewriter, store, "What about on ASUS?", session_id=first.session_id)
    assert len(rewriter.calls) == 1 and second.history_used == 2
    assert second.rewritten_query == "How do I set up port forwarding on an ASUS router?"
    assert second.filters.brands == ["asus"] and second.filters.brand_source == "rewrite"
    assert fake_search[-1][0].query == "How do I set up port forwarding on an ASUS router?"

    # previous answers never reach the answer model
    _, answer_input = generator.calls[-1]
    assert "Advanced Setup > Port Forwarding" not in answer_input.split("Question:")[1]
    assert "original wording" in answer_input and "What about on ASUS?" in answer_input

    messages, _ = store.get(first.session_id)
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[3].rewritten_query == second.rewritten_query and messages[3].brands == ["asus"]
    assert messages[3].sources == ["netgear.pdf, p. 139"]


@needs_redis
def test_request_brand_wins_with_conflict_warning(fake_search, store):
    generator = FakeGenerator("Click Add [1].")
    rewriter = FakeRewriter('{"query": "port forwarding on ASUS", "brands": ["asus"]}')
    first = chat(generator, rewriter, store, "Port forwarding on Netgear?")
    r = chat(generator, rewriter, store, "What about on ASUS?", session_id=first.session_id, brand="netgear")
    assert r.filters.brands == ["netgear"] and r.filters.brand_source == "request"
    assert r.warnings[0]["type"] == "brand_conflict"


def test_redis_down_still_answers_with_warning(fake_search):
    class DownStore:
        def history(self, *a):
            raise SessionUnavailable("Redis down")

        def append_turn(self, *a):
            raise AssertionError("must not be called when memory is unavailable")

    r = chat(FakeGenerator("Click Add [1]."), FakeRewriter("{}"), DownStore(), "What about on ASUS?", "abcdefgh")
    assert r.found_in_documents and r.warnings[0]["type"] == "history_unavailable" and r.history_used == 0


def test_save_failure_is_reported(fake_search):
    class SaveFails:
        def history(self, *a):
            return []

        def append_turn(self, *a):
            raise SessionUnavailable("Redis went away")

    r = chat(FakeGenerator("Click Add [1]."), None, SaveFails(), "Port forwarding on Netgear?")
    assert r.found_in_documents and r.warnings[-1]["type"] == "history_not_saved"


# ---------------------------------------------------------------- API

@needs_redis
def test_session_endpoints(store):
    from app.main import app
    from app.services.session_service import get_session_store

    store.append_turn("abcdefgh", msg("user", "q"), msg("assistant", "a [1]", sources=["x.pdf, p. 1"]))
    app.dependency_overrides[get_session_store] = lambda: store
    try:
        api = TestClient(app)
        got = api.get("/api/v1/sessions/abcdefgh")
        assert got.status_code == 200 and [m["content"] for m in got.json()["messages"]] == ["q", "a [1]"]
        assert 0 < got.json()["expires_in_seconds"] <= 100
        assert api.get("/api/v1/sessions/neverseen").status_code == 404
        assert api.delete("/api/v1/sessions/abcdefgh").status_code == 204
        assert api.get("/api/v1/sessions/abcdefgh").status_code == 404
        assert api.delete("/api/v1/sessions/abcdefgh").status_code == 204  # idempotent
        assert api.get("/api/v1/sessions/bad!").status_code == 422
    finally:
        app.dependency_overrides.clear()
