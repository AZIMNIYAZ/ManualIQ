import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import Settings  # noqa: E402
from app.models.chat import ChatRequest  # noqa: E402
from app.models.search import SearchFilters, SearchResponse, SearchResult  # noqa: E402
from app.services import chat_service  # noqa: E402
from app.services import generation_service as gen  # noqa: E402
from app.services.answer_checks import check_answer, is_not_found  # noqa: E402
from app.services.context_builder import build_context  # noqa: E402
from app.services.generation_service import GenerationResult  # noqa: E402
from app.services.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT  # noqa: E402


def result(rank, text, brand="netgear", model="Nighthawk RAX50", pages=(139, 139), section="Port forwarding"):
    return SearchResult(rank=rank, score=round(0.9 - rank / 100, 3), chunk_id=f"{brand}_{rank:04d}", brand=brand,
                        model=model, source=f"{brand}.pdf", page_start=pages[0], page_end=pages[1],
                        citation=f"{brand}.pdf, p. {pages[0]}", section=section, content_type="normal", text=text)


NETGEAR = [
    result(1, "To set up port forwarding:\n1. Enter http://www.routerlogin.net.\n2. Select ADVANCED > Advanced Setup > "
              "Port Forwarding/Port Triggering.\n3. Click the Add button. The router restarts within 8 seconds."),
    result(2, "The default LAN IP address is 192.168.1.1. A maximum of 64 clients can be blocked.", pages=(140, 141)),
]


class FakeGenerator:
    model = "fake-grok"

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return GenerationResult(answer=self.answer, model=self.model, input_tokens=1000, output_tokens=100,
                                reasoning_tokens=0, latency_ms=5.0, finish_reason="stop")


# ---------------------------------------------------------------- context builder

def test_context_lists_numbered_sources_with_brand_and_citation():
    ctx = build_context(NETGEAR, max_tokens=2500)
    assert ctx.text.startswith("[1] Brand: netgear | Model: Nighthawk RAX50 | netgear.pdf, p. 139 | Section: Port forwarding\n")
    assert "\n\n[2] Brand: netgear" in ctx.text and len(ctx.sources) == 2 and ctx.dropped == 0


def test_context_budget_drops_whole_lower_ranked_chunks():
    long = [result(i, "word " * 200) for i in range(1, 6)]
    ctx = build_context(long, max_tokens=500)
    assert len(ctx.sources) == 2 and ctx.dropped == 3  # ~215 tokens each: two fit
    assert ctx.text.count("word") == 400  # chunks are never cut in the middle


# ---------------------------------------------------------------- answer checks

def test_cited_answer_with_supported_values_has_no_warnings():
    answer = ("1. Enter http://www.routerlogin.net [1].\n2. Select ADVANCED > Advanced Setup > Port Forwarding/Port "
              "Triggering [1].\n3. Click Add; the router restarts within 8 seconds [1]. The LAN IP is 192.168.1.1 [2].")
    check = check_answer(answer, build_context(NETGEAR, 2500))
    assert check.cited == [1, 2] and check.warnings == [] and not check.not_found


def test_default_ip_from_general_knowledge_is_flagged():
    """The Tenda case: a correct-looking value that is not in any retrieved source."""
    check = check_answer("The default login IP is 192.168.0.1 [1].", build_context(NETGEAR, 2500))
    assert {"type": "unsupported_value", "detail": "ip_address '192.168.0.1' is not in any source"} in check.warnings


def test_value_from_uncited_source_and_invented_menu_path():
    check = check_answer("Block up to 64 clients [1]. Go to SECURITY > Firewall [1].", build_context(NETGEAR, 2500))
    types = [w["type"] for w in check.warnings]
    assert "value_in_uncited_source" in types  # 64 clients is in [2], cited [1]
    assert any("SECURITY > Firewall" in w["detail"] for w in check.warnings if w["type"] == "unsupported_value")


def test_invalid_and_missing_citations():
    ctx = build_context(NETGEAR, 2500)
    assert check_answer("See the manual [7].", ctx).warnings[0]["type"] == "invalid_citation"
    assert check_answer("Click Add.", ctx).warnings == [{"type": "no_citations", "detail": "The answer cites no source"}]


@pytest.mark.parametrize("text, expected", [
    (NOT_FOUND_ANSWER, True), ("I couldn’t find this in the provided manuals", True),
    ("I couldn't find this in the provided manuals, but routers often use 192.168.0.1 and you could try the "
     "default admin password printed on the label of the device.", False),  # long answer: not a refusal
    ("Click Add [1].", False),
])
def test_not_found_detection(text, expected):
    assert is_not_found(text) is expected


# ---------------------------------------------------------------- chat service (search faked)

@pytest.fixture
def fake_search(monkeypatch):
    def install(results):
        response = SearchResponse(query="q", collection="rag_350_60", query_tokens=20, query_truncated=False,
                                  filters=SearchFilters(brands=["netgear"], brand_source="detected", include_regulatory=False),
                                  results=results, timings_ms={"total": 1.0})
        monkeypatch.setattr(chat_service.retrieval_service, "search", lambda *a, **k: response)
    return install


def run(generator, message="How do I set up port forwarding on Netgear?", debug=False):
    return chat_service.answer(None, None, generator, Settings(_env_file=None), ChatRequest(message=message, debug=debug), ())


def test_answer_returns_only_cited_sources(fake_search):
    fake_search(NETGEAR)
    generator = FakeGenerator("1. Select ADVANCED > Advanced Setup > Port Forwarding/Port Triggering [1].")
    r = run(generator, debug=True)
    assert r.found_in_documents and [s.n for s in r.sources] == [1] and r.sources[0].citation == "netgear.pdf, p. 139"
    assert len(r.retrieved) == 2 and r.context.startswith("[1] Brand: netgear")
    system, user = generator.calls[0]
    assert system == SYSTEM_PROMPT and user.endswith("Question: How do I set up port forwarding on Netgear?")
    assert r.llm.input_tokens == 1000 and r.llm.prompt_version == "1.1"


def test_not_found_answer_has_no_sources(fake_search):
    fake_search(NETGEAR)
    r = run(FakeGenerator(NOT_FOUND_ANSWER))
    assert not r.found_in_documents and r.sources == [] and r.warnings == []


def test_llm_not_called_when_search_finds_nothing(fake_search):
    fake_search([])
    generator = FakeGenerator("should not be used")
    r = run(generator)
    assert generator.calls == [] and r.answer == NOT_FOUND_ANSWER and r.llm.model is None


# ---------------------------------------------------------------- API errors

from app.api.search import get_aliases  # noqa: E402
from app.database.qdrant import get_qdrant  # noqa: E402
from app.services.embedding_service import get_embedder  # noqa: E402

from app.services.session_service import get_session_store  # noqa: E402

OTHER_DEPENDENCIES = {get_qdrant: lambda: None, get_embedder: lambda: None, get_aliases: lambda: (),
                      get_session_store: lambda: None, gen.get_rewriter: lambda: None}

@pytest.mark.parametrize("error, status", [
    (gen.LLMAuthenticationError("auth"), 502), (gen.LLMRateLimited("slow down"), 429),
    (gen.LLMTimeout("timeout"), 504), (gen.LLMUnavailable("down"), 502),
])
def test_api_maps_llm_errors(fake_search, error, status):
    from app.main import app
    fake_search(NETGEAR)
    app.dependency_overrides.update({**OTHER_DEPENDENCIES, gen.get_generator: lambda: FakeGenerator(error=error)})
    try:
        r = TestClient(app).post("/api/v1/chat", json={"message": "port forwarding"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == status and r.json() == {"error": str(error)}


def test_api_without_llm_configured_is_503():
    from app.main import app
    app.state.generator = None  # as when XAI_API_KEY is missing
    app.dependency_overrides.update(OTHER_DEPENDENCIES)
    try:
        r = TestClient(app).post("/api/v1/chat", json={"message": "port forwarding"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and "XAI_API_KEY" in r.json()["error"]


def test_generator_not_created_without_key():
    assert gen.create_generator(Settings(_env_file=None)) is None
    with pytest.raises(gen.GenerationNotConfigured):
        gen.GrokGenerationService(Settings(_env_file=None))
