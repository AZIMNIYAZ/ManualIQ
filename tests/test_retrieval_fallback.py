"""Step 12: retrieval fallback (one more search and answer after a refusal)."""

import json
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import Settings  # noqa: E402
from app.models.chat import ChatRequest  # noqa: E402
from app.models.search import SearchFilters, SearchResponse  # noqa: E402
from app.services import chat_service  # noqa: E402
from app.services.brand_detection import build_alias_table  # noqa: E402
from app.services.generation_service import GenerationResult  # noqa: E402
from app.services.prompts import NOT_FOUND_ANSWER  # noqa: E402
from app.services.retrieval_fallback import (alternative_queries, exempt_names, merge, technical_terms,  # noqa: E402
                                             validate)
from tests.test_chat import result  # noqa: E402

ALIASES = build_alias_table({"dlink": ["D-Link"], "tenda": ["Tenda"], "netgear": ["Netgear"]})
NAMES = {"dlink": "D-Link DIR-X1560", "tenda": "Tenda AX3000 Wi-Fi 6 series", "netgear": "Netgear Nighthawk RAX50"}
EXEMPT = exempt_names(list(NAMES.values()))
GLOSSARY = {"names": NAMES, "manuals": {"dlink": ["QoS Engine", "Website Filter"], "tenda": ["Add a client to the blacklist"],
                                        "netgear": ["Port Forwarding"]}}


class ScriptedLLM:
    """Returns the scripted answers in order (the last one repeats)."""
    model = "fake-grok"

    def __init__(self, *answers, error=None):
        self.answers, self.error, self.calls = list(answers), error, []

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        return GenerationResult(answer=answer, model=self.model, input_tokens=1000, output_tokens=100,
                                reasoning_tokens=0, latency_ms=5.0, finish_reason="stop")


def queries(*qs) -> str:
    return json.dumps({"queries": list(qs)})


# ---------------------------------------------------------------- safeguards

def test_brand_and_model_names_need_not_survive_but_features_must():
    q = "How do I enable Wi-Fi 7 MLO on my D-Link DIR-X1560?"
    assert technical_terms(q, EXEMPT) == ["7", "MLO"]
    assert validate(q, "enable Wi-Fi 7 Multi-Link Operation MLO", ALIASES, EXEMPT) is None
    assert validate(q, "QoS Engine wireless settings", ALIASES, EXEMPT) == "dropped technical terms: ['7', 'MLO']"


def test_numbers_in_model_names_are_not_exempt():
    """'6' of 'Wi-Fi 6 series' must not exempt the '6' of '6 GHz'."""
    assert technical_terms("Enable the 6 GHz band on Tenda", EXEMPT) == ["6", "GHz"]


def test_alternative_may_not_add_a_brand():
    assert "names a brand" in validate("kick a device off for good", "Tenda blacklist add client", ALIASES, EXEMPT)
    assert validate("kick it off my Tenda for good", "Add a client to the blacklist", ALIASES, EXEMPT) is None


def test_alternatives_are_checked_deduplicated_and_capped():
    llm = ScriptedLLM(queries("Add a client to the blacklist", "add a client to the BLACKLIST",
                              "Netgear access control block device", "block device permanently", "remove client"))
    out = alternative_queries("How do I kick a device off for good?", "Tenda: ...", llm, ALIASES, EXEMPT, n=3)
    assert out.queries == ["Add a client to the blacklist", "block device permanently", "remove client"]
    assert out.rejected == [("Netgear access control block device", "names a brand the question does not: ['netgear']")]
    assert out.input_tokens == 1000 and out.error is None
    assert "Write 3 different search queries" in llm.calls[0][0] and "Manual contents:\nTenda: ..." in llm.calls[0][1]


@pytest.mark.parametrize("llm, error", [
    (ScriptedLLM("not json"), "invalid output: JSONDecodeError"),
    (ScriptedLLM(error=RuntimeError("timeout")), "call failed: timeout"),
])
def test_bad_output_or_failed_call_gives_no_alternatives(llm, error):
    out = alternative_queries("q", "g", llm, ALIASES, EXEMPT, 3)
    assert out.queries == [] and out.error == error


def test_off_topic_question_gives_no_alternatives():
    out = alternative_queries("What is the weather in London?", "g", ScriptedLLM(queries()), ALIASES, EXEMPT, 3)
    assert out.queries == [] and out.error is None


def test_merge_rrf_and_max():
    a = [result(1, "a"), result(2, "b")]
    b = [result(1, "c", brand="tenda"), result(2, "b")]
    assert [r.chunk_id for r in merge([a, b], "rrf", 5)] == ["netgear_0002", "netgear_0001", "tenda_0001"]
    assert [r.rank for r in merge([a, b], "max", 2)] == [1, 2]


# ---------------------------------------------------------------- chat integration (search faked)

DLINK_QOS = result(1, "QoS Engine: drag the device to the Highest Priority slot.", brand="dlink", model="DIR-X1560",
                   pages=(69, 70), section="QoS Engine")
UNRELATED = result(9, "Set up the guest zone.", brand="dlink", model="DIR-X1560", pages=(61, 61), section="Guest Zone")


@pytest.fixture
def search(monkeypatch):
    """Original question -> UNRELATED; any alternative -> DLINK_QOS. Records every call."""
    calls = []

    def fake(client, embedder, settings, request, aliases, detect_brand=True, brand_source="request", log_event=True):
        calls.append({"query": request.query, "brand": request.brand, "detect_brand": detect_brand})
        hits = [UNRELATED] if len(calls) == 1 else [DLINK_QOS]
        return SearchResponse(query=request.query, collection="rag_500_60_ctx", query_tokens=10, query_truncated=False,
                              filters=SearchFilters(brands=["dlink"], brand_source="detected", include_regulatory=False),
                              results=hits, timings_ms={"total": 1.0})
    monkeypatch.setattr(chat_service.retrieval_service, "search", fake)
    monkeypatch.setattr(chat_service, "load_glossary", lambda d: GLOSSARY)
    return calls


def run(generator, rewriter, fallback=True, message="How do I give my laptop the highest priority on my D-Link?"):
    settings = Settings(_env_file=None, retrieval_fallback=fallback)
    return chat_service.answer(None, None, generator, settings, ChatRequest(message=message, debug=True), ALIASES,
                               rewriter=rewriter)


def test_answered_questions_never_enter_the_fallback(search):
    """The 94% path: an answer on the first attempt is returned unchanged, with no extra calls."""
    generator, rewriter = ScriptedLLM("Use the Guest Zone [1]."), ScriptedLLM(queries("QoS Engine"))
    r = run(generator, rewriter)
    assert r.found_in_documents and r.fallback is None and "fallback" not in r.timings_ms
    assert len(generator.calls) == 1 and rewriter.calls == [] and len(search) == 1


def test_off_by_default(search):
    generator, rewriter = ScriptedLLM(NOT_FOUND_ANSWER), ScriptedLLM(queries("QoS Engine"))
    r = run(generator, rewriter, fallback=False)
    assert not r.found_in_documents and r.fallback is None and rewriter.calls == [] and len(search) == 1


def test_refusal_triggers_one_search_and_one_more_answer(search):
    generator = ScriptedLLM(NOT_FOUND_ANSWER, "Drag the laptop to Highest Priority in the QoS Engine [1].")
    rewriter = ScriptedLLM(queries("QoS Engine highest priority device", "QoS Engine priority"))
    r = run(generator, rewriter)
    assert r.found_in_documents and r.fallback.answered and r.fallback.queries == [
        "QoS Engine highest priority device", "QoS Engine priority"]
    # every alternative is searched with the ORIGINAL search's brand filter, no new detection
    assert [c["query"] for c in search[1:]] == r.fallback.queries
    assert all(c["brand"] == ["dlink"] and c["detect_brand"] is False for c in search[1:])
    # the second answer is for the ORIGINAL question, with the merged results
    assert len(generator.calls) == 2
    assert generator.calls[1][1].endswith("Question: How do I give my laptop the highest priority on my D-Link?")
    assert [s.chunk_id for s in r.retrieved] == ["dlink_0001", "dlink_0009"] and r.sources[0].section == "QoS Engine"
    assert r.llm.input_tokens == 2000 and r.fallback.input_tokens == 1000 and "fallback" in r.timings_ms
    assert "D-Link DIR-X1560: QoS Engine; Website Filter" in rewriter.calls[0][1]  # the detected brand's glossary only
    assert "Tenda" not in rewriter.calls[0][1]


def test_second_refusal_is_final(search):
    generator, rewriter = ScriptedLLM(NOT_FOUND_ANSWER), ScriptedLLM(queries("QoS Engine"))
    r = run(generator, rewriter)
    assert not r.found_in_documents and r.answer == NOT_FOUND_ANSWER and r.fallback.answered is False
    assert len(generator.calls) == 2 and len(rewriter.calls) == 1  # no loop


def test_no_surviving_alternative_keeps_the_refusal(search):
    generator, rewriter = ScriptedLLM(NOT_FOUND_ANSWER), ScriptedLLM(queries("Tenda blacklist"))
    r = run(generator, rewriter)
    assert not r.found_in_documents and r.fallback.queries == [] and len(r.fallback.rejected) == 1
    assert len(generator.calls) == 1 and len(search) == 1


def test_missing_glossary_is_a_warning_not_an_error(search, monkeypatch):
    def missing(d):
        raise FileNotFoundError("documents/glossary.json is missing")
    monkeypatch.setattr(chat_service, "load_glossary", missing)
    r = run(ScriptedLLM(NOT_FOUND_ANSWER), ScriptedLLM(queries("QoS Engine")))
    assert not r.found_in_documents and r.fallback is None
    assert {"type": "fallback_unavailable", "detail": "documents/glossary.json is missing"} in r.warnings
