from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.chunk import Chunk
from app.models.document import Document, Page
from app.services.brand_detection import build_alias_table
from evaluation import dataset
from evaluation.dataset import Absence, Evidence, Question, Turn, contains, normalise
from evaluation.judge import JudgeError, JudgeVerdict, judge, parse_verdict
from evaluation.labels import check_labels
from evaluation.metrics import (Retrieved, agreement, asks_which_router, first_hit_rank, matches, refusal_counts,
                                retrieval_summary)
from evaluation.scoring import blame, estimate, rewrite_ok, score_answer

ALIASES = build_alias_table({"netgear": ["Netgear"], "tenda": ["Tenda"], "dlink": ["D-Link"]})


def q(**kw) -> Question:
    base = {"id": "x-01", "category": "direct", "question": "How?", "brand": "netgear", "answerable": True,
            "evidence": [{"document_id": "netgear_rax50", "pages": [50, 51], "key_phrase": "Delete Keyword"}]}
    return Question(**{**base, **kw})


def chunk(doc="netgear_rax50", pages=(50, 50), text="Click the Delete Keyword button.") -> Retrieved:
    return Retrieved(chunk_id="c", document_id=doc, brand="netgear", page_start=pages[0], page_end=pages[1],
                     content_type="normal", score=0.8, text=text)


# ---------------------------------------------------------------- text matching

@pytest.mark.parametrize("text, fact", [
    ("Go to ADVANCED>Security >Block Sites", "Advanced > Security > Block Sites"),
    ("the router’s **Wi-Fi** password", "router's wifi password"),
    ("Select  L2TP/IPSec\nwith pre-shared key", "L2TP/IPSec with pre-shared key"),
])
def test_normalised_matching(text, fact):
    assert contains(text, fact)


def test_alternatives_count_if_any_matches():
    assert contains("Click Clear List.", ["Delete Keyword", "Clear List"])
    assert not contains("Click Apply.", ["Delete Keyword", "Clear List"])


# ---------------------------------------------------------------- dataset schema

def test_real_dataset_loads_and_is_consistent():
    questions, conversations = dataset.load_questions(), dataset.load_conversations()
    assert len(questions) == 40 and len(conversations) == 6
    ids = [x.id for x in questions] + [t.id for c in conversations for t in c.turns]
    assert len(ids) == len(set(ids))
    assert all(x.absence for x in questions if x.category == "unanswerable")
    assert all(x.policy == "per_brand" for x in questions if x.category == "no_brand")


def test_schema_rejects_inconsistent_labels():
    with pytest.raises(ValueError, match="need evidence"):
        q(evidence=[])
    with pytest.raises(ValueError, match="cannot have evidence"):
        q(answerable=False, policy="refuse")
    with pytest.raises(ValueError, match="absence"):
        q(category="unanswerable", answerable=False, policy="refuse", evidence=[])
    with pytest.raises(ValueError, match="policy"):
        q(category="no_brand", brand=None)  # no-brand questions must be labelled per_brand
    with pytest.raises(ValueError, match="reversed"):
        Evidence(document_id="d", pages=(5, 4), key_phrase="abc")


def test_dataset_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text("{}", encoding="utf-8")
    h1 = dataset.dataset_hash(a)
    a.write_text("{} ", encoding="utf-8")
    assert dataset.dataset_hash(a) != h1


# ---------------------------------------------------------------- label checker (synthetic manual)

def manual() -> Document:
    pages = [
        Page(page_number=1, raw_text="Contents", status="excluded", reason="toc"),
        Page(page_number=2, raw_text="Block Sites. Click the Delete Keyword button.",
             clean_text="Block Sites. Click the Delete Keyword button.", status="indexed"),
        Page(page_number=3, raw_text="FCC Part 15", clean_text="FCC Part 15", status="indexed",
             content_type="regulatory"),
    ]
    return Document(document_id="netgear_rax50", brand="netgear", model="RAX50", source="n.pdf", pages=pages)


def chunk_of(text: str, page: int = 2) -> Chunk:
    return Chunk(chunk_id="netgear_rax50_0000", document_id="netgear_rax50", brand="netgear", model="RAX50",
                 source="n.pdf", chunk_index=0, text=text, token_count=10, page_start=page, page_end=page,
                 content_type="normal", section=None, cleaner_version="1.3", chunking="350/60")


def ev(pages, phrase) -> dict:
    return {"document_id": "netgear_rax50", "pages": pages, "key_phrase": phrase}


def test_correct_label_passes():
    report = check_labels([q(evidence=[ev([2, 2], "delete keyword")])], [], [manual()])
    assert report.ok and report.warnings == [] and report.checked_evidence == 1


@pytest.mark.parametrize("evidence, problem", [
    (ev([2, 2], "Erase button"), "not on these pages"),
    (ev([1, 1], "Contents"), "no page in the range is indexed"),
    (ev([2, 9], "Delete Keyword"), "only 3 pages"),
    ({"document_id": "nope", "pages": [2, 2], "key_phrase": "Delete Keyword"}, "unknown document_id"),
])
def test_broken_labels_are_errors(evidence, problem):
    report = check_labels([q(evidence=[evidence])], [], [manual()])
    assert not report.ok and problem in report.errors[0]


def test_regulatory_evidence_and_split_phrase_are_warnings():
    report = check_labels([q(evidence=[ev([2, 3], "Delete Keyword")])], [], [manual()],
                          chunks=[chunk_of("Block Sites. Click the Delete"), chunk_of("Keyword button.")])
    assert report.ok
    assert any("regulatory" in w for w in report.warnings)
    assert any("split across chunks" in w for w in report.warnings)


def test_absence_terms_found_in_a_manual_are_errors():
    found = q(id="u-01", category="unanswerable", answerable=False, policy="refuse", evidence=[],
              absence=Absence(terms=["delete keyword"], manuals="all"))
    absent = q(id="u-02", category="unanswerable", answerable=False, policy="refuse", evidence=[],
               absence=Absence(terms=["WPA4"], manuals=["netgear_rax50"]))
    report = check_labels([found, absent], [], [manual()])
    assert len(report.errors) == 1 and "u-01" in report.errors[0]
    assert any("u-02" in line and "0 hits" in line for line in report.absence)


def test_duplicate_ids_are_errors():
    report = check_labels([q(evidence=[ev([2, 2], "Delete Keyword")])] * 2, [], [manual()])
    assert any("duplicate" in e for e in report.errors)


@pytest.mark.skipif(not list(Path("documents").glob("*.pdf")), reason="manual PDFs not present")
def test_real_labels_match_the_manuals():
    """Every key phrase is on its stated pages: a broken label fails the test suite."""
    from app.services.chunker import chunk_documents
    from app.services.document_loader import load_all_documents
    from app.services.text_cleaner import clean_documents

    s = Settings(_env_file=None)
    docs, _ = clean_documents(load_all_documents(s.documents_dir))
    chunks, _ = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
    report = check_labels(dataset.load_questions(), dataset.load_conversations(), docs, chunks)
    assert report.errors == []


# ---------------------------------------------------------------- retrieval metrics

def test_match_needs_same_manual_overlapping_pages_and_phrase():
    e = Evidence(document_id="netgear_rax50", pages=(50, 51), key_phrase="delete keyword")
    assert matches(chunk(pages=(49, 50)), e)
    assert not matches(chunk(pages=(52, 53)), e)  # pages do not overlap
    assert not matches(chunk(doc="tenda_ax3000"), e)  # other manual, same pages
    assert not matches(chunk(text="Click Apply."), e)  # right pages, phrase not in this chunk


def test_rank_hit_and_mrr():
    evidence = [Evidence(document_id="netgear_rax50", pages=(50, 51), key_phrase="Delete Keyword")]
    results = [chunk(text="other"), chunk(text="other"), chunk()]
    assert first_hit_rank(results, evidence) == 3
    s = retrieval_summary([{"rank": 1}, {"rank": 3}, {"rank": None}, {"rank": 7}])
    assert (s["hit@1"], s["hit@3"], s["hit@5"], s["hit@10"]) == (0.25, 0.5, 0.5, 0.75)
    assert s["mrr"] == round((1 + 1 / 3 + 0 + 1 / 7) / 4, 3)


def test_refusal_counts():
    rows = [{"answerable": True, "refused": False}, {"answerable": True, "refused": True},
            {"answerable": False, "refused": True}, {"answerable": False, "refused": False}]
    c = refusal_counts(rows)
    assert (c["answered"], c["false_refusals"], c["correct_refusals"], c["false_answers"]) == (1, 1, 1, 1)
    assert c["refusal_precision"] == 0.5 and c["refusal_recall"] == 0.5


def test_agreement_and_kappa():
    assert agreement([(True, True)] * 8 + [(False, False)] * 2) == {"n": 10, "agreement": 1.0, "kappa": 1.0,
                                                                     "disagreements": 0}
    # always "grounded": 80% raw agreement, but no better than chance
    a = agreement([(True, True)] * 8 + [(True, False)] * 2)
    assert a["agreement"] == 0.8 and a["kappa"] == 0.0


# ---------------------------------------------------------------- answer scoring

def verdict(**kw) -> JudgeVerdict:
    return JudgeVerdict(**{"grounded": True, "citations_correct": True, "brand_mixing": False, **kw})


def test_single_brand_answer_scoring():
    item = q(key_facts=["Block Sites", ["Delete Keyword", "Clear List"]], must_not_contain=["Add Keyword"])
    good = "Go to Security > Block Sites [1] and click Clear List [1]."
    assert score_answer(item, "single_brand", good, False, verdict(), ALIASES)["pass"]
    s = score_answer(item, "single_brand", "Go to Block Sites [1].", False, verdict(), ALIASES)
    assert not s["pass"] and s["key_facts_missing"] == ["Delete Keyword | Clear List"] and s["key_fact_coverage"] == 0.5
    s = score_answer(item, "single_brand", good + " Then click Add Keyword.", False, verdict(), ALIASES)
    assert not s["pass"] and s["must_not_found"] == ["Add Keyword"]
    assert not score_answer(item, "single_brand", good, False, verdict(grounded=False), ALIASES)["pass"]
    assert score_answer(item, "single_brand", "x", True, None, ALIASES)["outcome"] == "false_refusal"


def test_refusal_policy():
    item = q(category="unanswerable", answerable=False, policy="refuse", evidence=[],
             absence=Absence(terms=["WPA4"]))
    assert score_answer(item, "refuse", "I couldn't find this.", True, None, ALIASES)["outcome"] == "correct_refusal"
    s = score_answer(item, "refuse", "Enable WPA4 under Security [1].", False, verdict(), ALIASES)
    assert s["outcome"] == "false_answer" and not s["pass"]
    assert s["review_rule_b"]  # a reviewer may still accept it under rule B
    assert not score_answer(item, "refuse", "I couldn't find this.", True, None, ALIASES)["review_rule_b"]
    assert score_answer(item, "refuse", "Enable WPA4 [1]. Then", False, verdict(), ALIASES, truncated=True)["outcome"]         == "false_answer"  # cut off, but still a false answer


def test_truncated_answer_fails():
    item = q()
    full = "Select ADVANCED > Advanced Setup > Port Forwarding/Port Triggering [1]."
    assert score_answer(item, "single_brand", full, False, verdict(), ALIASES)["pass"]
    s = score_answer(item, "single_brand", full, False, verdict(), ALIASES, truncated=True)
    assert s["outcome"] == "truncated" and not s["pass"]


def test_per_brand_policy_never_rewards_mixing_or_only_asking():
    item = q(category="no_brand", brand=None, policy="per_brand")
    separated = "**Netgear:** Select Router Update [1].\n\n**Tenda:** Choose One-click upgrade [2]."
    assert score_answer(item, "per_brand", separated, False, verdict(), ALIASES)["outcome"] == "per_brand"
    assert not score_answer(item, "per_brand", separated, False, verdict(brand_mixing=True), ALIASES)["pass"]
    asked = "The steps differ by router. Which router brand do you have?"
    s = score_answer(item, "per_brand", asked, False, verdict(), ALIASES)
    assert s["outcome"] == "asked" and not s["pass"] and asks_which_router(asked)


def test_blame_points_at_the_failing_component():
    assert blame(True, True, True, True, True) is None
    assert blame(False, True, True, False, False) == "retrieval"  # even the expected question misses
    assert blame(False, True, False, False, True) == "rewrite"  # expected question would have found it
    assert blame(False, True, True, True, True) == "generation"  # evidence was in the context
    assert blame(False, False, False, None, None) == "rewrite"  # unrelated follow-up carried a brand


def test_rewrite_check():
    assert rewrite_ok(["netgear"], ["netgear"], "How do I delete a keyword?", [["delete", "remove"], "keyword"]) == (True, True)
    assert rewrite_ok([], ["netgear"], "Who won the World Cup?", ["World Cup"]) == (False, True)


# ---------------------------------------------------------------- judge (no API calls)

def test_judge_output_parsing():
    v = parse_verdict('```json\n{"grounded": false, "unsupported_claims": ["x"], "citations_correct": null, '
                      '"citation_problems": [], "brand_mixing": false, "explanation": "e"}\n```')
    assert v.grounded is False and v.citations_correct is None
    for bad in ("not json at all", '{"grounded": "maybe"}', '{"citations_correct": true}'):
        with pytest.raises(JudgeError):
            parse_verdict(bad)


class FakeJudgeLLM:
    model = "fake-judge"

    def __init__(self, text):
        self.text, self.calls = text, []

    def generate(self, system, user):
        from app.services.generation_service import GenerationResult
        self.calls.append(user)
        return GenerationResult(answer=self.text, model=self.model, input_tokens=900, output_tokens=60,
                                reasoning_tokens=0, latency_ms=1.0, finish_reason="stop")


def test_judge_sees_context_question_and_answer():
    llm = FakeJudgeLLM('{"grounded": true, "citations_correct": true, "brand_mixing": false}')
    v, usage = judge(llm, "[1] Brand: netgear | ...", "How?", "Click Add [1].")
    assert v.grounded and usage["input_tokens"] == 900
    assert "SOURCES:\n[1] Brand: netgear" in llm.calls[0] and "ANSWER:\nClick Add [1]." in llm.calls[0]
    with pytest.raises(JudgeError):  # invalid output is an error, never a silent pass
        judge(FakeJudgeLLM("Looks fine to me."), "ctx", "q", "a")


def test_cost_estimate():
    per_call = {"answer_in": 2000, "answer_out": 200, "reasoning": 300, "rewrite_in": 500, "rewrite_out": 50}
    e = estimate(10, 2, 10, per_call, price_in=2.0, price_out=10.0)
    assert e["input_tokens"] == 10 * 2000 + 2 * 500 + 10 * (2000 + 700)
    assert e["output_tokens"] == 10 * 500 + 2 * 50 + 10 * 250
    assert e["cost_usd"] == round((e["input_tokens"] * 2 + e["output_tokens"] * 10) / 1e6, 4)
    assert estimate(1, 0, 0, per_call, None, None)["cost_usd"] is None
    f = estimate(10, 2, 10, per_call, 2.0, 10.0, n_fallback=4)  # one query call + one more answer each
    assert f["input_tokens"] == e["input_tokens"] + 4 * (2200 + 2000)
    assert f["output_tokens"] == e["output_tokens"] + 4 * (80 + 500)
    assert (f["answer_calls"], f["fallback_calls"]) == (14, 4)


def test_turn_needs_expected_query_for_follow_ups():
    from evaluation.dataset import Conversation
    first = Turn(id="c-1", question="How do I block sites on Netgear?", brand="netgear", answerable=True,
                 evidence=[Evidence(document_id="netgear_rax50", pages=(49, 49), key_phrase="Block Sites")])
    follow = Turn(id="c-2", question="Undo that", brand="netgear", answerable=False)
    with pytest.raises(ValueError, match="expected_query"):
        Conversation(id="c", tests="t", turns=[first, follow])


# ---------------------------------------------------------------- planted-error judge tests

def test_planted_errors_change_exactly_what_they_claim():
    from evaluation.judge_tests import changed_value, invented_claim, mutate, other_brand_step, wrong_citation, wrong_menu

    assert wrong_menu("Go to ADVANCED > Security [1].") == "Go to ADVANCED > Wireless Bridge Settings [1]."
    assert changed_value("The IP is 192.168.1.1 [1].") == "The IP is 192.168.1.40 [1]."
    assert changed_value("Hold it for 7 seconds.") == "Hold it for 22 seconds."
    assert changed_value("The 2.4 GHz band.") is None  # numbers inside decimals are left alone
    assert "Reboot Schedule" in other_brand_step("Click Apply [2].", "netgear") and "[2]" in other_brand_step("Click Apply [2].", "netgear")
    assert other_brand_step("Click Apply.", None) is None  # only for single-brand answers
    assert invented_claim("Click Apply [1].").endswith("not applied. [1]")
    context = ("[1] Brand: netgear | Model: RAX50 | n.pdf, pp. 50–51 | Section: x\ntext\n\n"
               "[2] Brand: netgear | Model: RAX50 | n.pdf, p. 51 | Section: y\ntext\n\n"
               "[3] Brand: netgear | Model: RAX50 | n.pdf, p. 90 | Section: z\ntext")
    # [2] overlaps [1]'s pages (could support the same text), so the citation moves to [3]
    assert wrong_citation("Click A [1]. Click B [1].", context) == "Click A [3]. Click B [3]."
    assert mutate("wrong_menu", "No menu here.", "netgear", "") is None


def test_dataset_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.jsonl", tmp_path / "crlf.jsonl"
    lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
    assert dataset.dataset_hash(lf) == dataset.dataset_hash(crlf)


# ---------------------------------------------------------------- held-out set

def test_holdout_is_separate_from_the_development_set():
    holdout = dataset.load_questions(dataset.HOLDOUT_FILE)
    dev = [*dataset.load_questions(), *(t for c in dataset.load_conversations() for t in c.turns)]
    assert len(holdout) == 15 and not {h.id for h in holdout} & {d.id for d in dev}
    dev_pages = {(e.document_id, p) for d in dev for e in d.evidence for p in range(e.pages[0], e.pages[1] + 1)}
    for h in holdout:  # unseen content: no held-out evidence page is used by a development question
        for e in h.evidence:
            assert not {(e.document_id, p) for p in range(e.pages[0], e.pages[1] + 1)} & dev_pages, h.id


@pytest.mark.skipif(not list(Path("documents").glob("*.pdf")), reason="manual PDFs not present")
def test_real_holdout_labels_match_the_manuals():
    from app.services.document_loader import load_all_documents
    from app.services.text_cleaner import clean_documents

    docs, _ = clean_documents(load_all_documents(Settings(_env_file=None).documents_dir))
    assert check_labels(dataset.load_questions(dataset.HOLDOUT_FILE), [], docs).errors == []


def test_holdout_needs_typed_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    with pytest.raises(SystemExit, match="Cancelled"):
        dataset.confirm_holdout()
    monkeypatch.setattr("builtins.input", lambda _: "holdout")
    dataset.confirm_holdout()


# ---------------------------------------------------------------- Step 12 reporting

def test_wilson_interval_and_percentile():
    from evaluation.metrics import percentile, wilson
    lo, hi = wilson(55, 73)
    assert (round(lo, 2), round(hi, 2)) == (0.64, 0.84)
    assert wilson(0, 0) is None and wilson(10, 10)[1] == 1.0
    assert percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([], 50) is None


def test_compare_lists_question_flips(tmp_path):
    import importlib.util
    import json as _json
    spec = importlib.util.spec_from_file_location("eval_compare", "scripts/eval_compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    a, b = tmp_path / "a", tmp_path / "b"
    for run, ranks in ((a, {"q1": 1, "q2": None, "q3": 7}), (b, {"q1": 8, "q2": 2, "q3": 3})):
        run.mkdir()
        run.joinpath("results.jsonl").write_text(
            "\n".join(_json.dumps({"id": k, "answerable": True, "rank": v}) for k, v in ranks.items()), encoding="utf-8")
    oa, ob = compare.outcomes("retrieval", a), compare.outcomes("retrieval", b)
    assert oa == {"q1": True, "q2": False, "q3": False} and ob == {"q1": False, "q2": True, "q3": True}
