"""How one answer is scored (pure functions, tested without any API call).

Policies (labelled per question):
- single_brand: pass = answered, every key fact present, no wrong-procedure text, and the
  judge found it grounded (when judged).
- per_brand (no brand named): pass = answered, grounded, no brand mixing, and not merely
  asking which router. Asking is recorded as its own outcome ("asked"), not as a pass;
  mixing brands is always a fail.
- refuse: pass = the fixed not-found answer.

Conversation turns also get a blame when they fail: "rewrite" (the expected question would
have retrieved the evidence, the actual rewrite did not), "retrieval" (even the expected
question misses it), or "generation" (the evidence was in the context).
"""

from app.services.brand_detection import AliasTable
from evaluation.dataset import Item, contains, fact_label
from evaluation.judge import JudgeVerdict
from evaluation.metrics import asks_which_router, brands_named, key_facts, violations


def score_answer(item: Item, policy: str, answer: str, refused: bool, verdict: JudgeVerdict | None,
                 aliases: AliasTable) -> dict:
    named = brands_named(answer, aliases) if not refused else []
    asked = (not refused) and asks_which_router(answer)
    if refused:
        found, missing = [], [fact_label(f) for f in item.key_facts]
    else:
        found, missing = key_facts(answer, item.key_facts)
    bad = violations(answer, item.must_not_contain) if not refused else []
    grounded = verdict.grounded if verdict else None
    mixing = verdict.brand_mixing if verdict else None

    if not item.answerable:
        outcome = "correct_refusal" if refused else "false_answer"
        passed = refused
    elif refused:
        outcome, passed = "false_refusal", False
    elif policy == "per_brand":
        outcome = "asked" if asked and len(named) < 2 else ("per_brand" if len(named) >= 2 else "one_brand")
        passed = outcome != "asked" and mixing is not True and grounded is not False
    else:
        outcome = "answered"
        passed = not missing and not bad and grounded is not False and mixing is not True

    return {
        "outcome": outcome,
        "pass": passed,
        "key_facts_found": found,
        "key_facts_missing": missing,
        "key_fact_coverage": round(len(found) / len(item.key_facts), 3) if item.key_facts and item.answerable else None,
        "must_not_found": bad,
        "brands_named": named,
        "other_brands_named": [b for b in named if item.brand and b != item.brand],
        "asked_which_router": asked,
        "grounded": grounded,
        "citations_correct": verdict.citations_correct if verdict else None,
        "brand_mixing": mixing,
    }


def blame(passed: bool, answerable: bool, rewrite_ok: bool, actual_hit: bool | None,
          oracle_hit: bool | None) -> str | None:
    """Which component a failed conversation turn is blamed on."""
    if passed:
        return None
    if not answerable:  # should have refused: wrong brand carried over, or the model answered anyway
        return "rewrite" if not rewrite_ok else "generation"
    if oracle_hit is False:
        return "retrieval"
    if actual_hit is False:
        return "rewrite"
    return "generation"


def rewrite_ok(expected_brands: list[str], got_brands: list[str], query: str, required_terms: list) -> tuple[bool, bool]:
    """(brand correct, required terms present) for a follow-up's rewritten query."""
    return sorted(got_brands) == sorted(expected_brands), all(contains(query, t) for t in required_terms)


# ------------------------------------------------------------------ cost estimate

DEFAULT_TOKENS = {  # per call, used when the API log has no measurements yet
    "answer_in": 2600, "answer_out": 250, "reasoning": 300, "rewrite_in": 700, "rewrite_out": 40,
}
JUDGE_EXTRA_INPUT = 700  # judge prompt + the answer, on top of the same context


def estimate(n_answers: int, n_rewrites: int, n_judge: int, per_call: dict, price_in: float | None,
             price_out: float | None) -> dict:
    """Upper-bound token and cost estimate. Judge calls are counted for every answer, although
    refusals are not judged. One price pair is applied to all models (check xAI's pricing)."""
    tokens_in = (n_answers * per_call["answer_in"] + n_rewrites * per_call["rewrite_in"]
                 + n_judge * (per_call["answer_in"] + JUDGE_EXTRA_INPUT))
    tokens_out = (n_answers * (per_call["answer_out"] + per_call["reasoning"]) + n_rewrites * per_call["rewrite_out"]
                  + n_judge * 250)
    cost = None
    if price_in is not None and price_out is not None:
        cost = round((tokens_in * price_in + tokens_out * price_out) / 1e6, 4)
    return {"answer_calls": n_answers, "rewrite_calls": n_rewrites, "judge_calls": n_judge,
            "input_tokens": tokens_in, "output_tokens": tokens_out, "cost_usd": cost}
