"""Metric functions (pure, no services): retrieval matching, Hit@K/MRR, answer checks.

A retrieved chunk matches a gold evidence location when it is from the same manual, its
pages overlap the evidence pages, and it contains the key phrase. Evidence locations are
alternatives, so a question is a hit at K if any of the top K chunks matches any of them.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from app.services.brand_detection import AliasTable, detect_brands
from evaluation.dataset import Evidence, KeyFact, contains, fact_label, normalise

K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class Retrieved:
    chunk_id: str
    document_id: str
    brand: str
    page_start: int
    page_end: int
    content_type: str
    score: float
    text: str


def matches(chunk: Retrieved, ev: Evidence) -> bool:
    return (chunk.document_id == ev.document_id
            and chunk.page_start <= ev.pages[1] and chunk.page_end >= ev.pages[0]
            and normalise(ev.key_phrase) in normalise(chunk.text))


def first_hit_rank(results: list[Retrieved], evidence: list[Evidence]) -> int | None:
    """1-based rank of the first chunk matching any evidence location; None if none does."""
    for rank, chunk in enumerate(results, 1):
        if any(matches(chunk, ev) for ev in evidence):
            return rank
    return None


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def matched_documents(results: list[Retrieved], evidence: list[Evidence], k: int) -> set[str]:
    """Manuals with a matching chunk in the top k (for per-brand questions)."""
    return {ev.document_id for chunk in results[:k] for ev in evidence if matches(chunk, ev)}


def retrieval_summary(rows: list[dict]) -> dict:
    """rows: {"rank": int|None} per answerable question. Hit@K and MRR (cut off at the last K)."""
    n = len(rows)
    if not n:
        return {"n": 0}
    summary = {"n": n}
    for k in K_VALUES:
        summary[f"hit@{k}"] = round(sum(hit_at(r["rank"], k) for r in rows) / n, 3)
    summary["hit@5_ci95"] = wilson(sum(hit_at(r["rank"], 5) for r in rows), n)
    summary["mrr"] = round(sum(reciprocal_rank(r["rank"]) for r in rows) / n, 3)
    return summary


# ------------------------------------------------------------------ answers

def key_facts(answer: str, facts: list[KeyFact]) -> tuple[list[str], list[str]]:
    """(found, missing) key facts, as labels."""
    found = [fact_label(f) for f in facts if contains(answer, f)]
    missing = [fact_label(f) for f in facts if not contains(answer, f)]
    return found, missing


def violations(answer: str, must_not: list[str]) -> list[str]:
    return [m for m in must_not if contains(answer, m)]


ASK_RE = re.compile(r"\bwhich\b[^.?!\n]{0,60}\b(router|brand|model|device)\b[^.!\n]*\?", re.I)


def asks_which_router(answer: str) -> bool:
    return bool(ASK_RE.search(answer))


def brands_named(answer: str, aliases: AliasTable) -> list[str]:
    return detect_brands(answer, aliases)


def refusal_counts(rows: list[dict]) -> dict:
    """rows: {"answerable": bool, "refused": bool}.

    false_answers = answered when the manuals do not contain the answer (the dangerous error);
    false_refusals = refused although the answer was in the manuals."""
    c = defaultdict(int)
    for r in rows:
        if r["answerable"]:
            c["false_refusals" if r["refused"] else "answered"] += 1
        else:
            c["correct_refusals" if r["refused"] else "false_answers"] += 1
    refused = c["correct_refusals"] + c["false_refusals"]
    should_refuse = c["correct_refusals"] + c["false_answers"]
    return {
        **{k: c[k] for k in ("answered", "false_refusals", "correct_refusals", "false_answers")},
        "refusal_precision": round(c["correct_refusals"] / refused, 3) if refused else None,
        "refusal_recall": round(c["correct_refusals"] / should_refuse, 3) if should_refuse else None,
    }


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson confidence interval for a proportion: with 40 questions, small differences
    are inside the noise, and the interval shows it (e.g. 55/73 -> 64%-84%)."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """Judge vs human on one binary dimension: raw agreement and Cohen's kappa."""
    n = len(pairs)
    if not n:
        return {"n": 0}
    agree = sum(a == b for a, b in pairs) / n
    p_judge = sum(a for a, _ in pairs) / n
    p_human = sum(b for _, b in pairs) / n
    expected = p_judge * p_human + (1 - p_judge) * (1 - p_human)
    kappa = None if expected == 1 else round((agree - expected) / (1 - expected), 3)
    return {"n": n, "agreement": round(agree, 3), "kappa": kappa,
            "disagreements": sum(a != b for a, b in pairs)}
