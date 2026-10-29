"""Retrieval fallback (Step 12): one more search when the answer model refused.

The failure analysis showed questions in everyday wording ("kick it off for good") whose
evidence ranks ~100th, while the manual's wording ("add a client to the blacklist") ranks it
1st. When the answer is a refusal, an LLM that is given the manual's own feature names (the
glossary, from the tables of contents) writes a few alternative search queries; the original
question and the alternatives are searched, merged, and the answer model gets ONE more
attempt at the ORIGINAL question. No loop: a second refusal is final.

Why refusal is the trigger: answered questions never enter the fallback, so they cannot be
changed by it. Known limit: an answer built on wrong-but-similar pages does not trigger it.

Safeguards on each alternative (it is dropped if any fails; with none left the refusal stands):
- valid JSON, at most MAX_QUERY_CHARS, one line
- brand guard: it may not name a brand the question does not name (the brand FILTER is the
  original search's, so it cannot change anyway)
- technical terms of the question (product names, protocols, versions, numbers) must
  survive, so an unsupported feature ("Alexa", "AiMesh", "Wi-Fi 7") is never swapped for a
  supported one. Brand and model names are exempt: the filter already carries them.
The prompt is generic and the glossary holds every heading of every manual: nothing in it
comes from the evaluation questions.
"""

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from app.models.search import SearchResult
from app.services.brand_detection import AliasTable, detect_brands
from app.services.generation_service import GenerationService

FALLBACK_PROMPT_VERSION = "1.0"
MAX_QUERY_CHARS = 300
RRF_K = 60  # the constant from the original reciprocal rank fusion paper

FALLBACK_PROMPT = """A search of home Wi-Fi router user manuals found nothing that answers the user's question. Often the user describes what they want in everyday words, while the manual uses a feature name. You get the manuals' contents (their section headings) and the question.

Write {n} different search queries in the manual's vocabulary. Reply with JSON only, no other text: {{"queries": ["...", "..."]}}

Rules:
1. Keep exactly what the user wants to do. Do not answer the question.
2. Where a heading from the contents fits what the user wants, use its words. Try a different fitting heading or wording in each query.
3. Copy every product name, protocol, standard, version and number from the question exactly. If the question names a feature, keep that feature: never replace it with a different, related feature.
4. Do not add a router brand the question does not name.
5. If the question is not about routers or home networking, reply {{"queries": []}}.
6. Each query at most 25 words."""

GENERIC_TERMS = {"wifi", "wlan"}  # name no feature ("Wi-Fi 7" stays protected through the "7")
TECH_TERM_RE = re.compile(r"\b[\w./+-]*(?:\d|[A-Z][\w./+-]*[A-Z]|[a-z][A-Z])[\w./+-]*\b")


class _Output(BaseModel):
    queries: list[str] = Field(..., max_length=10)


@dataclass
class FallbackQueries:
    queries: list[str]  # the alternatives that passed every check
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (query, reason)
    error: str | None = None  # the call failed or the output was not valid JSON
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


def _squash(text: str) -> str:
    return re.sub(r"[\s\-_.]+", "", text).casefold()


def technical_terms(text: str, exempt: set[str] = frozenset()) -> list[str]:
    """exempt: squashed brand/model names, which need not survive."""
    return [t for t in TECH_TERM_RE.findall(text)
            if (len(t) > 1 or t.isdigit()) and _squash(t) not in GENERIC_TERMS and _squash(t) not in exempt]


def exempt_names(names: list[str]) -> set[str]:
    """Squashed words of the manuals' names ("D-Link DIR-X1560" -> {"dlink", "dirx1560"}), except
    plain numbers: "6" in "Wi-Fi 6 series" must not exempt the "6" of "6 GHz"."""
    return {_squash(w) for name in names for w in name.split() if not w.isdigit()}


def validate(original: str, query: str, aliases: AliasTable, exempt: set[str]) -> str | None:
    """None if the alternative is acceptable, else the reason it is not."""
    if not query or len(query) > MAX_QUERY_CHARS or "\n" in query:
        return "empty, too long or multi-line"
    added = set(detect_brands(query, aliases)) - set(detect_brands(original, aliases))
    if added:
        return f"names a brand the question does not: {sorted(added)}"
    squashed = _squash(query)
    dropped = [t for t in technical_terms(original, exempt) if _squash(t) not in squashed]
    if dropped:
        return f"dropped technical terms: {dropped}"
    return None


def alternative_queries(question: str, glossary: str, llm: GenerationService, aliases: AliasTable,
                        exempt: set[str], n: int) -> FallbackQueries:
    started = time.perf_counter()
    try:
        generated = llm.generate(FALLBACK_PROMPT.format(n=n), f"Manual contents:\n{glossary}\n\nQuestion: {question}")
    except Exception as e:  # the refusal stands
        return FallbackQueries([], error=f"call failed: {e}", latency_ms=round((time.perf_counter() - started) * 1000, 1))
    usage = {"model": generated.model, "input_tokens": generated.input_tokens,
             "output_tokens": generated.output_tokens, "latency_ms": generated.latency_ms}
    try:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", generated.answer.strip())
        raw = _Output(**json.loads(text)).queries
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return FallbackQueries([], error=f"invalid output: {type(e).__name__}", **usage)
    kept, rejected = [], []
    for q in (q.strip() for q in raw):
        problem = validate(question, q, aliases, exempt)
        if problem:
            rejected.append((q, problem))
        elif _squash(q) not in {_squash(k) for k in kept} | {_squash(question)}:
            kept.append(q)
    return FallbackQueries(kept[:n], rejected, **usage)


def merge(lists: list[list[SearchResult]], method: str, top_k: int) -> list[SearchResult]:
    """Each chunk once. rrf: sum of 1/(RRF_K + rank) over the lists (positions only). max: the
    chunk's best cosine score (comparable: one embedding model, one collection)."""
    best: dict[str, SearchResult] = {}
    fused: dict[str, float] = {}
    for results in lists:
        for r in results:
            if r.chunk_id not in best or r.score > best[r.chunk_id].score:
                best[r.chunk_id] = r
            fused[r.chunk_id] = fused.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + r.rank)
    key = (lambda cid: fused[cid]) if method == "rrf" else (lambda cid: best[cid].score)
    order = sorted(best, key=key, reverse=True)[:top_k]
    return [best[cid].model_copy(update={"rank": i}) for i, cid in enumerate(order, start=1)]
