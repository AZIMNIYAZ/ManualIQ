"""Turns a follow-up ("What about on ASUS?") into a standalone search query using the
conversation history, and decides which brands the new question is about.

Runtime validation (the output falls back to the original message if it fails):
- valid JSON with a non-empty query of at most 1,000 characters
- the query must look like a question, not an answer (no line breaks or numbered steps,
  at most 300 characters)
- brands must exist in the collection (unknown ones are dropped)
- brand-switch guard: if the latest message itself names a brand, that brand wins

Whether the meaning is preserved, or an unrelated question correctly did not inherit the
previous topic, cannot be checked per response: that is evaluated with scripted
conversations (scripts/conversation_check.py) and in Step 11.
"""

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from app.models.session import StoredMessage
from app.services.answer_checks import CITATION_RE
from app.services.brand_detection import AliasTable, detect_brands
from app.services.generation_service import GenerationService

REWRITE_PROMPT_VERSION = "1.1"  # 1.1: vague actions ("undo that") made explicit in the conversation's terms
MAX_QUERY_CHARS = 300  # longer "queries" are usually answers, not questions

REWRITE_SYSTEM_PROMPT = """You rewrite the latest user message of a conversation about home Wi-Fi routers into a standalone search query.

Reply with JSON only, no other text: {{"query": "...", "brands": [...]}}

Rules:
1. "query" is the latest user message rewritten so it can be understood without the conversation: replace words like "it", "that" or "there" with what they refer to. Keep its meaning; do not add facts.
2. Replace vague actions with the explicit action they mean, using the terms from the conversation: after blocking a device by adding it to the blacklist, "How do I undo that?" means "How do I remove the device from the blacklist?".
3. "brands" lists the router brands the NEW question is about, using only these identifiers: {brands}.
4. If the latest message names a brand, use only that brand: a switch ("What about on ASUS?") replaces the previous brand.
5. If the latest message continues the same topic without naming a brand ("Can I edit it later?"), keep the previous brand.
6. If the latest message is unrelated to the conversation, rewrite it on its own and do not carry over the previous brand or topic.
7. Do not answer the question. Only rewrite it."""


class RewriteOutput(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    brands: list[str] = []


@dataclass
class RewriteResult:
    query: str  # what to search for
    brands: list[str] | None  # None: no decision (fall back to alias detection on the query)
    rewritten: bool  # False if the original message is used (no history, or the rewrite failed)
    warnings: list[dict] = field(default_factory=list)
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


def format_history(history: list[StoredMessage], answer_chars: int) -> str:
    lines = []
    for m in history:
        if m.role == "user":
            lines.append(f"User: {m.content}")
        else:
            text = " ".join(CITATION_RE.sub("", m.content).split())
            if len(text) > answer_chars:
                text = text[:answer_chars] + " ..."
            topic = f" (searched: {m.rewritten_query})" if m.rewritten_query else ""
            lines.append(f"Assistant{topic}: {text}")
    return "\n".join(lines)


def _parse(text: str) -> RewriteOutput:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)  # tolerate a fenced code block
    return RewriteOutput.model_validate(json.loads(text))


def _looks_like_answer(query: str) -> bool:
    return "\n" in query or bool(re.search(r"(^|\s)\d+\.\s", query)) or len(query) > MAX_QUERY_CHARS


def rewrite(
    message: str,
    history: list[StoredMessage],
    rewriter: GenerationService,
    valid_brands: list[str],
    aliases: AliasTable,
    answer_chars: int = 300,
) -> RewriteResult:
    system = REWRITE_SYSTEM_PROMPT.format(brands=", ".join(sorted(valid_brands)))
    user = f"Conversation:\n{format_history(history, answer_chars)}\n\nLatest user message: {message}"

    def fallback(reason: str, **usage) -> RewriteResult:
        return RewriteResult(query=message, brands=None, rewritten=False,
                             warnings=[{"type": "rewrite_failed", "detail": reason}], **usage)

    try:
        generated = rewriter.generate(system, user)
    except Exception as e:  # the chat still answers, using the original message
        return fallback(f"rewrite call failed: {e}")
    usage = {"model": generated.model, "input_tokens": generated.input_tokens,
             "output_tokens": generated.output_tokens, "latency_ms": generated.latency_ms}

    try:
        output = _parse(generated.answer)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        return fallback(f"invalid rewrite output: {type(e).__name__}", **usage)
    query = output.query.strip()
    if not query or _looks_like_answer(query):
        return fallback("rewrite looks like an answer, not a question", **usage)

    warnings = []
    brands = sorted({b.lower() for b in output.brands})
    unknown = [b for b in brands if b not in valid_brands]
    if unknown:
        warnings.append({"type": "rewrite_unknown_brand", "detail": f"dropped {unknown}"})
        brands = [b for b in brands if b in valid_brands]
    named = [b for b in detect_brands(message, aliases) if b in valid_brands]
    if named and named != brands:  # the brand-switch guard
        warnings.append({"type": "rewrite_brand_corrected", "detail": f"{brands} -> {named} (named in the message)"})
        brands = named
    return RewriteResult(query=query, brands=brands, rewritten=True, warnings=warnings, **usage)
