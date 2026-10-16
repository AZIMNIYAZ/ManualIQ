"""LLM judge for groundedness, citation correctness and brand mixing.

The judge sees exactly what the answer model saw (the numbered context), the question and
the answer, and judges against the sources only, never against its own knowledge.

Limits, stated in the write-up: the judge is an LLM from the same family as the answer
model (a different, non-reasoning model at temperature 0), so it can share blind spots.
Its verdicts are only used if they agree with the human grades (scripts/judge_agreement.py),
reported separately for groundedness and for citations.

An unparseable verdict is recorded as an error and counted, never treated as a pass.
"""

import json
import re

from pydantic import BaseModel, ValidationError

from app.services.generation_service import GenerationService

JUDGE_PROMPT_VERSION = "1.0"
JUDGE_MAX_OUTPUT_TOKENS = 600

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a support assistant that answers questions about home Wi-Fi routers from numbered manual excerpts (SOURCES).

Judge the ANSWER only against the SOURCES. Do not use your own knowledge of routers: a claim that is true in real life but not stated in the SOURCES is unsupported.

Return only a JSON object with these keys:
{"grounded": true or false,
 "unsupported_claims": ["short quote of each unsupported claim"],
 "citations_correct": true, false or null,
 "citation_problems": ["[n]: what is wrong"],
 "brand_mixing": true or false,
 "explanation": "one or two sentences"}

Rules:
1. grounded is true only if every factual claim in the ANSWER (steps, menu paths, button and field names, values, limits, warnings) is stated in the SOURCES. Paraphrasing is fine. Obvious connecting words ("then", "first log in") are fine when the SOURCES imply them.
2. A claim that appears in the SOURCES but for a different brand, model, operating system, band or procedure than the one the ANSWER applies it to is unsupported.
3. citations_correct is true if every [n] marker points to a source that supports the sentence or step it is attached to; false if any marker points to a source that does not support it; null if the ANSWER has no [n] markers.
4. brand_mixing is true if the ANSWER combines steps or values from different brands into one procedure, or presents one brand's steps as another brand's. Separate answers under clearly labelled brand headings are not mixing.
5. If the ANSWER only says the information could not be found, return grounded true, citations_correct null, brand_mixing false."""


class JudgeVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = []
    citations_correct: bool | None
    citation_problems: list[str] = []
    brand_mixing: bool
    explanation: str = ""


class JudgeError(Exception):
    """The judge's output could not be parsed or validated."""


def judge_user_message(context: str, question: str, answer: str) -> str:
    return f"SOURCES:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"


def parse_verdict(text: str) -> JudgeVerdict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise JudgeError(f"no JSON object in judge output: {text[:200]!r}")
    try:
        return JudgeVerdict(**json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise JudgeError(f"invalid judge output ({e.__class__.__name__}): {text[:200]!r}") from e


def judge(judge_llm: GenerationService, context: str, question: str, answer: str) -> tuple[JudgeVerdict, dict]:
    """Returns the verdict and the call's usage; raises JudgeError on unusable output."""
    result = judge_llm.generate(JUDGE_SYSTEM_PROMPT, judge_user_message(context, question, answer))
    usage = {"model": result.model, "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
             "reasoning_tokens": result.reasoning_tokens, "latency_ms": result.latency_ms}
    return parse_verdict(result.answer), usage
