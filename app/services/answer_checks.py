"""Checks an answer against the sources it was generated from.

Citations are references, not proof: these checks catch the cheap, specific failures
automatically, and manual review (or an LLM judge in evaluation) is still needed to
confirm that whole statements are supported.

- invalid_citation: [n] that does not exist in the context (a hallucinated reference)
- no_citations: an answer that cites nothing (breaks the prompt's rule 2)
- unsupported_value: an IP address, menu path, URL/domain or number with a unit that
  appears in none of the sources (e.g. a default IP from the model's general knowledge)
- value_in_uncited_source: the value exists, but only in a source the answer did not cite
"""

import re
from dataclasses import dataclass, field

from app.services.brand_detection import URL_RE
from app.services.context_builder import Context
from app.services.prompts import NOT_FOUND_ANSWER

CITATION_RE = re.compile(r"\[(\d{1,2})\]")
IP_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:seconds?|minutes?|hours?|days?|clients?|devices?|characters?|"
    r"MB|GB|Mbps|Gbps|GHz|MHz|ms|%)(?!\w)",
    re.IGNORECASE,
)
# "ADVANCED > Setup": the word directly before and after each ">" must appear the same way in a source
MENU_STEP_RE = re.compile(r"([\w/&().-]+)\s*>\s*([\w/&().-]+)")


@dataclass
class AnswerCheck:
    not_found: bool
    cited: list[int] = field(default_factory=list)  # valid citation numbers, in order of first use
    warnings: list[dict] = field(default_factory=list)


def is_not_found(answer: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()  # noqa: E731
    return norm(NOT_FOUND_ANSWER) in norm(answer) and len(answer) < 2 * len(NOT_FOUND_ANSWER)


def _normalise(text: str) -> str:
    text = re.sub(r"\s*>\s*", ">", text.lower())
    return re.sub(r"\s+", " ", text)


def _values(answer: str) -> list[tuple[str, str]]:
    """(kind, value) pairs to verify, with citation markers removed first."""
    text = CITATION_RE.sub(" ", answer)
    values = [("ip_address", m) for m in IP_RE.findall(text)]
    values += [("url", m.rstrip(".,;:)")) for m in (u.group(0) for u in URL_RE.finditer(text))]
    values += [("number", m.group(0)) for m in NUMBER_UNIT_RE.finditer(text)]
    values += [("menu_path", f"{a} > {b}") for a, b in MENU_STEP_RE.findall(text)]
    seen, unique = set(), []
    for kind, value in values:
        if (kind, value.lower()) not in seen:
            seen.add((kind, value.lower()))
            unique.append((kind, value))
    return unique


def _present(kind: str, value: str, normalised_text: str) -> bool:
    v = _normalise(value)
    if kind == "number":
        return v.replace(" ", "") in normalised_text.replace(" ", "")
    if kind == "url":
        return v.split("://")[-1].removeprefix("www.").rstrip("/") in normalised_text
    return v in normalised_text


def check_answer(answer: str, context: Context) -> AnswerCheck:
    if is_not_found(answer):
        return AnswerCheck(not_found=True)

    valid = {s.n for s in context.sources}
    cited, invalid = [], []
    for n in (int(x) for x in CITATION_RE.findall(answer)):
        target = cited if n in valid else invalid
        if n not in target:
            target.append(n)

    check = AnswerCheck(not_found=False, cited=cited)
    for n in invalid:
        check.warnings.append({"type": "invalid_citation", "detail": f"[{n}] does not exist ({len(valid)} sources)"})
    if not cited:
        check.warnings.append({"type": "no_citations", "detail": "The answer cites no source"})

    by_n = {s.n: _normalise(s.result.text) for s in context.sources}
    cited_text = " ".join(by_n[n] for n in cited)
    all_text = " ".join(by_n.values())
    for kind, value in _values(answer):
        if _present(kind, value, cited_text):
            continue
        if _present(kind, value, all_text):
            check.warnings.append({"type": "value_in_uncited_source", "detail": f"{kind} {value!r}"})
        else:
            check.warnings.append({"type": "unsupported_value", "detail": f"{kind} {value!r} is not in any source"})
    return check
