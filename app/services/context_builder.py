"""Turns retrieved chunks into the numbered, labelled sources the LLM answers from.

Each source states its brand, model, citation and section, so the model can see which
router each passage belongs to. Sources are added in rank order until the token budget
is reached; a chunk is never cut in the middle (lower-ranked whole chunks are dropped).
Tokens are counted with the bge tokenizer: an approximation of the LLM's own count.
"""

from dataclasses import dataclass

from app.models.search import SearchResult
from app.services.tokenizer import count_tokens


@dataclass(frozen=True)
class ContextSource:
    n: int  # the number the LLM cites: [n]
    result: SearchResult


@dataclass(frozen=True)
class Context:
    text: str
    sources: list[ContextSource]
    tokens: int
    dropped: int  # retrieved chunks left out because of the budget


def format_source(n: int, r: SearchResult) -> str:
    header = f"[{n}] Brand: {r.brand} | Model: {r.model} | {r.citation}"
    if r.section:
        header += f" | Section: {r.section}"
    return f"{header}\n{r.text}"


def build_context(results: list[SearchResult], max_tokens: int) -> Context:
    blocks, sources, total = [], [], 0
    for r in results:
        block = format_source(len(sources) + 1, r)
        tokens = count_tokens(block)
        if sources and total + tokens > max_tokens:
            break  # keep rank order: everything below this rank is dropped too
        blocks.append(block)
        sources.append(ContextSource(n=len(sources) + 1, result=r))
        total += tokens
    return Context(text="\n\n".join(blocks), sources=sources, tokens=total, dropped=len(results) - len(sources))
