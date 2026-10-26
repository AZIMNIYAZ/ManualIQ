"""Contextual headers (Step 12, R2): the text that is EMBEDDED for a chunk gets a short header

    D-Link DIR-X1560 | Website Filter
    <chunk text>

so steps that never name their feature ("1. Click Add...") are embedded with it. Only the
embedding input changes: the stored text, the context sent to the LLM and the citations are
the chunk text exactly as before.

Section labels come from the chunker's nearest heading, which is sometimes an artefact of a
diagram or table ("Modem", "Lifetime", "Figure 7. Cable your router"). A label is used only
if it is trusted; otherwise the last trusted heading before it in the same manual is used
if it is on the same page (fallback; a heading further back is often a different topic: with
any distance "LED Control" got "Quick VPN", with 2 pages "Time Configuration" still did), or the
header has brand and model only (omitted). A missing section loses a little help; a wrong one
points retrieval at the wrong topic:

  trusted  - found in the manual's table of contents (numbering normalised: "4 1", "3.6.2"),
           - a numbered heading ("9.11.5 Auto system maintenance"),
           - a procedure title ("To set up Port Forwarding")
  rejected - "Figure..."/"Table..." captions, "...on page 24", single words not in the
             contents, and any other label not in the contents (conservative)

The embedding model reads at most 512 tokens. Instead of letting it cut the end of a chunk,
the header is shortened to fit: the section is dropped first, then the model, then the whole
header. Chunk text is never truncated.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from app.models.chunk import Chunk
from app.models.document import Document

HEADER_VERSION = "1.1"  # 1.1: fallback only to a trusted heading on the same page
SECTION_MAX_TOKENS = 24
FALLBACK_MAX_PAGES = 0  # same page only: with 2 pages, D-Link "Time Configuration" got "Quick VPN"

NUMBERING_RE = re.compile(r"^\s*\d+(?:[.\s]\d+)*\.?\s+(?=[A-Za-z])")
NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:[.\s]\d+)+\.?\s+[A-Za-z]")  # "3.6.2 URL Filter", "4 1 Use..."
PROCEDURE_RE = re.compile(r"^To\s+\w+(?:\s+\S+){1,}", re.I)
CAPTION_RE = re.compile(r"^(figure|table|step)\b", re.I)
PAGE_REF_RE = re.compile(r"\bon page\b", re.I)


def _norm(text: str) -> str:
    text = text.replace("’", "'").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w' ]+", " ", text)).strip()


def toc_text(doc: Document) -> str:
    """The manual's table-of-contents pages (excluded from search, still in raw_text)."""
    return _norm(" ".join(p.raw_text for p in doc.pages if p.reason == "toc"))


def classify_label(label: str | None, toc: str) -> str:
    """'trusted' or 'rejected' (None counts as rejected: nothing to trust)."""
    if not label or len(re.findall(r"[A-Za-z]", label)) < 3:
        return "rejected"
    if CAPTION_RE.match(label) or PAGE_REF_RE.search(label):
        return "rejected"
    key = _norm(NUMBERING_RE.sub("", label))
    if key and key in toc:
        return "trusted"
    if NUMBERED_HEADING_RE.match(label) or PROCEDURE_RE.match(label.strip()):
        return "trusted"
    return "rejected"


@dataclass
class HeaderReport:
    trusted: int = 0
    fallback: int = 0
    omitted: int = 0
    shortened: Counter = field(default_factory=Counter)  # "no_section", "brand_only", "none"
    header_tokens: list[int] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)  # (document, original label, replacement)

    def as_dict(self) -> dict:
        tokens = sorted(self.header_tokens)
        return {
            "header_version": HEADER_VERSION,
            "sections": {"trusted": self.trusted, "fallback": self.fallback, "omitted": self.omitted},
            "headers_shortened_to_fit": dict(self.shortened),
            "header_tokens_median": tokens[len(tokens) // 2] if tokens else 0,
            "header_tokens_max": tokens[-1] if tokens else 0,
            "rejected_labels": [{"document": d, "label": lab, "replaced_by": rep, "chunks": n}
                                for (d, lab, rep), n in sorted(self.rejected.items(), key=lambda kv: (kv[0][0], -kv[1]))],
        }


def resolve_sections(chunks: list[Chunk], docs: list[Document], report: HeaderReport) -> dict[str, str | None]:
    """chunk_id -> the section label to put in the header (None: brand and model only)."""
    tocs = {d.document_id: toc_text(d) for d in docs}
    resolved: dict[str, str | None] = {}
    last_trusted: dict[str, tuple[str, int]] = {}  # document -> (heading, page where it was trusted)
    for c in sorted(chunks, key=lambda c: (c.document_id, c.chunk_index)):
        if classify_label(c.section, tocs.get(c.document_id, "")) == "trusted":
            resolved[c.chunk_id] = c.section.strip()
            last_trusted[c.document_id] = (c.section.strip(), c.page_start)
            report.trusted += 1
            continue
        previous = last_trusted.get(c.document_id)
        replacement = previous[0] if previous and c.page_start - previous[1] <= FALLBACK_MAX_PAGES else None
        resolved[c.chunk_id] = replacement
        if replacement:
            report.fallback += 1
        else:
            report.omitted += 1
        if c.section:
            report.rejected[(c.document_id, c.section.strip(), replacement or "(none)")] += 1
    return resolved


def _cap_section(section: str, count: Callable[[str], int]) -> str:
    words = section.split()
    while words and count(" ".join(words)) > SECTION_MAX_TOKENS:
        words.pop()
    return " ".join(words)


def build_embedding_texts(
    chunks: list[Chunk],
    docs: list[Document],
    brand_names: dict[str, str],
    count_with_special_tokens: Callable[[str], int],
    max_tokens: int = 512,
) -> tuple[list[str], list[str], HeaderReport]:
    """(texts to embed, header used per chunk, report), in the order of `chunks`."""
    report = HeaderReport()
    sections = resolve_sections(chunks, docs, report)
    models = {d.document_id: d.model for d in docs}
    texts, headers = [], []
    for c in chunks:
        brand = brand_names.get(c.brand, c.brand)
        model = models.get(c.document_id, c.model)
        section = _cap_section(sections[c.chunk_id], count_with_special_tokens) if sections[c.chunk_id] else ""
        options = ([(f"{brand} {model} | {section}", None)] if section else []) + [
            (f"{brand} {model}", "no_section"), (brand, "brand_only"), ("", "none")]
        for header, shortened in options:
            text = f"{header}\n{c.text}" if header else c.text
            if count_with_special_tokens(text) <= max_tokens:
                break
        if shortened and (section or shortened != "no_section"):
            report.shortened[shortened] += 1
        report.header_tokens.append(count_with_special_tokens(header) - 2 if header else 0)
        texts.append(text)
        headers.append(header)
    return texts, headers, report
