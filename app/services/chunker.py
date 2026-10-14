"""Splits cleaned documents into chunks for embedding.

1. Segments: runs of consecutive indexed pages with the same content_type.
   A chunk never crosses an excluded page or mixes regulatory and normal text.
2. Units: PDF lines are visual, so wrapped lines are merged back into logical
   units (sentence, step, bullet, heading, procedure intro), each knowing its pages.
3. Packing: units are added until the next one would exceed the chunk size; the
   chunk is then cut at the best nearby boundary (heading > procedure > step >
   sentence), as long as the chunk is at least MIN_FILL full.
4. Overlap: the next chunk starts with the last whole units of the previous
   one, up to the overlap size.

Sizes are measured with bge's tokenizer, so no chunk exceeds what bge reads.
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass

from app.core.config import MAX_CHUNK_TOKENS
from app.models.chunk import Chunk, ChunkingReport
from app.models.document import Document, Page
from app.services.text_cleaner import CLEANER_VERSION
from app.services.tokenizer import count_tokens, get_tokenizer

MIN_FILL = 0.6  # a chunk must be this full before an earlier, better cut point is preferred...
MIN_FILL_HEADING = 0.3  # ...except at a heading: a topic boundary is worth a less-full chunk
MIN_CHUNK_TOKENS = 20  # smaller chunks (e.g. a cover page on its own) are dropped
TAIL_MERGE_SHARE = 0.25  # a segment's last chunk with less new text than this is merged backwards
MAX_HEADING_WORDS = 10

# "1. ", "1 ) ", "10.In" (no space), "Step 4", "(2) ", "A. "
STEP_RE = re.compile(r"^(\d{1,2}\s?[.)](\s|$)|\d{1,2}\.(?=[A-Z])|Step\s?\d+\b|\(\d{1,2}\)\s|[A-H]\.\s)")
FIRST_STEP_RE = re.compile(r"^(1\s?[.)]|Step\s?1\b|\(1\)|A\.)")
BARE_STEP_RE = re.compile(r"^(Step\s?\d+|\d{1,2}\s?[.)])$")  # "Step 3" alone on a line
PROCEDURE_RE = re.compile(r"^To\s.{3,100}:$")  # "To set up a default DMZ server:"
# "A.1 Factory settings", "3.6 Firewall", "10.4 Forgot my password"; needs a real word, so a
# table row like "2.4 G 2.4 - 2.4835 GHz 100 mW" does not count
NUMBERED_HEADING_RE = re.compile(r"^(?:[A-H]|\d{1,2})(?:\.\d{1,2}){1,3}\.?\s*[A-Z][A-Za-z]*[a-z]{2,}")
ENDS_SENTENCE_RE = re.compile(r"[.!?:][\"'’”)\]]*$")
CONNECTOR_END_RE = re.compile(
    r"(,|-|\b(and|or|the|a|an|to|of|for|in|on|with|from|by|your|that|is|are|be|as|at|this|if|then))$", re.I
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")

# Lower is better when choosing where to cut
CUT_PRIORITY = {"heading": 0, "procedure": 1, "step": 2, "sentence": 3, "mid_sentence": 4}


@dataclass
class Unit:
    text: str
    kind: str  # heading | procedure | step | bullet | text
    page_start: int
    page_end: int
    tokens: int = 0

    @property
    def ends_sentence(self) -> bool:
        return bool(ENDS_SENTENCE_RE.search(self.text))


# ---------------------------------------------------------------- 1. segments
def _segments(doc: Document) -> list[list[Page]]:
    segments: list[list[Page]] = []
    current: list[Page] = []
    for page in doc.pages:
        if page.status != "indexed":
            if current:
                segments.append(current)
            current = []
            continue
        if current and current[-1].content_type != page.content_type:
            segments.append(current)
            current = []
        current.append(page)
    if current:
        segments.append(current)
    return segments


# ---------------------------------------------------------------- 2. units
def _line_kind(line: str) -> str:
    if PROCEDURE_RE.match(line):
        return "procedure"
    if line.startswith("- "):
        return "bullet"
    if STEP_RE.match(line):
        return "step"
    return "text"


def _continues(prev: Unit, line: str) -> bool:
    """Is `line` the wrapped continuation of `prev`?"""
    if BARE_STEP_RE.match(prev.text):
        return True  # "Step 3" + "Tap the Cloud Service icon."
    if prev.ends_sentence:
        return False
    return line[:1].islower() or bool(CONNECTOR_END_RE.search(prev.text))


def _is_heading_like(text: str) -> bool:
    words = text.split()
    return (
        0 < len(words) <= MAX_HEADING_WORDS
        and (text[:1].isupper() or text[:1].isdigit())
        and any(c.isalpha() for c in text)
        and not re.search(r"[.,;:!?]$", text)
        and not CONNECTOR_END_RE.search(text)
    )


def _build_units(pages: list[Page]) -> list[Unit]:
    units: list[Unit] = []
    for page in pages:
        for line in page.clean_text.splitlines():
            line = line.strip()
            if not line:
                continue
            kind = _line_kind(line)
            if units and kind == "text" and _continues(units[-1], line):
                units[-1].text = f"{units[-1].text} {line}"
                units[-1].page_end = page.page_number
            else:
                units.append(Unit(text=line, kind=kind, page_start=page.page_number, page_end=page.page_number))

    # Headings: short title-like lines followed by a sentence, step or procedure
    for i, unit in enumerate(units):
        nxt = units[i + 1] if i + 1 < len(units) else None
        if unit.kind == "text" and _is_heading_like(unit.text) and (
            NUMBERED_HEADING_RE.match(unit.text)  # "A.1 Factory settings": a heading even before a table
            or (nxt is not None and (nxt.kind in ("step", "procedure", "bullet") or nxt.ends_sentence))
        ):
            unit.kind = "heading"

    lengths = get_tokenizer()([u.text for u in units], add_special_tokens=False)["input_ids"] if units else []
    for unit, ids in zip(units, lengths):
        unit.tokens = len(ids)
    return units


def _split_oversized(unit: Unit, size: int) -> list[Unit]:
    """Splits a unit longer than the chunk size at sentence ends, then at token boundaries."""
    pieces: list[str] = []
    current = ""
    for sentence in SENTENCE_SPLIT_RE.split(unit.text):
        candidate = f"{current} {sentence}".strip()
        if current and count_tokens(candidate) > size:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    result: list[str] = []
    tokenizer = get_tokenizer()
    for piece in pieces:
        if count_tokens(piece) <= size:
            result.append(piece)
            continue
        offsets = tokenizer(piece, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        for i in range(0, len(offsets), size):
            window = offsets[i:i + size]
            result.append(piece[window[0][0]:window[-1][1]].strip())

    parts = [Unit(text=t, kind=unit.kind if i == 0 else "text", page_start=unit.page_start, page_end=unit.page_end)
             for i, t in enumerate(result) if t]
    for part in parts:
        part.tokens = count_tokens(part.text)
    return parts


# ---------------------------------------------------------------- 3 + 4. packing with overlap
def _cut_type(units: list[Unit], k: int) -> str:
    """How good it is to cut just before units[k]."""
    unit = units[k]
    if units[k - 1].kind in ("heading", "procedure"):
        return "mid_sentence"  # never separate a heading or "To do X:" from what follows it
    if unit.kind == "heading":
        return "heading"
    if unit.kind == "procedure" or (unit.kind == "step" and FIRST_STEP_RE.match(unit.text)):
        return "procedure"
    if unit.kind in ("step", "bullet"):
        return "step"
    return "sentence" if units[k - 1].ends_sentence else "mid_sentence"


def _heading_level(text: str) -> int:
    """'3.6 Firewall' -> 2, '3.6.1 General' -> 3; unnumbered headings rank below numbered ones."""
    m = re.match(r"^((?:[A-H]|\d{1,2})(?:\.\d{1,2}){1,3})", text)
    return m.group(1).count(".") + 1 if m and NUMBERED_HEADING_RE.match(text) else 9


@dataclass
class _Span:
    start: int  # first unit (including overlap)
    new_from: int  # first unit not in the previous chunk
    end: int  # one past the last unit
    cut: str


def _pack(units: list[Unit], size: int, overlap: int) -> list[_Span]:
    prefix = [0]
    for u in units:
        prefix.append(prefix[-1] + u.tokens)

    def tokens(a: int, b: int) -> int:
        return prefix[b] - prefix[a]

    spans: list[_Span] = []
    start = new_from = 0
    n = len(units)
    while new_from < n:
        # Drop overlap units if they leave no room for the next new unit
        while start < new_from and tokens(start, new_from + 1) > size:
            start += 1

        j = new_from
        while j < n and tokens(start, j + 1) <= size:
            j += 1
        if j == n:
            spans.append(_Span(start, new_from, n, "segment_end"))
            break

        def rank(k: int) -> tuple[int, int, int]:
            # Best cut type first; among headings the higher level ("3.6" before "3.6.1", so a
            # section stays with its sub-sections); then the later, fuller cut
            cut = _cut_type(units, k)
            level = _heading_level(units[k].text) if cut == "heading" else 0
            return CUT_PRIORITY[cut], level, -k

        def fill_needed(k: int) -> float:
            return MIN_FILL_HEADING if _cut_type(units, k) == "heading" else MIN_FILL

        candidates = {k for k in range(new_from + 1, j + 1) if tokens(start, k) >= fill_needed(k) * size} | {j}
        best_k = min(candidates, key=rank)
        best_cut = _cut_type(units, best_k)
        # Never leave a heading or "To do X:" at the end: move it to the next chunk with its content
        while best_k - 1 > new_from and units[best_k - 1].kind in ("heading", "procedure"):
            best_k -= 1
            best_cut = _cut_type(units, best_k)
        spans.append(_Span(start, new_from, best_k, best_cut))

        # Overlap keeps a procedure connected across a cut; at a new topic it would only
        # mix the previous topic into this chunk, so none is added there.
        next_start = best_k
        if best_cut not in ("heading", "procedure"):
            while next_start - 1 >= new_from and tokens(next_start - 1, best_k) <= overlap:
                next_start -= 1
        start, new_from = next_start, best_k

    # A tiny last chunk is merged into the previous one if the result still fits
    if len(spans) >= 2:
        last, prev = spans[-1], spans[-2]
        if tokens(last.new_from, last.end) < TAIL_MERGE_SHARE * size and tokens(prev.start, last.end) <= size:
            spans[-2:] = [_Span(prev.start, prev.new_from, last.end, "segment_end_merged")]
    return spans


# ---------------------------------------------------------------- entry point
def chunk_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> tuple[list[Chunk], ChunkingReport]:
    if not 0 < chunk_size <= MAX_CHUNK_TOKENS:
        raise ValueError(f"chunk_size must be 1..{MAX_CHUNK_TOKENS}, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(f"chunk_overlap must be 0..{chunk_size - 1}, got {chunk_overlap}")

    chunking = f"{chunk_size}/{chunk_overlap}"
    chunks: list[Chunk] = []
    per_doc: Counter = Counter()
    cut_types: Counter = Counter()
    overlaps: list[int] = []
    n_segments = dropped = tail_merges = oversized = ending_with_title = 0

    for doc in docs:
        index = 0
        for segment in _segments(doc):
            n_segments += 1
            units: list[Unit] = []
            for unit in _build_units(segment):
                if unit.tokens > chunk_size:
                    oversized += 1
                    units.extend(_split_oversized(unit, chunk_size))
                else:
                    units.append(unit)
            if not units:
                continue

            section_at: list[str | None] = []
            section = None
            for unit in units:
                if unit.kind in ("heading", "procedure"):
                    section = unit.text.rstrip(":")
                section_at.append(section)

            for span in _pack(units, chunk_size, chunk_overlap):
                text = "\n".join(u.text for u in units[span.start:span.end])
                token_count = count_tokens(text)
                if token_count > chunk_size:  # cannot happen if unit counts are right; never store an oversized chunk
                    raise RuntimeError(f"{doc.document_id}: chunk of {token_count} tokens exceeds {chunk_size}")
                if token_count < MIN_CHUNK_TOKENS:
                    dropped += 1
                    continue
                if span.cut == "segment_end_merged":
                    tail_merges += 1
                cut_types[span.cut] += 1
                if span.start < span.new_from:
                    overlaps.append(sum(u.tokens for u in units[span.start:span.new_from]))

                chunk_units = units[span.start:span.end]
                if chunk_units[-1].kind in ("heading", "procedure"):
                    ending_with_title += 1
                chunks.append(Chunk(
                    chunk_id=f"{doc.document_id}_{index:04d}",
                    document_id=doc.document_id,
                    brand=doc.brand,
                    model=doc.model,
                    source=doc.source,
                    chunk_index=index,
                    text=text,
                    token_count=token_count,
                    page_start=min(u.page_start for u in chunk_units),
                    page_end=max(u.page_end for u in chunk_units),
                    content_type=segment[0].content_type,
                    section=section_at[span.new_from],
                    cleaner_version=CLEANER_VERSION,
                    chunking=chunking,
                ))
                per_doc[doc.document_id] += 1
                index += 1

    counts = [c.token_count for c in chunks] or [0]
    report = ChunkingReport(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        documents=len(docs),
        segments=n_segments,
        chunks=len(chunks),
        chunks_per_document=dict(per_doc),
        dropped_too_small=dropped,
        tail_merges=tail_merges,
        oversized_units_split=oversized,
        token_min=min(counts),
        token_median=int(statistics.median(counts)),
        token_max=max(counts),
        multi_page_chunks=sum(1 for c in chunks if c.page_end > c.page_start),
        ending_with_title=ending_with_title,
        avg_overlap_tokens=round(statistics.mean(overlaps), 1) if overlaps else 0.0,
        cut_types=dict(sorted(cut_types.items(), key=lambda kv: CUT_PRIORITY.get(kv[0], 9))),
    )
    return chunks, report
