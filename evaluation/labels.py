"""Automatic checks of the evaluation labels against the manuals themselves.

- Evidence: the key phrase must appear (after normalisation) in the cleaned text of the
  stated PDF pages of that manual, and those pages must be indexed. A broken label is an
  error: every metric computed with it would be wrong.
- Chunks: the key phrase should fit inside one chunk of the current chunk settings that
  overlaps the pages; otherwise retrieval can never "find" it. This depends on the chunk
  size, so it is a warning, reported per configuration.
- Absence: the terms recorded for an unanswerable question are searched in the raw and the
  cleaned text of the listed manuals. Any hit is an error. Zero hits is a safeguard, not
  proof of absence: the manual-search record and the reviewer's own check are what count.
"""

from dataclasses import dataclass, field

from app.models.chunk import Chunk
from app.models.document import Document
from evaluation.dataset import Conversation, Item, Question, normalise


@dataclass
class LabelReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    absence: list[str] = field(default_factory=list)  # one line per unanswerable item: terms and hits
    checked_evidence: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _pages_text(doc: Document, first: int, last: int, raw: bool = False) -> str:
    pages = [p for p in doc.pages if first <= p.page_number <= last]
    return "\n".join((p.raw_text if raw else p.clean_text) or "" for p in pages)


def _check_item(item: Item, docs: dict[str, Document], chunks: dict[str, list[Chunk]] | None,
                report: LabelReport) -> None:
    for ev in item.evidence:
        report.checked_evidence += 1
        where = f"{item.id}: {ev.document_id} pp. {ev.pages[0]}-{ev.pages[1]}"
        doc = docs.get(ev.document_id)
        if doc is None:
            report.errors.append(f"{where}: unknown document_id")
            continue
        last_page = doc.pages[-1].page_number
        if ev.pages[1] > last_page:
            report.errors.append(f"{where}: the manual has only {last_page} pages")
            continue
        pages = [p for p in doc.pages if ev.pages[0] <= p.page_number <= ev.pages[1]]
        if not any(p.status == "indexed" for p in pages):
            report.errors.append(f"{where}: no page in the range is indexed (all excluded by the cleaner)")
            continue
        if any(p.content_type == "regulatory" for p in pages):
            report.warnings.append(f"{where}: regulatory page (filtered from search by default)")
        phrase = normalise(ev.key_phrase)
        if phrase not in normalise(_pages_text(doc, *ev.pages)):
            in_raw = phrase in normalise(_pages_text(doc, *ev.pages, raw=True))
            report.errors.append(f"{where}: key phrase {ev.key_phrase!r} not on these pages"
                                 + (" (it is in the raw text: changed by the cleaner?)" if in_raw else ""))
            continue
        if chunks is not None:
            candidates = [c for c in chunks.get(ev.document_id, [])
                          if c.page_start <= ev.pages[1] and c.page_end >= ev.pages[0]]
            if not any(phrase in normalise(c.text) for c in candidates):
                report.warnings.append(f"{where}: key phrase {ev.key_phrase!r} is split across chunks "
                                       "at this chunk size (retrieval cannot match it)")

    if item.absence is not None:
        manuals = list(docs) if item.absence.manuals == "all" else item.absence.manuals
        hits = []
        for doc_id in manuals:
            doc = docs.get(doc_id)
            if doc is None:
                report.errors.append(f"{item.id}: absence check names unknown manual {doc_id!r}")
                continue
            for term in item.absence.terms:
                t = normalise(term)
                pages = sorted({p.page_number for p in doc.pages
                                if t in normalise(p.raw_text) or t in normalise(p.clean_text or "")})
                if pages:
                    hits.append(f"{term!r} in {doc_id} pp. {pages[:8]}")
        scope = "all manuals" if item.absence.manuals == "all" else ", ".join(manuals)
        terms = ", ".join(repr(t) for t in item.absence.terms)
        if hits:
            report.errors.append(f"{item.id}: absence terms found: {'; '.join(hits)} - re-check the label")
            report.absence.append(f"{item.id}: {terms} in {scope} (raw + cleaned): FOUND {'; '.join(hits)}")
        else:
            report.absence.append(f"{item.id}: {terms} in {scope} (raw + cleaned): 0 hits")


def check_labels(questions: list[Question], conversations: list[Conversation], docs: list[Document],
                 chunks: list[Chunk] | None = None) -> LabelReport:
    """`docs` must be cleaned (clean_text/status set). `chunks` enables the per-configuration
    chunk check."""
    report = LabelReport()
    by_id = {d.document_id: d for d in docs}
    by_doc: dict[str, list[Chunk]] | None = None
    if chunks is not None:
        by_doc = {}
        for c in chunks:
            by_doc.setdefault(c.document_id, []).append(c)

    ids = [q.id for q in questions] + [c.id for c in conversations] + [t.id for c in conversations for t in c.turns]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        report.errors.append(f"duplicate ids: {duplicates}")
    brands = {d.brand for d in docs}
    for item in [*questions, *(t for c in conversations for t in c.turns)]:
        if item.brand is not None and item.brand not in brands:
            report.errors.append(f"{item.id}: unknown brand {item.brand!r}")
        _check_item(item, by_id, by_doc, report)
    return report
