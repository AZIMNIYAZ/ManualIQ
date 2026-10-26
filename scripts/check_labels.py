"""Step 11: checks the evaluation labels against the manuals (free, no services needed).

Run from the project root:
    python scripts/check_labels.py
    python scripts/check_labels.py --size 200 --overlap 40   # chunk check for another configuration

Errors (exit code 1): a key phrase not on its pages, an excluded page, an absence term that
is actually in a manual, duplicate ids. Warnings: phrases split across chunks at this chunk
size, regulatory evidence pages.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.services.chunker import chunk_documents  # noqa: E402
from app.services.document_loader import load_all_documents  # noqa: E402
from app.services.text_cleaner import clean_documents  # noqa: E402
from evaluation.dataset import (HOLDOUT_FILE, dataset_hash, fact_label, load_conversations,  # noqa: E402
                                load_questions, normalise)
from evaluation.labels import check_labels  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--out")
    parser.add_argument("--holdout", action="store_true", help="check the held-out questions (labels only, no system run)")
    parser.add_argument("--review", help="write a Markdown review sheet (evidence excerpts, PDF pages) to this file")
    args = parser.parse_args()
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")

    if args.holdout:
        questions, conversations = load_questions(HOLDOUT_FILE), []
    else:
        questions, conversations = load_questions(), load_conversations()
    docs, _ = clean_documents(load_all_documents(settings.documents_dir))
    chunks, _ = chunk_documents(docs, args.size, args.overlap)
    report = check_labels(questions, conversations, docs, chunks)

    turns = sum(len(c.turns) for c in conversations)
    print(f"Dataset {dataset_hash(HOLDOUT_FILE) if args.holdout else dataset_hash()}"
          f"{' (HELD-OUT)' if args.holdout else ''}: {len(questions)} questions "
          f"({', '.join(f'{k} {v}' for k, v in Counter(q.category for q in questions).items())}), "
          f"{len(conversations)} conversations ({turns} turns)")
    print(f"Evidence locations checked: {report.checked_evidence} | chunk check at {args.size}/{args.overlap}")
    print("\nABSENCE CHECKS (a safeguard, not proof: confirm with Ctrl+F in the PDFs)")
    for line in report.absence:
        print("  " + line)
    print(f"\nWARNINGS ({len(report.warnings)})")
    for w in report.warnings:
        print("  " + w)
    print(f"\nERRORS ({len(report.errors)})")
    for e in report.errors:
        print("  " + e)
    print("\nLabels OK" if report.ok else "\nLabels have errors")
    if args.review:
        write_review(Path(args.review), questions, conversations, docs, report)
        print(f"Review sheet: {args.review}")
    sys.exit(0 if report.ok else 1)


def excerpt(doc, first: int, last: int, phrase: str, width: int = 160) -> str:
    """The cleaned text around the key phrase, to compare with the PDF quickly."""
    for page in doc.pages:
        if first <= page.page_number <= last and page.clean_text:
            flat = " ".join(page.clean_text.split())
            i = normalise(flat).find(normalise(phrase))
            if i >= 0:  # normalise keeps lengths close enough for an excerpt window
                start = max(0, i - width // 2)
                return f"p. {page.page_number}: …{flat[start:start + width + len(phrase)]}…"
    return "(phrase spans a page break)"


def write_review(path: Path, questions, conversations, docs, report) -> None:
    by_id = {d.document_id: d for d in docs}
    lines = ["# Label review", "",
             "For each item, open the PDF at the page shown (the page number in the PDF viewer's toolbar, "
             "not the number printed on the page) and check: the evidence answers the question; the key "
             "facts are what a correct answer must say; nothing else in the manual answers it better "
             "(if it does, it must be added as evidence). Unanswerable: search the PDF yourself (Ctrl+F) "
             "for the terms listed, plus any other wording you can think of.", ""]

    def item_lines(item, extra=""):
        out = [f"- [ ] **{item.id}**{extra}: {item.question}",
               f"  - brand filter expected: {item.brand or 'none'} | answerable: {item.answerable}"]
        for ev in item.evidence:
            doc = by_id[ev.document_id]
            out.append(f"  - evidence: `{doc.source}` pp. {ev.pages[0]}-{ev.pages[1]}, key phrase "
                       f"\"{ev.key_phrase}\"  \n    {excerpt(doc, *ev.pages, ev.key_phrase)}")
        if item.key_facts:
            out.append("  - key facts: " + "; ".join(fact_label(f) for f in item.key_facts))
        if item.must_not_contain:
            out.append("  - must not contain: " + "; ".join(item.must_not_contain))
        if item.absence:
            out.append(f"  - absence: searched {item.absence.terms} in {item.absence.manuals}")
        if getattr(item, "expected_query", None):
            out.append(f"  - expected rewrite: \"{item.expected_query}\" (must contain: "
                       + "; ".join(fact_label(t) for t in item.required_terms) + ")")
        if item.note:
            out.append(f"  - note: {item.note}")
        return out

    category = None
    for item in questions:
        if item.category != category:
            category = item.category
            lines += ["", f"## {category}", ""]
        lines += item_lines(item)
    lines += ["", "## conversations", ""]
    for conv in conversations:
        lines.append(f"### {conv.id}: {conv.tests}")
        for n, turn in enumerate(conv.turns, 1):
            lines += item_lines(turn, f" (turn {n})")
        lines.append("")
    lines += ["## Absence checks (automatic)", ""] + [f"- {a}" for a in report.absence]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
