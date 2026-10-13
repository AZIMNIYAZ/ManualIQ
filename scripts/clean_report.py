"""Step 3 check: cleans every document and prints what the cleaner did.

Run from the project root:
    python scripts/clean_report.py                     # full report for all documents
    python scripts/clean_report.py tplink_archer_ax55  # report for one document
    python scripts/clean_report.py tplink_archer_ax55 61   # raw vs clean text of one page

The full report (including every repair) is also saved as JSON to
evaluation/results/cleaning/<document_id>.json.
"""

import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.services.document_loader import load_all_documents  # noqa: E402
from app.services.text_cleaner import clean_document  # noqa: E402

OUT_DIR = Path("evaluation/results/cleaning")


def show_page(doc, page_no: int) -> None:
    page = doc.pages[page_no - 1]
    print(f"=== {doc.document_id} page {page_no}: status={page.status} content_type={page.content_type} "
          f"reason={page.reason} flags={list(page.flags)}")
    print("\n--- RAW ---------------------------------------------------------------")
    print(page.raw_text)
    print("\n--- CLEAN -------------------------------------------------------------")
    print(page.clean_text if page.clean_text is not None else "(excluded: no clean text)")


def print_report(report) -> None:
    r = report
    print(f"\n{'=' * 78}\n{r.document_id}  (cleaner {r.cleaner_version})")
    print(f"pages: {r.pages_total} total, {r.pages_indexed} indexed, {r.pages_excluded} excluded | "
          f"chars: {r.chars_raw} raw -> {r.chars_clean} clean ({r.chars_clean / r.chars_raw:.0%})")

    print(f"\nEXCLUDED ({len(r.excluded)})")
    for n in r.excluded:
        print(f"  p{n.page:<4} {n.label:12} {n.preview}")

    print(f"\nREGULATORY - indexed, filtered by default ({len(r.regulatory)})")
    for n in r.regulatory:
        print(f"  p{n.page:<4} {n.preview}")

    print(f"\nFLAGGED FOR REVIEW - kept ({len(r.flagged)})")
    for n in r.flagged:
        print(f"  p{n.page:<4} {n.label:20} {n.preview}")

    print(f"\nHEADER/FOOTER LINES REMOVED ({len(r.header_lines)} patterns)")
    for line, count in r.header_lines.items():
        print(f"  {count:4}x  {line[:70]}")

    print(f"\nREPAIR COUNTS: {r.repair_counts}")
    by_kind: dict[str, Counter] = {}
    for rep in r.repairs:
        if rep.kind != "bullet":
            by_kind.setdefault(rep.kind, Counter())[f"{rep.original!r} -> {rep.fixed!r}"] += 1
    for kind, counter in by_kind.items():
        print(f"  {kind}: " + "; ".join(f"{k} x{v}" for k, v in counter.most_common(15))
              + (f"; ... (+{len(counter) - 15} more)" if len(counter) > 15 else ""))


def main() -> None:
    docs = load_all_documents(get_settings().documents_dir)
    args = sys.argv[1:]
    if args:
        docs = [d for d in docs if d.document_id == args[0]]
        if not docs:
            sys.exit(f"Unknown document_id: {args[0]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    for doc in docs:
        cleaned, report = clean_document(doc)
        if len(args) == 2:
            show_page(cleaned, int(args[1]))
            return
        print_report(report)
        (OUT_DIR / f"{doc.document_id}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        totals.update(pages=report.pages_total, indexed=report.pages_indexed, excluded=report.pages_excluded,
                      regulatory=len(report.regulatory), flagged=len(report.flagged),
                      raw=report.chars_raw, clean=report.chars_clean)

    if len(docs) > 1:
        print(f"\n{'=' * 78}\nTOTAL: {totals['pages']} pages, {totals['indexed']} indexed "
              f"({totals['regulatory']} regulatory), {totals['excluded']} excluded, {totals['flagged']} flagged | "
              f"chars {totals['raw']} -> {totals['clean']}")
        print(f"Full reports: {OUT_DIR}/")


if __name__ == "__main__":
    main()
