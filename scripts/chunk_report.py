"""Step 4 check: chunks every document and prints statistics.

Run from the project root:
    python scripts/chunk_report.py                  # report for CHUNK_SIZE/CHUNK_OVERLAP from .env
    python scripts/chunk_report.py --samples 5      # ...plus 5 random chunks to read
    python scripts/chunk_report.py --multipage 2    # ...plus 2 random chunks that span pages
    python scripts/chunk_report.py --compare        # chunk counts for several settings side by side
    python scripts/chunk_report.py --size 200 --overlap 40
    python scripts/chunk_report.py --samples 5 --out chunks.txt   # save as UTF-8 (don't use '>' in PowerShell 5.1)
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.services.chunker import chunk_documents  # noqa: E402
from app.services.document_loader import load_all_documents  # noqa: E402
from app.services.text_cleaner import clean_documents  # noqa: E402

COMPARE_SETTINGS = [(200, 40), (350, 0), (350, 30), (350, 60), (480, 80)]


def print_chunk(c) -> None:
    pages = f"p. {c.page_start}" if c.page_start == c.page_end else f"p. {c.page_start}-{c.page_end}"
    print(f"\n--- {c.chunk_id} | {c.brand} {c.model} | {pages} | {c.token_count} tokens | "
          f"{c.content_type} | section: {c.section!r}")
    print(c.text)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--multipage", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", help="write the report to this file as UTF-8 (avoids PowerShell's '>' "
                                      "turning ’ into ΓÇÖ)")
    args = parser.parse_args()
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")

    docs, _ = clean_documents(load_all_documents(settings.documents_dir))

    if args.compare:
        print(f"{'setting':>8} {'chunks':>6} {'median tok':>10} {'multi-page':>10} {'avg overlap':>11}  cut types")
        for size, overlap in COMPARE_SETTINGS:
            _, r = chunk_documents(docs, size, overlap)
            print(f"{size:>4}/{overlap:<3} {r.chunks:>6} {r.token_median:>10} {r.multi_page_chunks:>10} "
                  f"{r.avg_overlap_tokens:>11}  {r.cut_types}")
        return

    chunks, r = chunk_documents(docs, args.size, args.overlap)
    print(f"Chunking {r.chunk_size}/{r.chunk_overlap}: {r.documents} documents, {r.segments} segments -> {r.chunks} chunks")
    for doc_id, n in r.chunks_per_document.items():
        print(f"  {doc_id:20} {n:4}")
    print(f"tokens per chunk: min {r.token_min}, median {r.token_median}, max {r.token_max} (limit {r.chunk_size})")
    print(f"multi-page chunks: {r.multi_page_chunks} ({r.multi_page_chunks / max(r.chunks, 1):.0%})")
    print(f"chunks ending with a heading or procedure intro: {r.ending_with_title}")
    print(f"average overlap: {r.avg_overlap_tokens} tokens (setting {r.chunk_overlap})")
    print(f"dropped as too small: {r.dropped_too_small} | tail merges: {r.tail_merges} | "
          f"oversized units split: {r.oversized_units_split}")
    total = sum(r.cut_types.values())
    print("where chunks were cut: " + ", ".join(f"{k} {v} ({v / total:.0%})" for k, v in r.cut_types.items()))

    rng = random.Random(args.seed)
    for c in rng.sample(chunks, min(args.samples, len(chunks))):
        print_chunk(c)
    multi = [c for c in chunks if c.page_end > c.page_start]
    for c in rng.sample(multi, min(args.multipage, len(multi))):
        print_chunk(c)


if __name__ == "__main__":
    main()
