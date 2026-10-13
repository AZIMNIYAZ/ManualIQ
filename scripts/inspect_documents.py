"""Step 2 check: loads every document in the manifest and prints statistics.

Run from the project root:
    python scripts/inspect_documents.py              # summary table
    python scripts/inspect_documents.py tenda_ax3000 61   # print one page's raw text
"""

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.document_loader import load_all_documents  # noqa: E402


def main() -> None:
    settings = get_settings()
    docs = load_all_documents(settings.documents_dir)

    if len(sys.argv) == 3:
        doc_id, page_no = sys.argv[1], int(sys.argv[2])
        doc = next(d for d in docs if d.document_id == doc_id)
        print(doc.pages[page_no - 1].text)
        return

    tok = AutoTokenizer.from_pretrained(settings.embedding_model)
    print(f"{'document_id':20} {'brand':8} {'pages':>5} {'empty':>5} {'chars':>8} {'tokens':>7} {'median/page':>11}")
    total_tokens = 0
    for d in docs:
        counts = [len(tok.encode(p.text, add_special_tokens=False)) for p in d.pages]
        empty = sum(1 for p in d.pages if len(p.text.strip()) < 30)
        total_tokens += sum(counts)
        print(
            f"{d.document_id:20} {d.brand:8} {len(d.pages):5} {empty:5} {d.text_length:8} "
            f"{sum(counts):7} {int(statistics.median(counts)):11}"
        )
    step = settings.chunk_size - settings.chunk_overlap
    print(f"\n{len(docs)} documents, {sum(len(d.pages) for d in docs)} pages, {total_tokens} tokens (raw, before cleaning)")
    print(f"Estimated chunks at {settings.chunk_size}/{settings.chunk_overlap}: ~{round(total_tokens / step)}")


if __name__ == "__main__":
    main()
