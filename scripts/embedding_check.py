"""Step 5 check: embeds every chunk and previews retrieval, without Qdrant.

This is a smoke test, not the evaluation (Step 11): read whether each retrieved
passage actually answers its question.

Run from the project root:
    python scripts/embedding_check.py --out embedding_check.txt
    python scripts/embedding_check.py --full --out embedding_check.txt   # complete top-1 chunks
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.chunker import chunk_documents  # noqa: E402
from app.services.document_loader import load_all_documents  # noqa: E402
from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.text_cleaner import clean_documents  # noqa: E402

# (question, brand the answer should come from); None = out of scope
QUESTIONS = [
    ("How do I set up port forwarding on the Netgear router?", "netgear"),
    ("How do I block services on my Netgear router?", "netgear"),
    ("What is the default login IP address of the Tenda router?", "tenda"),
    ("How do I remove a client from the blacklist on the Tenda router?", "tenda"),
    ("How do I use the Tether app to manage my TP-Link router?", "tplink"),
    ("How do I connect a device using the router's WPS PIN on TP-Link?", "tplink"),
    ("How do I enable the firewall on the ASUS router?", "asus"),
    ("How do I use AiDisk to share files from a USB drive on ASUS?", "asus"),
    ("How do I connect to Quick VPN from a Mac with the D-Link router?", "dlink"),
    ("How do I mount the D-Link router on a wall?", "dlink"),
    ("What is the weather forecast for tomorrow?", None),
    ("Who won the football World Cup in 2022?", None),
]
TOP = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="write the report to this file as UTF-8")
    parser.add_argument("--full", action="store_true",
                        help="print the complete top-1 chunk (with prefix), to judge whether it answers")
    args = parser.parse_args()
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")

    settings = get_settings()
    docs, _ = clean_documents(load_all_documents(settings.documents_dir))
    chunks, _ = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    service = get_embedding_service()

    # 1. Embed all chunks
    start = time.perf_counter()
    vectors = service.embed_documents([c.text for c in chunks])
    seconds = time.perf_counter() - start
    norms = np.linalg.norm(vectors, axis=1)
    print(f"Model: {service.model_version}, batch size {service.batch_size}")
    print(f"Embedded {len(chunks)} chunks ({settings.chunk_size}/{settings.chunk_overlap}) in {seconds:.1f}s "
          f"= {len(chunks) / seconds:.1f} chunks/s")
    print(f"Vectors: shape {vectors.shape}, norms {norms.min():.4f}-{norms.max():.4f}, "
          f"all finite: {bool(np.isfinite(vectors).all())}")

    # Search like the default /search will: regulatory chunks filtered out
    searchable = [i for i, c in enumerate(chunks) if c.content_type == "normal"]
    matrix = vectors[searchable]

    def search(query_vector):
        scores = matrix @ query_vector
        order = np.argsort(-scores)[:TOP]
        return [(chunks[searchable[i]], float(scores[i])) for i in order]

    # 2. Retrieval preview, with and without the query prefix
    results = {"with prefix": [], "without prefix": []}
    top1 = {"with prefix": {"in": [], "out": []}, "without prefix": {"in": [], "out": []}}
    for question, brand in QUESTIONS:
        print(f"\n{'=' * 100}\nQ: {question}   (expected: {brand or 'OUT OF SCOPE'})")
        for mode in ("with prefix", "without prefix"):
            if mode == "with prefix":
                q = service.embed_query(question).vector
            else:
                q = service.embed_documents([question])[0]
            hits = search(q)
            top1[mode]["in" if brand else "out"].append(hits[0][1])
            if brand:
                results[mode].append(any(c.brand == brand for c, _ in hits))
            ok = "" if not brand else ("  [right manual in top 3]" if results[mode][-1] else "  [MISSED]")
            print(f"  -- {mode}:{ok}")
            for rank, (c, score) in enumerate(hits, 1):
                pages = f"p.{c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "")
                preview = " ".join(c.text.split())[:150]
                print(f"     {rank}. {score:.3f} {c.chunk_id} {pages} [{c.section}]\n        {preview}")
            if args.full and mode == "with prefix":
                print("     FULL TOP-1 CHUNK:\n" + "\n".join("       | " + line for line in hits[0][0].text.splitlines()))

    # 3. Summary
    print(f"\n{'=' * 100}\nSUMMARY")
    for mode in ("with prefix", "without prefix"):
        hits = results[mode]
        ins, outs = top1[mode]["in"], top1[mode]["out"]
        print(f"  {mode:15}: right manual in top {TOP} for {sum(hits)}/{len(hits)} in-scope questions | "
              f"top-1 score in-scope: min {min(ins):.3f} median {statistics.median(ins):.3f} max {max(ins):.3f} | "
              f"out-of-scope: {', '.join(f'{s:.3f}' for s in outs)}")
    all_scores = (matrix @ service.embed_query(QUESTIONS[0][0]).vector)
    print(f"  measured score range for one query over all chunks: {all_scores.min():.3f} to {all_scores.max():.3f}")
    print("\nNow read the passages above: does the top result actually answer each question?")


if __name__ == "__main__":
    main()
