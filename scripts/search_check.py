"""Step 7 check: the Step 5 questions through the real search (Qdrant), with and
without brand detection.

Run from the project root (needs Qdrant running and rag_350_60 ingested):
    python scripts/search_check.py --out search_check.txt
    python scripts/search_check.py --full --out search_check.txt   # complete top-1 chunks
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.core.config import get_settings  # noqa: E402
from app.database.qdrant import create_qdrant_client  # noqa: E402
from app.models.search import SearchRequest  # noqa: E402
from app.services.brand_detection import alias_table_from_manifest  # noqa: E402
from app.services.embedding_service import get_embedding_service  # noqa: E402
from app.services.retrieval_service import search  # noqa: E402

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
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.out:
        sys.stdout = open(args.out, "w", encoding="utf-8")

    settings = get_settings()
    client, embedder = create_qdrant_client(settings), get_embedding_service()
    aliases = alias_table_from_manifest(settings.documents_dir)

    totals = {True: [], False: []}
    regulatory_seen = 0
    for question, brand in QUESTIONS:
        print(f"\n{'=' * 100}\nQ: {question}   (expected brand: {brand})")
        for detect in (False, True):
            r = search(client, embedder, settings, SearchRequest(query=question), aliases, detect_brand=detect)
            totals[detect].append(r.timings_ms["total"])
            regulatory_seen += sum(x.content_type == "regulatory" for x in r.results)
            top_brand_ok = r.results[0].brand == brand
            label = "with brand detection" if detect else "no brand filter    "
            print(f"  -- {label}: filter={r.filters.brands or 'all'} ({r.filters.brand_source}) | "
                  f"rank-1 brand {'OK' if top_brand_ok else 'WRONG'} | {r.timings_ms['total']:.0f} ms")
            for x in r.results[:3]:
                print(f"     {x.rank}. {x.score:.3f} {x.citation} [{x.section}]")
            if args.full and detect:
                print("     FULL TOP-1 CHUNK:\n" + "\n".join("       | " + line for line in r.results[0].text.splitlines()))

    print(f"\n{'=' * 100}\nregulatory chunks in default results: {regulatory_seen} (expected 0)")
    for detect in (False, True):
        print(f"latency {'with' if detect else 'without'} detection: median {statistics.median(totals[detect]):.0f} ms, "
              f"max {max(totals[detect]):.0f} ms")
    print("Read the top results: does rank 1 answer each question?")


if __name__ == "__main__":
    main()
